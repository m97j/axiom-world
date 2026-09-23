"""Paired v1 task evaluation before publication; no training or Hub writes.

Uses frozen v1 data and archived evaluator dependencies. This is a current-stack
precision comparison, not a reproduction of the historical execution environment.
No automatic equivalence or publication approval is issued.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

LEGACY = "ff984e4bef63a2a04d3606b01e01da4152a91a21"
DATA_REVISION = "20a669ce2782546572dc6445e8b5cba62577ac34"
FREEZE = "sha256:3cdcbc30c99e492c45e126cd22feb903219b264f50cbd004ab01b734bd83b3b0"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def summarize_pairs(reference, candidate):
    if not reference or len(reference) != len(candidate):
        raise ValueError("Missing or unequal paired rows")
    result = {}
    for a, b in zip(reference, candidate, strict=True):
        if (a["id"], a["suite"], a["prompt_sha256"]) != (b["id"], b["suite"], b["prompt_sha256"]):
            raise ValueError("Pair identity or rendered prompt/tokenization differs")
        for row in (a, b):
            if row["verdict"]["status"] not in ("passed", "failed"):
                raise ValueError("Infrastructure/indeterminate verdict; do not interpret as task failure")
        r = result.setdefault(a["suite"], {"episodes": 0, "reference_pass": 0, "candidate_pass": 0,
            "regressions": 0, "improvements": 0, "identical_text": 0,
            "reference_truncated": 0, "candidate_truncated": 0,
            "reference_schema_failed": 0, "candidate_schema_failed": 0})
        ap, bp = a["verdict"]["status"] == "passed", b["verdict"]["status"] == "passed"
        r["episodes"] += 1
        r["reference_pass"] += ap
        r["candidate_pass"] += bp
        r["regressions"] += ap and not bp
        r["improvements"] += bp and not ap
        r["identical_text"] += a["prediction"] == b["prediction"]
        for label, row in (("reference", a), ("candidate", b)):
            r[label + "_truncated"] += row["truncated"]
            r[label + "_schema_failed"] += row["verdict"]["reason_code"].startswith("gate_failed:")
    for r in result.values():
        r["reference_pass_rate"] = r["reference_pass"] / r["episodes"]
        r["candidate_pass_rate"] = r["candidate_pass"] / r["episodes"]
        r["pass_rate_delta"] = r["candidate_pass_rate"] - r["reference_pass_rate"]
    return result


def worker(request, arm):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # PYTHONPATH points exclusively to the archived v1 src directory.
    from axiom_world.core.config_loader import resolve
    from axiom_world.data.bundle import build_data_bundle
    from axiom_world.models.builder import build_for_inference
    from axiom_world.verifiers.hybrid import default_playworld_verifier

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    out = Path(request["output"])
    config, _, _ = resolve("configs/experiments/eval_playworld.yaml", [])
    if config.runtime.precision != "bf16" or config.evaluation.few_shot_k:
        raise ValueError("Unexpected archived v1 evaluation profile")
    freeze = read(out / "data/freeze_manifest.json")
    bundles = {name: build_data_bundle(out / "data" / f"{name}.jsonl", "evaluation",
               expected_fingerprint=entry["fingerprint"]) for name, entry in freeze["suites"].items()}
    if len(bundles) != 5 or any(len(b.records) != 300 for b in bundles.values()):
        raise ValueError("Expected exactly five frozen suites of 300 episodes")
    verifier = default_playworld_verifier()
    if request["data_only"]:
        write(out / "data_verified.json", {"freeze": FREEZE, "episodes": 1500,
              "suites": {k: b.fingerprint for k, b in bundles.items()}})
        return
    if arm == "reference":
        # Verify both tokenizers render identical task inputs BEFORE GPU allocation.
        original_tok = AutoTokenizer.from_pretrained(config.model.repo_id, revision=config.model.revision)
        candidate_tok = AutoTokenizer.from_pretrained(request["model"], local_files_only=True)
        for tok in (original_tok, candidate_tok):
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
        if (original_tok.pad_token_id, original_tok.eos_token_id,
            original_tok.convert_tokens_to_ids("<|im_end|>")) != (
            candidate_tok.pad_token_id, candidate_tok.eos_token_id,
            candidate_tok.convert_tokens_to_ids("<|im_end|>")):
            raise ValueError("Tokenizer termination/padding contract differs")
        for bundle in bundles.values():
            for record in bundle.records:
                message = [{"role": "user", "content": "\n".join(m.content for m in record.prompt)}]
                texts = [tok.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                         + config.evaluation.opener_seed for tok in (original_tok, candidate_tok)]
                ids = [tok(text, add_special_tokens=False)["input_ids"]
                       for tok, text in zip((original_tok, candidate_tok), texts, strict=True)]
                if ids[0] != ids[1]:
                    raise ValueError(f"Tokenizer prompt mismatch before model loading: {record.id}")
        print("[preflight] All 1500 task prompt token sequences match", flush=True)
        model, tokenizer = build_for_inference(config, request["adapter"])
    else:
        model = AutoModelForCausalLM.from_pretrained(request["model"], dtype=torch.float32,
            device_map="cuda", attn_implementation="sdpa", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(request["model"], local_files_only=True)
        if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise ValueError("Candidate did not load in FP32")
    tokenizer.padding_side = "left"
    model.eval()
    stop_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")})
    if any(not isinstance(x, int) or x < 0 for x in stop_ids):
        raise ValueError("Invalid termination token")
    reference = None
    if arm == "candidate":
        reference = [json.loads(line) for line in (out / "reference.jsonl").read_text(encoding="utf-8").splitlines()]
    count = 0
    with (out / f"{arm}.jsonl").open("x", encoding="utf-8") as stream, torch.inference_mode():
        for name, bundle in bundles.items():
            for start in range(0, len(bundle.records), request["batch_size"]):
                records = bundle.records[start:start + request["batch_size"]]
                texts = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": "\n".join(m.content for m in r.prompt)}],
                    tokenize=False, add_generation_prompt=True) + config.evaluation.opener_seed for r in records]
                encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
                hashes = [digest(ids) for ids in encoded]
                if reference is not None:
                    for offset, (r, h) in enumerate(zip(records, hashes, strict=True)):
                        previous = reference[count + offset]
                        if (r.id, name, h) != (previous["id"], previous["suite"], previous["prompt_sha256"]):
                            raise ValueError("Candidate prompt differs from archived reference; stop comparison")
                inputs = tokenizer(texts, return_tensors="pt", padding=True,
                                   add_special_tokens=False, return_token_type_ids=False).to(model.device)
                generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids)
                rows = generated[:, inputs["input_ids"].shape[1]:].cpu().tolist()
                predictions = tokenizer.batch_decode(rows, skip_special_tokens=True)
                for r, ids, text, h in zip(records, rows, predictions, hashes, strict=True):
                    verdict = verifier.verify(text, {"scenario": r.scenario}).model_dump(mode="json")
                    trace = {"id": r.id, "suite": name, "prompt_sha256": h, "prediction": text,
                             "truncated": len(ids) == 1024 and not any(x in stop_ids for x in ids),
                             "generated_ids": ids, "verdict": verdict}
                    stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
                stream.flush()
                count += len(records)
                print(f"[{arm}] {name}: {start + len(records)}/300; total {count}/1500", flush=True)
    write(out / f"{arm}_complete.json", {"episodes": count, "gpu": torch.cuda.get_device_name(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--worker", choices=["reference", "candidate"])
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(read(args.request), args.worker)
        return
    if args.output is None or (not args.data_only and args.release is None) or args.batch_size < 1:
        parser.error("Need a new --output, --release for evaluation, and positive batch size")
    from huggingface_hub import hf_hub_download

    from axiom_world.models.champion_release import inventory
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    archive = subprocess.check_output(["git", "-C", str(root), "archive", LEGACY,
        "src", "configs", "data/eval_suites/freeze_manifest.json"])
    legacy = out / "legacy"
    legacy.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(legacy, filter="data")
    data = out / "data"
    data.mkdir()
    frozen = read(legacy / "data/eval_suites/freeze_manifest.json")
    if frozen["manifest_fingerprint"] != FREEZE:
        raise ValueError("Archived freeze differs")
    for name in ("freeze_manifest.json", *(f"{s}.jsonl" for s in frozen["suites"])):
        cached = hf_hub_download("m97j/aw-playworld", "eval_suites/v1/" + name,
                                repo_type="dataset", revision=DATA_REVISION)
        (data / name).write_bytes(Path(cached).read_bytes())
    if read(data / "freeze_manifest.json") != frozen:
        raise ValueError("Dataset freeze differs from archived v1")
    request = {"output": str(out), "batch_size": args.batch_size, "data_only": args.data_only,
               "legacy_code": LEGACY, "data_revision": DATA_REVISION, "freeze": FREEZE,
               "scope": "Paired current-stack comparison; historical runtime not reproduced",
               "max_new_tokens": 1024, "attention": "sdpa", "seed": 42,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if not args.data_only:
        release = args.release.resolve()
        receipt = read(release / "verified.json")
        stage = release / "model"
        if (receipt.get("status") != "verified" or inventory(stage) != receipt["files"]
                or not receipt["verification"]["standalone"]["passed"]
                or receipt["verification"]["dtype"] != "float32"):
            raise ValueError("Release must be the verified, unchanged FP32 payload")
        request.update(model=str(stage), adapter=str(stage / "adapter"),
                       release_receipt_sha256=hashlib.sha256((release / "verified.json").read_bytes()).hexdigest())
    write(out / "request.json", request)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(legacy / "src"), PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    for arm in (("reference",) if args.data_only else ("reference", "candidate")):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", arm,
            "--request", str(out / "request.json")], cwd=legacy, env=env, check=True)
    if args.data_only:
        print("Frozen data validated; no model loaded")
        return
    if inventory(stage) != receipt["files"]:
        raise ValueError("Release changed during evaluation")
    rows = {arm: [json.loads(s) for s in (out / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines()]
            for arm in ("reference", "candidate")}
    if any(len(r) != 1500 for r in rows.values()):
        raise ValueError("Incomplete evaluation")
    result = {"status": "completed_requires_review", "request": request,
              "suites": summarize_pairs(rows["reference"], rows["candidate"]),
              "publication_approved": False}
    write(out / "comparison.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
