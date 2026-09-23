"""Local FP32 export and fresh-process verification; no Hub writes.

The probes are engineering checks, not a reproduction of the v1 benchmark.
GPU/model imports are lazy so release planning remains CPU/lightweight.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

from axiom_world.core.fingerprints import fingerprint_file

# Fixed before execution; failures require investigation, never automatic relaxation.
GATES = {"max_abs": 0.5, "mean_abs": 0.03, "top1_agreement": 0.99,
         "generation_agreement": 0.90}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def inventory(root: Path) -> list[dict]:
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Release payload must not contain symlinks")
    return [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size,
             "sha256": fingerprint_file(p)} for p in sorted(root.rglob("*")) if p.is_file()]


FP32_GATES = {"max_abs": 0.001, "mean_abs": 0.0001, "top1_agreement": 0.99,
              "generation_agreement": 0.90}


def compare_logits(reference, actual, reference_generations, generations, *, gates=None) -> dict:
    import torch

    gates = GATES if gates is None else gates
    if set(reference) != set(actual) or not reference or not reference_generations:
        raise ValueError("Missing probe outputs")
    maximum, total, count, matches, positions = 0.0, 0.0, 0, 0, 0
    for key, ref in reference.items():
        value = actual[key]
        if ref.shape != value.shape or not torch.isfinite(ref).all() or not torch.isfinite(value).all():
            raise ValueError("Non-finite or incompatible probe logits")
        diff = (ref.float() - value.float()).abs()
        maximum = max(maximum, diff.max().item())
        total += diff.sum().item()
        count += diff.numel()
        matches += (ref.argmax(-1) == value.argmax(-1)).sum().item()
        positions += ref.shape[0]
    result = {"max_abs": maximum, "mean_abs": total / count,
              "top1_agreement": matches / positions,
              "generation_agreement": sum(a == b for a, b in zip(
                  reference_generations, generations, strict=True)) / len(reference_generations)}
    result["passed"] = (result["max_abs"] <= gates["max_abs"]
                        and result["mean_abs"] <= gates["mean_abs"]
                        and result["top1_agreement"] >= gates["top1_agreement"]
                        and result["generation_agreement"] >= gates["generation_agreement"])
    return result


def _probe(model, tokenizer, probes):
    import torch

    logits, generations = {}, []
    model.eval()
    with torch.inference_mode():
        for i, probe in enumerate(probes):
            text = (tokenizer.apply_chat_template(probe["messages"], tokenize=False,
                    add_generation_prompt=True) if "messages" in probe else probe["text"])
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(model.device)
            if inputs.input_ids.shape[1] > 1024:
                raise ValueError("Probe exceeds 1024 tokens; do not silently truncate")
            logits[str(i)] = model(**inputs).logits[0].float().cpu().contiguous()
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
            generations.append(output[0, inputs.input_ids.shape[1]:].cpu().tolist())
    return logits, generations


def _check_saved_adapter(model, adapter: Path) -> None:
    """Attest all loaded adapter tensors, including both full saved modules."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors import safe_open

    loaded = get_peft_model_state_dict(model)
    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(loaded):
            raise ValueError("Loaded adapter tensor keys differ from checkpoint")
        for key in saved.keys():  # noqa: SIM118 - safetensors handle is not a dict
            view, tensor = saved.get_slice(key), loaded[key]
            if list(tensor.shape) != view.get_shape():
                raise ValueError(f"Adapter shape mismatch: {key}")
            for start in range(0, tensor.shape[0], 256):
                actual = tensor[start:start + 256].detach().cpu()
                expected = view[start:start + 256].to(dtype=actual.dtype)
                if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
                    raise ValueError(f"Adapter load mismatch: {key}")


def _check_saved_modules(model, adapter: Path) -> None:
    import torch
    from safetensors import safe_open

    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as saved:
        for name, module in [("embed_tokens", model.get_input_embeddings()),
                             ("lm_head", model.get_output_embeddings())]:
            keys = [key for key in saved.keys() if key.endswith(f".{name}.weight")]  # noqa: SIM118
            if len(keys) != 1:
                raise ValueError(f"Missing/ambiguous saved module: {name}")
            view = saved.get_slice(keys[0])
            if list(module.weight.shape) != view.get_shape():
                raise ValueError(f"Saved module shape mismatch: {name}")
            for start in range(0, module.weight.shape[0], 256):
                actual = module.weight[start:start + 256].detach().cpu()
                if not torch.equal(actual, view[start:start + 256].to(actual.dtype)):
                    raise ValueError(f"Merged saved module mismatch: {name}")


def merge_worker(request: dict) -> None:
    import torch
    from peft import PeftModel
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stage, scratch, adapter = map(Path, (request["stage"], request["scratch"], request["adapter"]))
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None or not tokenizer.chat_template:
        raise ValueError("Champion tokenizer must include pad token and chat template")
    if request.get("dtype") != "float32":
        raise ValueError("This release path requires explicit dtype=float32")
    torch.set_float32_matmul_precision("highest")
    base = AutoModelForCausalLM.from_pretrained(
        request["base"], revision=request["revision"], dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map=request["device"], trust_remote_code=False)
    if getattr(base.config, "quantization_config", None):
        raise ValueError("Quantized base is not a BF16 release source")
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
    _check_saved_adapter(model, adapter)
    # Explicit export decoding contract. Raw tokenizer bytes remain unchanged.
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos is None or tokenizer.convert_ids_to_tokens(eos) != "<|im_end|>":
        raise ValueError("Missing chat termination token")
    model.generation_config.eos_token_id = list(dict.fromkeys([tokenizer.eos_token_id, eos]))
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.do_sample = False
    print("[probe] BF16-base / PEFT default promotion reference", flush=True)
    historical, historical_gen = _probe(model, tokenizer, request["probes"])
    model.float()
    print("[probe] FP32 adapter reference", flush=True)
    before, before_gen = _probe(model, tokenizer, request["probes"])
    precision = {"reference": "BF16-base PEFT default promotion, not a replay of the historical evaluator",
                 "scope": "Engineering probes only; historical benchmark equivalence unverified",
                 "historical_vs_fp32_adapter": compare_logits(
                     historical, before, historical_gen, before_gen)}
    write_json(scratch / "precision.json", precision)
    print("[merge] FP32, safe_merge=True", flush=True)
    merged = model.merge_and_unload(safe_merge=True, progressbar=False).eval()
    # Preserve actual trained module sharing, rather than the base config's assumption.
    merged.config.tie_word_embeddings = (
        merged.get_input_embeddings().weight.data_ptr()
        == merged.get_output_embeddings().weight.data_ptr())
    if any("lora_" in key or "modules_to_save" in key for key in merged.state_dict()):
        raise ValueError("PEFT tensors survived merge")
    _check_saved_modules(merged, adapter)
    after, after_gen = _probe(merged, tokenizer, request["probes"])
    report = compare_logits(before, after, before_gen, after_gen, gates=FP32_GATES)
    precision["historical_vs_fp32_merged"] = compare_logits(
        historical, after, historical_gen, after_gen)
    write_json(scratch / "precision.json", precision)
    print(json.dumps({"merge": report, "precision": precision}, indent=2), flush=True)
    report["runtime"] = {"device": request["device"], "torch_cuda": torch.version.cuda,
                         "gpu_name": torch.cuda.get_device_name() if request["device"] == "cuda" else None,
                         "attention": "sdpa", "reference": "BF16-loaded base and PEFT adapter promoted together to FP32",
                         "dtype": "float32", "matmul_precision": "highest", "gates": FP32_GATES}
    write_json(scratch / "merge.json", report)
    if not report["passed"]:
        raise ValueError("Adapter-versus-merged gate failed; artifacts preserved for diagnosis")
    if any(p.dtype != torch.float32 for p in merged.parameters() if p.is_floating_point()):
        raise ValueError("Merged model is not wholly FP32")
    merged.save_pretrained(stage, safe_serialization=True, max_shard_size="5GB")
    save_file(after, scratch / "merged_logits.safetensors")
    write_json(scratch / "generations.json", after_gen)
    del model, base, merged
    gc.collect()


def reload_worker(request: dict) -> None:
    # Deliberately no PEFT import: block implicit adapter fallback as well.
    import importlib.abc

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class NoPeft(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "peft" or fullname.startswith("peft."):
                raise ImportError("Standalone verification forbids PEFT")

    sys.meta_path.insert(0, NoPeft())
    stage, scratch = Path(request["stage"]), Path(request["scratch"])
    if (stage / "adapter_config.json").exists():
        raise ValueError("Root adapter config would trigger automatic adapter loading")
    tokenizer = AutoTokenizer.from_pretrained(stage, local_files_only=True, trust_remote_code=False)
    if request.get("dtype") != "float32":
        raise ValueError("Standalone request must specify float32")
    torch.set_float32_matmul_precision("highest")
    model, loading = AutoModelForCausalLM.from_pretrained(
        stage, dtype=torch.float32, device_map=request["device"],
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False,
        output_loading_info=True)
    if any(loading.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]):
        raise ValueError(f"Standalone state-dict load was incomplete: {loading}")
    if getattr(model.config, "quantization_config", None):
        raise ValueError("Unexpected quantization")
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError("Standalone model is not wholly FP32")
    _check_saved_modules(model, stage / "adapter")
    actual, generations = _probe(model, tokenizer, request["probes"])
    result = compare_logits(load_file(scratch / "merged_logits.safetensors"), actual,
                            json.loads((scratch / "generations.json").read_text()), generations)
    # Same serialized FP32 weights, device and backend: require exact replay.
    result["passed"] = result["max_abs"] == 0 and result["generation_agreement"] == 1
    result["dtype"] = "float32"
    result["offline"] = True
    result["peft_imported"] = any(k == "peft" or k.startswith("peft.") for k in sys.modules)
    result["passed"] = result["passed"] and not result["peft_imported"]
    write_json(scratch / "standalone.json", result)
    if not result["passed"]:
        raise ValueError("Fresh-process standalone verification failed")


def run_workers(request: dict) -> dict:
    scratch = Path(request["scratch"])
    scratch.mkdir(parents=True, exist_ok=False)
    request_path = scratch / "request.json"
    write_json(request_path, request)
    for mode in ("merge", "reload"):
        env = os.environ.copy()
        if mode == "reload":
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        subprocess.run([sys.executable, "-m", __name__, mode, str(request_path)],
                       check=True, env=env)
    return {"gates": FP32_GATES, "dtype": "float32",
            "historical_comparison_gates": GATES,
            "precision": json.loads((scratch / "precision.json").read_text()), "probe_count": len(request["probes"]),
            "merge": json.loads((scratch / "merge.json").read_text()),
            "standalone": json.loads((scratch / "standalone.json").read_text())}


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    {"merge": merge_worker, "reload": reload_worker}[sys.argv[1]](request)
