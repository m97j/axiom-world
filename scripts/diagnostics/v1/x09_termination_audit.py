#!/usr/bin/env python
"""x09: Termination audit — diagnose EOS non-emission after B1/B2 SFT.

v0.6.1-NATIVE REWRITE. The previous draft targeted the pre-v0.1.0 codebase
(`load_config`, `data.SFTProcessor._row_to_example`) which no longer exists.
This version uses only v0.6.1 surfaces:

  - axiom_world.core.config_loader.resolve
  - axiom_world.models.builder / training.factory.build_trainer
  - axiom_world.verifiers.general.extract_final_answer / ExactAnswerVerifier
  - run artifacts: artifacts/evaluation_summary.json,
                   artifacts/evaluation_<suite>.jsonl (fields: id, suite,
                   scenario_family_id, prediction, verdict{reason_code,...})

Hypothesis T (termination bug):
  models/builder.py sets tokenizer.pad_token = tokenizer.eos_token when pad
  is undefined. If TRL's SFT label pipeline (assistant_only_loss path or the
  LM collator) masks pad positions via `labels[input_ids == pad_token_id] =
  -100`, then with pad_id == eos_id every TERMINAL stop token of every
  assistant turn is also masked; the model never receives gradient toward
  emitting <|im_end|>/<|endoftext|> and cannot learn to stop. Evidence:
  truncated_outputs = 1500/1500 in probe evals; GSM8K retention collapse.

NOTE the generation side is already ruled out in v0.6.1: both
scripts/run_evaluation.py and scripts/run_p1_eval.py pass eos AND <|im_end|>
in eos_token_id. If the model emitted a stop token, decoding would stop.

Three independent checks:

  A) run audit (no GPU): for each eval run dir, read
     artifacts/evaluation_summary.json (decoding.truncated_outputs — the
     authoritative token-id-level stop signal; transcripts are decoded with
     skip_special_tokens=True so stop strings can NEVER appear in text and
     must not be grepped for) and aggregate the evaluation_*.jsonl traces:
     verdict.reason_code distribution, empty predictions, and a runaway
     heuristic (repeated tail n-gram) as a soft non-termination indicator.

  B) gsm8k drift audit (no GPU): run_p1_eval.py persists only a summary, so
     this check consumes an optional per-row predictions jsonl (rows with at
     least {"prediction": str, "gold_answer": str} — see the --help epilog
     for the 6-line dump patch). It compares, using the verifier's OWN
     extract_final_answer (boxed -> markers -> last number), the marker-
     anchored answer vs the naive last-number vs gold, quantifying how much
     of the collapse is answer-extraction drift caused by runaway text
     appended after the true answer (a termination symptom) versus genuine
     wrong answers (a competence symptom).

  C) label audit (CPU, tiny model): builds the REAL trainer via
     axiom_world.training.factory.build_trainer with a tiny random model
     sharing the Qwen3 tokenizer (diag_trainer_labels/e07 pattern), then for
     both the PROCESSED example and one COLLATED batch finds the terminal
     assistant stop token and reports whether its label is -100. Masked in
     the collated batch while pad_id == eos_id  =>  HYPOTHESIS T CONFIRMED.

Usage (Colab, repo root, package installed as `axiom_world`):
  python x09_termination_audit.py \
      --run-dirs runs/<b1-probe-eval-run> runs/<b2-probe-eval-run> \
      --sft-config configs/experiments/b1_general_sft.yaml \
      --sft-jsonl data/p1/p1_general_train.jsonl \
      --p1-predictions runs/p1_eval_b1_predictions.jsonl \
      --n-label-samples 4 \
      --out runs/x09_termination_audit.json

`--p1-predictions` requires the small run_p1_eval.py dump patch (see
DUMP_PATCH below); omit the flag to skip check B.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

DUMP_PATCH = """\
# --- run_p1_eval.py per-row dump patch (check B prerequisite) ---
# add near the other arguments:
#   parser.add_argument("--dump-predictions", default=None)
# inside the scoring loop, after `verdict = verifier.verify(...)`:
#   if args.dump_predictions:
#       with open(args.dump_predictions, "a", encoding="utf-8") as fh:
#           fh.write(json.dumps({
#               "id": record.get("id"),
#               "task_family": record.get("task_family"),
#               "prediction": text,
#               "gold_answer": record["metadata"]["gold_answer"],
#               "passed": value == 1.0,
#           }, ensure_ascii=False) + "\\n")
"""


# ---------------------------------------------------------------------------
# Check A — run audit (artifacts + traces; NO stop-string grepping)
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_trace_rows(run_dir: str):
    pattern = os.path.join(run_dir, "**", "evaluation*.jsonl")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    row = json.loads(clean_line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and "prediction" in row:
                    yield row


def _tail_repetition(text: str, window: int = 32, lookback: int = 512) -> bool:
    """Soft runaway indicator: the final `window` chars recur earlier in the
    tail. Degenerate loops (the classic non-termination signature) trip this;
    normal completions rarely do."""
    tail = text[-window:]
    if len(tail) < window:
        return False
    return tail in text[-(lookback + window) : -window]


def audit_run(run_dir: str) -> dict:
    """NOTE (v0.6.2 fix): fetched run dirs can contain DUPLICATE copies of the
    suite jsonl files (e.g. repo-root artifacts plus a nested runs/ copy — the
    b1-probe fetch materialized 19 files vs 8 and produced 3000 trace rows for
    a 1500-episode eval, halving the apparent truncation rate). Rows are now
    deduplicated by (suite, id) so counts match the summary's episode count."""
    summary = None
    for candidate in sorted(
        glob.glob(os.path.join(run_dir, "**", "evaluation_summary.json"), recursive=True)
    ):
        summary = _read_json(Path(candidate))
        if summary is not None:
            break

    reason_codes: Counter[str] = Counter()
    suites: Counter[str] = Counter()
    n = empty = runaway = duplicates = 0
    seen: set[tuple[str, str]] = set()
    length_chars: list[int] = []
    for row in _iter_trace_rows(run_dir):
        key = (str(row.get("suite")), str(row.get("id")))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        n += 1
        suites[str(row.get("suite"))] += 1
        text = row.get("prediction") or ""
        length_chars.append(len(text))
        if not text.strip():
            empty += 1
        elif _tail_repetition(text):
            runaway += 1
        verdict = row.get("verdict") or {}
        reason_codes[str(verdict.get("reason_code"))] += 1

    decoding = (summary or {}).get("decoding", {})
    truncated = decoding.get("truncated_outputs")
    report = {
        "run_dir": run_dir,
        "trace_rows": n,
        "duplicate_rows_dropped": duplicates,
        "suites": dict(suites),
        "summary_found": summary is not None,
        # authoritative token-level stop signal (counted at generation time
        # in run_evaluation.make_batch_generator, BEFORE skip_special_tokens)
        "truncated_outputs": truncated,
        "truncation_rate": round(truncated / n, 4) if n and isinstance(truncated, int) else None,
        "max_new_tokens": decoding.get("max_new_tokens"),
        "conditioning": (summary or {}).get("conditioning"),
        "verdict_reason_codes": dict(reason_codes.most_common()),
        "empty_predictions": empty,
        "runaway_tail_repetition": runaway,
        "runaway_rate": round(runaway / n, 4) if n else None,
        "prediction_chars": {
            "mean": round(sum(length_chars) / n, 1) if n else None,
            "max": max(length_chars) if length_chars else None,
        },
    }
    if isinstance(truncated, int) and n and truncated >= n:
        report["read_out"] = (
            "ALL outputs hit max_new_tokens although eos AND <|im_end|> are in "
            "the stopping list -> the model is not EMITTING a stop token. "
            "Consistent with hypothesis T; see label_audit verdict."
        )
    return report


# ---------------------------------------------------------------------------
# Check B — GSM8K answer-extraction drift (verifier-identical logic)
# ---------------------------------------------------------------------------

def audit_gsm8k_drift(predictions_jsonl: str) -> dict:
    from axiom_world.core.enums import VerificationStatus
    from axiom_world.verifiers.general import ExactAnswerVerifier, extract_final_answer

    verifier = ExactAnswerVerifier()
    n = 0
    verifier_pass = 0
    marker_present = 0
    drift_marker_vs_lastnum = 0
    drift_caused_failures = 0

    import re

    number_re = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

    def last_number(text: str) -> str | None:
        found = number_re.findall(text or "")
        return found[-1].replace(",", "").rstrip(".") if found else None

    with open(predictions_jsonl, encoding="utf-8") as handle:
        for line in handle:
            clean_line = line.strip()
            if not clean_line:
                continue
            row = json.loads(clean_line)
            text = row.get("prediction") or ""
            gold = row.get("gold_answer") or row.get("metadata", {}).get("gold_answer")
            if not text or gold is None:
                continue
            n += 1

            verdict = verifier.verify(text.strip(), {"answer": gold})
            passed = verdict.status is VerificationStatus.PASSED
            verifier_pass += int(passed)

            anchored = extract_final_answer(text)  # boxed -> markers -> last number
            naive = last_number(text)
            has_marker = any(
                m in text.lower() for m in ("####", "answer:", "final answer:", "정답:")
            )
            marker_present += int(has_marker)
            if (
                has_marker
                and anchored is not None
                and naive is not None
                and anchored.replace(",", "").rstrip(".") != naive
            ):
                drift_marker_vs_lastnum += 1
                # would the marker-anchored answer alone have passed?
                if not passed:
                    anchored_verdict = verifier.verify(
                        f"#### {anchored}", {"answer": gold}
                    )
                    if anchored_verdict.status is VerificationStatus.PASSED:
                        drift_caused_failures += 1

    return {
        "predictions_file": predictions_jsonl,
        "rows": n,
        "verifier_pass_rate": round(verifier_pass / n, 4) if n else None,
        "marker_present_rate": round(marker_present / n, 4) if n else None,
        "marker_vs_lastnum_drift": drift_marker_vs_lastnum,
        # failures where the anchored answer was RIGHT but trailing runaway
        # text changed nothing for the verifier (v0.6.1 extract_final_answer
        # prefers markers, so drift here is informational) — while
        # drift_caused_failures > 0 would indicate extraction, not
        # competence/termination, as a co-factor of the collapse.
        "drift_caused_failures": drift_caused_failures,
        "read_out": (
            "high truncation + LOW drift_caused_failures => collapse is real "
            "non-termination/competence, not answer extraction; "
            "drift_caused_failures > 0 => part of the GSM8K drop is runaway "
            "text confusing extraction (still a termination symptom)."
        ),
    }


# ---------------------------------------------------------------------------
# Check C — label audit through the REAL v0.6.1 trainer factory
# ---------------------------------------------------------------------------

def audit_labels(sft_config: str, sft_jsonl: str, n_samples: int) -> dict:
    import tempfile

    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from axiom_world.core.config_loader import resolve
    from axiom_world.training.factory import build_trainer

    config, _, _ = resolve(sft_config, [])
    repo = config.model.tokenizer_repo_id or config.model.repo_id
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=config.model.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # replicate models/builder.py:32-33

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = {tokenizer.eos_token_id}
    if isinstance(im_end_id, int) and im_end_id >= 0:
        stop_ids.add(im_end_id)

    report: dict = {
        "config": sft_config,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "im_end_token_id": im_end_id,
        "pad_equals_eos": tokenizer.pad_token_id == tokenizer.eos_token_id,
        "assistant_only_loss": bool(config.training.assistant_only_loss),
        "samples": [],
    }

    # Tiny random model with the SAME vocab (e07 pattern): preprocessing and
    # collation depend on the tokenizer, not on weights.
    tiny_config = AutoConfig.from_pretrained(repo, revision=config.model.revision)
    for name, value in {
        "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 32,
    }.items():
        if hasattr(tiny_config, name):
            setattr(tiny_config, name, value)
    tiny_model = AutoModelForCausalLM.from_config(tiny_config)

    rows = []
    with open(sft_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= n_samples:
                break
            clean_line = line.strip()
            if clean_line:
                payload = json.loads(clean_line)
                rows.append({"messages": payload["messages"]})
    dataset = Dataset.from_list(rows)

    def _inspect(input_ids: list[int], labels: list[int] | None, stage: str) -> dict:
        # walk back over padding (pad may EQUAL eos, so require label==-100
        # together with pad id to treat a position as padding — an eos with a
        # live label must NOT be skipped)
        j = len(input_ids) - 1
        while j > 0 and input_ids[j] == tokenizer.pad_token_id and (
            labels is None or labels[j] == -100
        ):
            j -= 1
        window_lo = max(0, j - 5)
        entry = {
            "stage": stage,
            "terminal_pos": j,
            "terminal_token_id": input_ids[j],
            "terminal_token": tokenizer.convert_ids_to_tokens([input_ids[j]])[0],
            "terminal_is_stop_token": input_ids[j] in stop_ids,
            "terminal_label_masked": (labels[j] == -100) if labels is not None else None,
            "tail_tokens": tokenizer.convert_ids_to_tokens(input_ids[window_lo : j + 1]),
            "tail_labels": labels[window_lo : j + 1] if labels is not None else None,
        }
        if not entry["terminal_is_stop_token"]:
            # look for the LAST stop token anywhere and report its label
            stop_positions = [i for i, t in enumerate(input_ids) if t in stop_ids]
            if stop_positions:
                k = stop_positions[-1]
                entry["last_stop_pos"] = k
                entry["last_stop_label_masked"] = (
                    labels[k] == -100 if labels is not None else None
                )
            else:
                entry["last_stop_pos"] = None
        return entry

    masked_collated = 0
    with tempfile.TemporaryDirectory() as tmp:
        trainer = build_trainer(config, tiny_model, tokenizer, dataset, output_dir=tmp)
        processed = trainer.train_dataset
        collator = trainer.data_collator

        for i in range(min(n_samples, len(processed))):
            example = processed[i]
            sample_report: dict = {"idx": i}
            ids = example.get("input_ids")
            if ids is None:
                sample_report["note"] = (
                    "processed example has no input_ids; trainer tokenizes at "
                    f"collation time (keys: {list(example)})"
                )
            else:
                sample_report["processed"] = _inspect(ids, example.get("labels"), "PROCESSED")

            batch = collator([processed[i]])
            batch_ids = batch["input_ids"][0].tolist()
            batch_labels = batch["labels"][0].tolist() if "labels" in batch else None
            collated = _inspect(batch_ids, batch_labels, "COLLATED")
            sample_report["collated"] = collated
            terminal_masked = (
                collated["terminal_is_stop_token"] and collated["terminal_label_masked"]
            )
            fallback_masked = not collated["terminal_is_stop_token"] and bool(
                collated.get("last_stop_label_masked")
            )
            if terminal_masked or fallback_masked:
                masked_collated += 1
            report["samples"].append(sample_report)

    report["collated_terminal_masked_count"] = masked_collated
    if masked_collated > 0 and report["pad_equals_eos"]:
        report["verdict"] = (
            "HYPOTHESIS-T CONFIRMED: the terminal stop token reaching the "
            "forward pass is label-masked (-100) under pad_id == eos_id; the "
            "model receives no gradient toward stopping. Fix: give the "
            "tokenizer a DISTINCT pad token (e.g. reuse an unused special "
            "token or add one + resize) in models/builder.py, or ensure the "
            "collator preserves the label on the FIRST eos of each turn; "
            "then retrain B1/B2."
        )
    elif masked_collated > 0:
        report["verdict"] = (
            "Terminal stop token masked despite pad != eos — inspect the "
            "assistant_only_loss span computation (turn-boundary off-by-one)."
        )
    else:
        report["verdict"] = (
            "Terminal stop labels are LIVE in the collated batch — hypothesis "
            "T rejected at the data pipeline level. Next suspects: (1) "
            "training never converged on the stop token (check "
            "diag_training_dynamics on its logit), (2) eval-time rendering "
            "mismatch between training template and generation prompt."
        )
    return report


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=DUMP_PATCH,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dirs", nargs="*", default=[],
                        help="eval run dirs (check A)")
    parser.add_argument("--p1-predictions", default=None,
                        help="per-row predictions jsonl from patched run_p1_eval (check B)")
    parser.add_argument("--sft-config", default=None,
                        help="SFT recipe YAML, e.g. configs/experiments/b1_general_sft.yaml (check C)")
    parser.add_argument("--sft-jsonl", default=None,
                        help="SFT training jsonl with 'messages' rows (check C)")
    parser.add_argument("--n-label-samples", type=int, default=4)
    parser.add_argument("--out", default="runs/x09_termination_audit.json")
    args = parser.parse_args()

    out: dict = {}
    if args.run_dirs:
        out["run_audit"] = [audit_run(d) for d in args.run_dirs]
    if args.p1_predictions:
        out["gsm8k_drift_audit"] = audit_gsm8k_drift(args.p1_predictions)
    if args.sft_config and args.sft_jsonl:
        out["label_audit"] = audit_labels(args.sft_config, args.sft_jsonl, args.n_label_samples)
    elif args.sft_config or args.sft_jsonl:
        out["label_audit"] = {"skipped": "both --sft-config and --sft-jsonl are required"}

    if not out:
        parser.error("nothing to do: pass --run-dirs and/or --p1-predictions "
                     "and/or (--sft-config + --sft-jsonl)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
