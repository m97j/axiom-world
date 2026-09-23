"""Offline tiny Qwen2 fixture exercises real PEFT saved modules and both workers."""
import shutil

import pytest


@pytest.mark.parametrize("tied", [False, True])
def test_tiny_fp32_merge_and_fresh_offline_reload(tmp_path, tied, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from axiom_world.models.champion_release import run_workers

    torch.manual_seed(7)
    base_dir, adapter, stage = [tmp_path / n for n in ("base", "adapter", "model")]
    vocab = {"<unk>": 0, "<|endoftext|>": 1, "<|im_end|>": 2, "hello": 3, "world": 4}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>",
        eos_token="<|endoftext|>", pad_token="<|endoftext|>", additional_special_tokens=["<|im_end|>"])
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=5, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, tie_word_embeddings=tied, max_position_embeddings=128, eos_token_id=1, pad_token_id=1))
    model.to(torch.bfloat16).save_pretrained(base_dir)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"],
        modules_to_save=["lm_head", "embed_tokens"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.005)
            if "modules_to_save" in name:
                parameter.add_(0.01 if "lm_head" in name else 0.02)
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    stage.mkdir()
    shutil.copytree(adapter, stage / "adapter")
    tok.save_pretrained(stage)
    request = {"base": str(base_dir), "revision": None, "adapter": str(adapter),
        "stage": str(stage), "scratch": str(tmp_path / "verification"), "device": "cpu", "dtype": "float32",
        "probes": [{"text": "hello world"}, {"messages": [{"role": "user", "content": "hello"}]}]}
    if tied:
        import subprocess

        import axiom_world.models.champion_release as release
        original_run = subprocess.run
        calls = []
        def interrupted_run(argv, **kwargs):
            calls.append(argv[3])
            if calls == ["merge", "reload"]:
                raise RuntimeError("simulated reload interruption")
            return original_run(argv, **kwargs)
        monkeypatch.setattr(release.subprocess, "run", interrupted_run)
        with pytest.raises(RuntimeError, match="simulated"):
            run_workers(request)
        report = run_workers(request, reload_only=True)
        assert calls == ["merge", "reload", "reload"]
    else:
        report = run_workers(request)
    assert report["merge"]["passed"]
    assert report["standalone"]["passed"]
    assert report["standalone"]["peft_imported"] is False
    assert not (stage / "adapter_config.json").exists()
    assert (stage / "model.safetensors").is_file()

    assert report["dtype"] == "float32"
    assert "historical_vs_fp32_merged" in report["precision"]
    import json
    config = json.loads((stage / "config.json").read_text())
    assert config["tie_word_embeddings"] is False

    # Derive and offline-reload a BF16 candidate without changing the FP32 source.
    if not tied:
        import importlib.util
        from pathlib import Path

        from axiom_world.models.champion_release import inventory, write_json
        spec = importlib.util.spec_from_file_location("derive", Path(__file__).resolve().parents[2]
            / "scripts/publish/v1/derive_bf16_candidate.py")
        derive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(derive)
        (stage / "provenance").mkdir()
        write_json(stage / "provenance/manifest.json", {"verification": report})
        write_json(stage / "provenance/merge_probes.json", request["probes"])
        before = inventory(stage)
        write_json(tmp_path / "verified.json", {"status": "verified", "verification": report, "files": before})
        candidate = tmp_path / "bf16-candidate"
        derive.derive(tmp_path, candidate, "cpu")
        receipt = json.loads((candidate / "verified.json").read_text())
        assert receipt["status"] == "verified_candidate"
        assert receipt["verification"]["dtype"] == "bfloat16"
        assert receipt["verification"]["standalone"]["max_abs"] == 0
        assert receipt["verification"]["standalone"]["peft_imported"] is False
        assert "fp32_vs_bf16" in receipt["verification"]
        assert inventory(stage) == before


def test_qwen3_bf16_cast_preserves_rope_and_reload_logits(tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from axiom_world.models.champion_release import cast_weights_preserving_runtime_buffers

    torch.manual_seed(23)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=256)).eval()
    rotary = model.model.rotary_emb
    original = rotary.inv_freq.clone()
    assert not torch.equal(original, original.bfloat16().float())
    # Reproduce the faulty path: to() rounds a buffer absent from state_dict.
    model.to(torch.bfloat16)
    assert not torch.equal(rotary.inv_freq.float(), original)
    assert not any("inv_freq" in key for key in model.state_dict())
    rotary.inv_freq = original
    cast_weights_preserving_runtime_buffers(model, torch.bfloat16)
    assert rotary.inv_freq.dtype == torch.float32
    assert torch.equal(rotary.inv_freq, original)
    model.save_pretrained(tmp_path)
    reloaded = Qwen3ForCausalLM.from_pretrained(tmp_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).eval()
    assert torch.equal(reloaded.model.rotary_emb.inv_freq, original)
    inputs = torch.arange(128).remainder(32).unsqueeze(0)
    with torch.inference_mode():
        assert torch.equal(model(inputs).logits, reloaded(inputs).logits)
