"""Offline tiny Qwen2 fixture exercises real PEFT saved modules and both workers."""
import shutil

import pytest


def test_tiny_bf16_merge_and_fresh_offline_reload(tmp_path):
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
        num_key_value_heads=2, max_position_embeddings=128, eos_token_id=1, pad_token_id=1))
    model.to(torch.bfloat16).save_pretrained(base_dir)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"],
        modules_to_save=["lm_head", "embed_tokens"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.005)
            if "modules_to_save" in name:
                parameter.add_(0.01)
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    stage.mkdir()
    shutil.copytree(adapter, stage / "adapter")
    tok.save_pretrained(stage)
    report = run_workers({"base": str(base_dir), "revision": None, "adapter": str(adapter),
        "stage": str(stage), "scratch": str(tmp_path / "verification"), "device": "cpu",
        "probes": [{"text": "hello world"}, {"messages": [{"role": "user", "content": "hello"}]}]})
    assert report["merge"]["passed"]
    assert report["standalone"]["passed"]
    assert report["standalone"]["peft_imported"] is False
    assert not (stage / "adapter_config.json").exists()
    assert (stage / "model.safetensors").is_file()
