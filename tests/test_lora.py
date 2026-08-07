"""
Tests for LoRA injection, freezing, merging, and adapter checkpointing
(src/lora.py and the LoRA helpers in src/utils.py).
"""

import torch

from src.model import GPT, GPTConfig
from src.lora import inject_lora, mark_only_lora_as_trainable, merge_lora, LoRALinear, DEFAULT_TARGET_NAMES
from src.utils import save_lora_checkpoint, load_lora_adapter


def test_equivalence_at_init():
    """lora_B starts at zero, so injecting LoRA must not change the model's output."""
    torch.manual_seed(0)
    config = GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    model = GPT(config=config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits_before, _ = model(idx)

    replaced = inject_lora(model, target_names=DEFAULT_TARGET_NAMES, r=2, alpha=4)
    assert len(replaced) > 0, "no layers were replaced with LoRA"
    model.eval()

    with torch.no_grad():
        logits_after, _ = model(idx)

    diff = (logits_before - logits_after).abs().max().item()
    print(f"max abs diff: {diff}")

    assert torch.allclose(logits_before, logits_after, atol=1e-5), "diff is out of tolerance"


def test_only_lora_trainable():
    """mark_only_lora_as_trainable must leave exactly the LoRA A/B params trainable, everything else frozen."""
    torch.manual_seed(0)
    config = GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    model = GPT(config=config)

    total_before = sum(p.numel() for p in model.parameters())

    inject_lora(model, target_names=DEFAULT_TARGET_NAMES, r=2, alpha=4)
    mark_only_lora_as_trainable(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"total parameters: {total_before}")
    print(f"trainable parameters(LoRA): {trainable}")
    print(f"frozen parameters: {frozen}")

    assert trainable > 0, "no trainable parameters found"
    assert frozen > 0, "no frozen parameters found,freeze might be failed"
    assert trainable < total_before * 0.5, (
        "number of trainable parameters is more than total parameters,freeze might be failed "
    )

    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.requires_grad, f"expected parameter {name} to be trainable but it is not"
        else:
            assert not p.requires_grad, f"expected parameter {name} to be frozen but it is not"


def test_merge_preserves_output():
    """merge_lora must fold the LoRA update into the base weights without changing the model's output."""
    torch.manual_seed(0)
    config = GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    model = GPT(config=config)

    inject_lora(model, target_names=DEFAULT_TARGET_NAMES, r=2, alpha=4)

    for m in model.modules():
        if isinstance(m, LoRALinear):
            with torch.no_grad():
                m.lora_B.add_(torch.rand_like(m.lora_B) * 0.01)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        logits_before, _ = model(idx)

    merge_lora(model)
    model.eval()

    remaining = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    assert remaining == 0, f"found {remaining} LoRALinear left after merge"

    with torch.no_grad():
        logits_after, _ = model(idx)

    diff = (logits_before - logits_after).abs().max().item()
    print(f"max abs logit diff: {diff:.2e}")

    assert torch.allclose(logits_before, logits_after, atol=1e-4), (
        "diff is out of tolerance"
    )


def test_adapter_checkpoint_roundtrip(tmp_path):
    """save_lora_checkpoint + load_lora_adapter must reproduce the trained model's output on a
    fresh instance holding the same base weights."""
    torch.manual_seed(0)
    config = GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    model_a = GPT(config=config)
    base_state = model_a.state_dict()  # pre-injection base weights, shared with model_b below

    inject_lora(model_a, target_names=DEFAULT_TARGET_NAMES, r=2, alpha=4)
    mark_only_lora_as_trainable(model_a)

    for m in model_a.modules():
        if isinstance(m, LoRALinear):
            with torch.no_grad():
                m.lora_B.add_(torch.rand_like(m.lora_B) * 0.01)

    model_a.eval()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        logits_before, _ = model_a(idx)

    filepath = tmp_path / "adapter.pt"
    save_lora_checkpoint(
        model_a, r=2, alpha=4, target_names=DEFAULT_TARGET_NAMES,
        step=0, val_loss=0.0, filepath=str(filepath),
    )
    assert filepath.exists(), "checkpoint file was not written to disk"

    adapter = torch.load(filepath, weights_only=False)["lora_state_dict"]
    assert len(adapter) > 0, "lora_state_dict is empty"
    assert all(("lora_A" in k or "lora_B" in k) for k in adapter.keys()), (
        "lora_state_dict should include only lora_A/lora_B keys"
    )

    model_b = GPT(config=config)
    model_b.load_state_dict(base_state)  # same base weights as model_a, pre-injection
    checkpoint = load_lora_adapter(model_b, str(filepath))

    assert checkpoint["r"] == 2
    assert checkpoint["alpha"] == 4
    assert checkpoint["target_names"] == DEFAULT_TARGET_NAMES

    model_b.eval()
    with torch.no_grad():
        logits_after, _ = model_b(idx)

    diff = (logits_before - logits_after).abs().max().item()
    print(f"max abs logit diff: {diff:.2e}")

    assert torch.allclose(logits_before, logits_after, atol=1e-5), (
        "outputs should match after saving and reloading the LoRA adapter from disk"
    )
