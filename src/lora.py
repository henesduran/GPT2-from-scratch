"""
LoRA (Low-Rank Adaptation) for the GPT-2.

Wraps target nn.Linear layers (e.g. c_attn, c_proj, c_fc) with a frozen
base layer plus a trainable low-rank update (B @ A), so fine-tuning only
touches a small number of parameters instead of the full model.

Usage:
  replaced = inject_lora(model, r=8, alpha=16)
  mark_only_lora_as_trainable(model)
  ... train ...
  merge_lora(model)  # folds LoRA weights back into the base layers
"""

import torch
import torch.nn as nn

DEFAULT_TARGET_NAMES = ["c_attn", "c_proj", "c_fc"]

class LoRALinear(nn.Module):
    """Freezes a base nn.Linear and adds a trainable low-rank (r) side path."""

    def __init__(self, base_linear: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()

        self.base = base_linear
        base_linear.weight.requires_grad_(False)
        if base_linear.bias is not None:
            base_linear.bias.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.empty(r, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r))

        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_x = self.dropout(x)
        lora_out = (lora_x @ self.lora_A.T) @ self.lora_B.T  # (..., in) -> (..., r) -> (..., out)
        return base_out + self.scaling * lora_out


def inject_lora(model: nn.Module, target_names: list = DEFAULT_TARGET_NAMES, r: int = 8, alpha: int = 16, dropout: float = 0.0):
    """Replaces every nn.Linear whose leaf name is in target_names with a LoRALinear, in-place.

    Returns the dotted names of the layers that were replaced.
    """
    replaced = []
    for name, module in list(model.named_modules()):
        leaf = name.split(".")[-1]
        if isinstance(module, nn.Linear) and leaf in target_names:
            parent = model
            *path, leaf = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, leaf, LoRALinear(module, r, alpha, dropout))
            replaced.append(name)
    return replaced

def mark_only_lora_as_trainable(model: nn.Module):
    """Freezes every parameter except the LoRA A/B matrices."""
    for name, parameter in list(model.named_parameters()):
        if "lora_A" in name or "lora_B" in name:
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(False)

def merge_lora(model: nn.Module):
    """Folds each LoRALinear's low-rank update into its base weight and
    swaps the wrapper back out for the plain nn.Linear, in-place."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            W_merged = module.base.weight + module.scaling * (module.lora_B @ module.lora_A)
            module.base.weight.copy_(W_merged)

            parent = model
            *path, leaf = name.split(".")
            for p in path:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            setattr(parent, leaf, module.base)

def lora_state_dict(model: nn.Module):
    """Returns only the LoRA A/B entries of the model's state dict (for saving adapters)."""
    return {key: value for key, value in model.state_dict().items() if "lora_A" in key or "lora_B" in key}
