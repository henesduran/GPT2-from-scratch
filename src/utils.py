import os
import random
import numpy as np
import torch

from src.lora import lora_state_dict, inject_lora


def set_seed(seed=1337):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Sets the device available for the system."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_checkpoint(model, optimizer, config, step, val_loss, filepath):
    """Saves the model state and configuration to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # get the raw model if it is wrapped by DDP or torch.compile
    raw_model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model": raw_model.state_dict(),
        "config": config,
        "step": step,
        "val_loss": val_loss,
        "optimizer": optimizer.state_dict() if optimizer else None,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, device="cpu"):
    """
    Loads the checkpoint file from the disk.
    Removes '_orig_mod.' prefixes resulting from torch.compile.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]

    # remove prefix after torch.compile
    prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix) :]] = state_dict.pop(k)
    checkpoint["model"] = state_dict
    return checkpoint

def save_lora_checkpoint(model, r, alpha, target_names, step, val_loss, filepath, optimizer=None):
    """Saves only the LoRA adapter weights plus the hyperparameters needed to re-inject them."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # get the raw model if it is wrapped by DDP or torch.compile
    raw_model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "lora_state_dict": lora_state_dict(raw_model),
        "r": r,
        "alpha": alpha,
        "target_names": target_names,
        "step": step,
        "val_loss": val_loss,
        "optimizer": optimizer.state_dict() if optimizer else None,
    }
    torch.save(checkpoint, filepath)

def load_lora_adapter(model, filepath, device="cpu"):
    """
    Injects LoRA layers into model and loads a saved adapter's weights onto them.
    model must already hold the matching base weights (e.g. via GPT.from_pretrained
    or load_checkpoint); only the LoRA A/B matrices come from this checkpoint.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    inject_lora(model, checkpoint["target_names"], checkpoint["r"], checkpoint["alpha"])
    model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
    return checkpoint