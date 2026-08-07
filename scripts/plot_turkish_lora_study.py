"""
Plots the Turkish LoRA rank-sweep results (MVP vs full-scale runs) produced
by scripts/turkish_lora_study.py: Turkish validation loss vs. rank, and
HellaSwag forgetting (acc_norm drop from baseline) vs. rank, for both the
12.4M-token MVP corpus and the 27.7M-token full-scale corpus.

Usage:
  python scripts/plot_turkish_lora_study.py
"""

import json
import os

import matplotlib.pyplot as plt

RUNS = [
    ("MVP (12.4M tokens, 500 steps)", "results/turkish_lora_study_mvp.json", "tab:blue", "o"),
    ("Full (27.7M tokens, 1000 steps)", "results/turkish_lora_study_full.json", "tab:orange", "s"),
]
OUTPUT = "results/turkish_lora_study_plots.png"


def load_run(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ranks = [rec["r"] for rec in data["ranks"] if "error" not in rec]
    val_loss = [rec["final_val_loss"] for rec in data["ranks"] if "error" not in rec]
    forget = [rec["forgetting_delta_acc_norm"] for rec in data["ranks"] if "error" not in rec]
    return ranks, val_loss, forget


def main():
    fig, (ax_loss, ax_forget) = plt.subplots(1, 2, figsize=(11, 4.5))

    for label, path, color, marker in RUNS:
        ranks, val_loss, forget = load_run(path)
        ax_loss.plot(ranks, val_loss, marker=marker, color=color, label=label)
        ax_forget.plot(ranks, forget, marker=marker, color=color, label=label)

    ax_loss.set_xscale("log", base=2)
    ax_loss.set_xticks([1, 4, 8, 16, 32])
    ax_loss.set_xticklabels([1, 4, 8, 16, 32])
    ax_loss.set_xlabel("LoRA rank (r)")
    ax_loss.set_ylabel("Turkish validation loss")
    ax_loss.set_title("Turkish adaptation quality vs. rank")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_forget.set_xscale("log", base=2)
    ax_forget.set_xticks([1, 4, 8, 16, 32])
    ax_forget.set_xticklabels([1, 4, 8, 16, 32])
    ax_forget.set_xlabel("LoRA rank (r)")
    ax_forget.set_ylabel("HellaSwag acc_norm drop from baseline")
    ax_forget.set_title("Forgetting vs. rank")
    ax_forget.axhline(0, color="gray", linewidth=0.8)
    ax_forget.grid(alpha=0.3)
    ax_forget.legend()

    fig.suptitle("GPT-2 (124M) + LoRA: English->Turkish adaptation vs. HellaSwag forgetting")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
