"""
Orchestrates the Turkish LoRA adaptation study: sweeps LoRA rank, and for
each rank measures (a) Turkish fine-tuning quality (final train/val loss)
and (b) how much of the base model's English capability is retained
(HellaSwag acc_norm before vs. after), plus qualitative sample generations.

Each rank is trained via train_lora.py in its own subprocess (isolated CUDA
memory, clean per-run logs). Results are flushed to disk after every rank so
an interrupted sweep never loses completed work.

Usage:
  python scripts/turkish_lora_study.py \
    --pretrained gpt2 --data_dir data/turkish_wiki_mvp \
    --ranks 1,4,8,16,32 --alpha_ratio 2 \
    --batch_size 4 --block_size 1024 --total_batch_size 32768 \
    --max_steps 500 --warmup_steps 50 --max_lr 3e-4 --weight_decay 0.1 --seed 1337 \
    --hellaswag_data hellaswag/hellaswag_val.jsonl --hellaswag_limit 2000 --hellaswag_full_baseline \
    --checkpoint_root checkpoint/turkish_lora_study --runs_root runs/turkish_lora_study \
    --results_path results/turkish_lora_study_mvp.json
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import tiktoken

from src.model import GPT
from src.utils import get_device, load_lora_adapter
from src.hellaswag import evaluate as hellaswag_evaluate

PROMPTS = [
    "Türkiye'nin başkenti",
    "Yapay zeka teknolojisi son yıllarda",
    "Bilim insanları yeni bir",
    "Osmanlı İmparatorluğu döneminde",
]

TRAIN_LOSS_RE = re.compile(r"\| loss: ([\d.]+) \|")
TOKSEC_RE = re.compile(r"tok/sec:\s*([\d.]+)")
VALLOSS_RE = re.compile(r"Validation loss \(Step \d+\): ([\d.]+)")


def generate_samples(model, enc, device, max_length=60, seed=42):
    """Top-k=50 sampling, returns {prompt: decoded_text}. Local copy of src/infer.py's
    generate_text() adapted to return strings instead of printing, so infer.py stays untouched."""
    model.eval()
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed)
    samples = {}
    for prompt in PROMPTS:
        xgen = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            while xgen.size(1) < max_length:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _ = model(xgen)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xgen = torch.cat((xgen, torch.gather(topk_indices, -1, ix)), dim=1)
        samples[prompt] = enc.decode(xgen[0].tolist())
    return samples


def free_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def train_one_rank(args, r, alpha, log_path, run_dir):
    """Runs train_lora.py in a subprocess for one (r, alpha) config. Returns (elapsed_seconds, log_text)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [
        sys.executable, "train_lora.py",
        "--pretrained", args.pretrained,
        "--data_dir", args.data_dir,
        "--log_dir", run_dir,
        "--batch_size", str(args.batch_size),
        "--block_size", str(args.block_size),
        "--total_batch_size", str(args.total_batch_size),
        "--max_steps", str(args.max_steps),
        "--warmup_steps", str(args.warmup_steps),
        "--max_lr", str(args.max_lr),
        "--weight_decay", str(args.weight_decay),
        "--lora_r", str(r),
        "--lora_alpha", str(alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--seed", str(args.seed),
    ]
    if args.dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return 0.0, ""

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    elapsed = time.time() - t0

    with open(log_path, "r", encoding="utf-8") as f:
        log_text = f.read()
    return elapsed, log_text


def latest_checkpoint(run_dir):
    ckpts = sorted(glob.glob(os.path.join(run_dir, "lora_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint found in {run_dir}")
    return ckpts[-1]


def main():
    parser = argparse.ArgumentParser(description="Turkish LoRA rank-sweep study")
    parser.add_argument("--pretrained", type=str, default="gpt2")
    parser.add_argument("--data_dir", type=str, default="data/turkish_wiki_mvp")
    parser.add_argument("--ranks", type=str, default="1,4,8,16,32", help="Comma-separated LoRA ranks to sweep")
    parser.add_argument("--alpha_ratio", type=float, default=2.0, help="alpha = alpha_ratio * r for every rank")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--total_batch_size", type=int, default=32768)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--hellaswag_data", type=str, default="hellaswag/hellaswag_val.jsonl")
    parser.add_argument("--hellaswag_limit", type=int, default=2000, help="Examples used for the per-rank before/after comparison")
    parser.add_argument("--hellaswag_full_baseline", action="store_true", help="Also run the full 10042-example HellaSwag once on the base model")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoint/turkish_lora_study")
    parser.add_argument("--runs_root", type=str, default="runs/turkish_lora_study")
    parser.add_argument("--results_path", type=str, default="results/turkish_lora_study_mvp.json")
    parser.add_argument("--gen_max_length", type=int, default=60)
    parser.add_argument("--gen_seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Print planned subprocess commands without running them")
    args = parser.parse_args()

    ranks = [int(r) for r in args.ranks.split(",")]
    device = get_device()
    enc = tiktoken.get_encoding("gpt2")

    results = {
        "meta": {
            "pretrained": args.pretrained,
            "data_dir": args.data_dir,
            "alpha_ratio": args.alpha_ratio,
            "train_args": {
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "total_batch_size": args.total_batch_size,
                "max_steps": args.max_steps,
                "warmup_steps": args.warmup_steps,
                "max_lr": args.max_lr,
                "weight_decay": args.weight_decay,
                "lora_dropout": args.lora_dropout,
                "seed": args.seed,
            },
            "hellaswag_limit": args.hellaswag_limit,
        },
        "baseline": None,
        "ranks": [],
    }

    print("=== Baseline (pretrained, no LoRA) ===")
    if args.dry_run:
        results["baseline"] = {"dry_run": True}
    else:
        model = GPT.from_pretrained(args.pretrained)
        model.to(device)
        model.eval()

        acc, acc_norm = hellaswag_evaluate(model, device, args.hellaswag_data, limit=args.hellaswag_limit)
        baseline = {"hellaswag_limited": {"limit": args.hellaswag_limit, "acc": acc, "acc_norm": acc_norm}}

        if args.hellaswag_full_baseline:
            full_acc, full_acc_norm = hellaswag_evaluate(model, device, args.hellaswag_data, limit=None)
            baseline["hellaswag_full"] = {"acc": full_acc, "acc_norm": full_acc_norm}

        baseline["samples"] = generate_samples(model, enc, device, args.gen_max_length, args.gen_seed)
        results["baseline"] = baseline
        free_model(model)

    save_results(results, args.results_path)
    print(f"Baseline done, saved to {args.results_path}")

    for r in ranks:
        alpha = int(round(args.alpha_ratio * r))
        print(f"\n=== LoRA r={r}, alpha={alpha} ===")
        run_dir = os.path.join(args.checkpoint_root, f"r{r}")
        log_path = os.path.join(args.runs_root, f"r{r}_train.log")

        record = {"r": r, "alpha": alpha}
        try:
            elapsed, log_text = train_one_rank(args, r, alpha, log_path, run_dir)
            record["train_seconds"] = elapsed

            if args.dry_run:
                results["ranks"].append(record)
                save_results(results, args.results_path)
                continue

            toksec_matches = TOKSEC_RE.findall(log_text)
            loss_matches = TRAIN_LOSS_RE.findall(log_text)
            valloss_matches = VALLOSS_RE.findall(log_text)
            record["final_tok_per_sec"] = float(toksec_matches[-1]) if toksec_matches else None
            record["final_train_loss"] = float(loss_matches[-1]) if loss_matches else None
            record["final_val_loss"] = float(valloss_matches[-1]) if valloss_matches else None

            ckpt_path = latest_checkpoint(run_dir)
            record["checkpoint_path"] = ckpt_path

            model = GPT.from_pretrained(args.pretrained)
            load_lora_adapter(model, ckpt_path)  # loads onto CPU first, then .to(device) moves base+adapter together
            model.to(device)
            model.eval()

            acc, acc_norm = hellaswag_evaluate(model, device, args.hellaswag_data, limit=args.hellaswag_limit)
            record["hellaswag"] = {"acc": acc, "acc_norm": acc_norm}
            record["forgetting_delta_acc_norm"] = results["baseline"]["hellaswag_limited"]["acc_norm"] - acc_norm

            record["samples"] = generate_samples(model, enc, device, args.gen_max_length, args.gen_seed)
            free_model(model)

        except Exception as e:
            record["error"] = str(e)
            print(f"Rank r={r} failed: {e}")

        results["ranks"].append(record)
        save_results(results, args.results_path)
        print(f"r={r} done, saved to {args.results_path}")

    print("\n=== Summary ===")
    print(f"{'r':>4s} {'alpha':>6s} {'val_loss':>10s} {'hs_acc_norm':>12s} {'forget_delta':>13s}")
    if results["baseline"] and "hellaswag_limited" in results["baseline"]:
        print(f"{'base':>4s} {'-':>6s} {'-':>10s} {results['baseline']['hellaswag_limited']['acc_norm']:>12.4f} {'-':>13s}")
    for rec in results["ranks"]:
        if "error" in rec:
            print(f"{rec['r']:>4d} {rec['alpha']:>6d} FAILED: {rec['error']}")
            continue
        val_loss = rec.get("final_val_loss")
        hs = rec.get("hellaswag", {}).get("acc_norm")
        delta = rec.get("forgetting_delta_acc_norm")
        print(
            f"{rec['r']:>4d} {rec['alpha']:>6d} "
            f"{val_loss if val_loss is not None else float('nan'):>10.4f} "
            f"{hs if hs is not None else float('nan'):>12.4f} "
            f"{delta if delta is not None else float('nan'):>13.4f}"
        )


if __name__ == "__main__":
    main()
