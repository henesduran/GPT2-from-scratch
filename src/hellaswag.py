"""
HellaSwag evaluation for the from-scratch GPT-2.

Scores each of the 4 candidate endings by the (masked) cross-entropy of its
tokens given the context, and picks the lowest-loss ending.
Reports acc (sum loss) and acc_norm (length-normalized loss).

Usage:
  python -m src.hellaswag --pretrained gpt2
  python -m src.hellaswag --checkpoint checkpoint/model_00049.pt
"""

import argparse
import json
import os
import urllib.request

import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model import GPT
from src.utils import get_device, load_checkpoint

DATA_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"

def download_data(data_dir: str = "hellaswag") -> str:
    """Downloads the validation split (10,042 examples) if not present."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "hellaswag_val.jsonl")
    if not os.path.exists(path):
        print(f"Downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return path

def iterate_example(path : str):
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
def render_example(example : dict,enc):
    """
    Builds a (4, T) token tensor and a (4, T) mask tensor for one example.
    mask = 1 on ending tokens only (context and padding are 0).
    """

    ctx_tokens = enc.encode(example["ctx"])
    label = int(example["label"])

    tok_rows , mask_rows = [] , []
    for ending in example["endings"]:
        end_tokens = enc.encode( " " + ending) # leading space changes BPE behaviour
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens)) #only interested in ending loss,since context is already given to model

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4,max_len),dtype=torch.long)
    masks = torch.zeros((4,max_len),dtype=torch.long)

    for i , (token,mask) in enumerate(zip(tok_rows,mask_rows)):
        tokens[i,:len(token)] = torch.tensor(token,dtype=torch.long)
        masks[i,:len(mask)] = torch.tensor(mask,dtype=torch.long)
    return tokens , masks ,label


@torch.no_grad()
def evaluate(model, device: str, data_path: str, limit: int | None = None):
    enc = tiktoken.get_encoding("gpt2")
    model.eval()


    num_correct, num_correct_norm, num_total = 0, 0, 0
    examples = iterate_example(data_path)

    for i, example in enumerate(tqdm(examples, total=limit or 10042, desc="HellaSwag")):
        if limit is not None and i >= limit:
            break

        tokens, mask, label = render_example(example, enc)
        tokens, mask = tokens.to(device), mask.to(device)

        logits, _ = model(tokens) #(4,T, vocab_size)

        # shifting: logits at position t predict token t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_tokens = tokens[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()  # align mask with TARGET tokens

        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_tokens.view(-1),
            reduction="none",
        ).view(tokens.size(0), -1)  # (4, T-1) per-position loss

        masked_losses = losses * shift_mask
        sum_loss = masked_losses.sum(dim=1)               # sum along rows -> (4,)
        avg_loss = sum_loss / shift_mask.sum(dim=1)       # (4,) normalized by length(number of tokens)

        #get the option with minimum loss
        pred = sum_loss.argmin().item() 
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

    acc = num_correct / num_total
    acc_norm = num_correct_norm / num_total
    print(f"\nHellaSwag ({num_total} examples)")
    print(f"acc:      {num_correct}/{num_total} = {acc:.4f}")
    print(f"acc_norm: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
    return acc, acc_norm 


def main():
    parser = argparse.ArgumentParser(description="HellaSwag evaluation")
    parser.add_argument("--pretrained", type=str, default=None,
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="Evaluate OpenAI pretrained weights")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Evaluate a saved training checkpoint")
    parser.add_argument("--data_dir", type=str, default="hellaswag",
                        help="Directory to store the dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples (smoke test)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if args.pretrained is not None:
        model = GPT.from_pretrained(args.pretrained)
    elif args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint, device=device)
        model = GPT(checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
    else:
        parser.error("Either --pretrained or --checkpoint must be provided")

    model.to(device)
    data_path = download_data(args.data_dir)
    evaluate(model, device, data_path, limit=args.limit)

if __name__ == "__main__":
    main()


