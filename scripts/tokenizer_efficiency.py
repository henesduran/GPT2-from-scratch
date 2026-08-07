"""
Measures gpt2 BPE tokenizer efficiency (tokens per word) on Turkish vs English
text, as supporting evidence for the Turkish LoRA adaptation study: GPT-2's
tokenizer was built almost entirely from English text, so non-English text
(especially agglutinative languages with non-ASCII characters) fragments into
far more subword tokens per word, shrinking the effective context window.

Usage:
  python scripts/tokenizer_efficiency.py --n_docs 200 --output results/tokenizer_efficiency.json
"""

import argparse
import json
import os

import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")


def measure(texts):
    """Returns tokens/word and bytes/token aggregated over a list of texts."""
    total_words = 0
    total_tokens = 0
    total_bytes = 0
    for text in texts:
        total_words += len(text.split())
        total_tokens += len(enc.encode_ordinary(text))
        total_bytes += len(text.encode("utf-8"))
    return {
        "docs": len(texts),
        "words": total_words,
        "tokens": total_tokens,
        "tokens_per_word": total_tokens / total_words,
        "bytes_per_token": total_bytes / total_tokens,
    }


def load_english_texts(n_docs):
    """Uses HellaSwag's ctx field (already local, no download) as the English reference corpus."""
    texts = []
    with open("hellaswag/hellaswag_val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= n_docs:
                break
            texts.append(json.loads(line)["ctx"])
    return texts


def load_turkish_texts(n_docs):
    """Streams n_docs articles from Turkish Wikipedia (no full download)."""
    ds = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    return [ex["text"] for ex in ds.take(n_docs)]


def main():
    parser = argparse.ArgumentParser(description="Turkish vs English gpt2-tokenizer efficiency")
    parser.add_argument("--n_docs", type=int, default=200, help="Number of documents per language")
    parser.add_argument("--output", type=str, default="results/tokenizer_efficiency.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading {args.n_docs} English texts (HellaSwag ctx)...")
    english = measure(load_english_texts(args.n_docs))

    print(f"Streaming {args.n_docs} Turkish texts (Turkish Wikipedia)...")
    turkish = measure(load_turkish_texts(args.n_docs))

    results = {
        "tokenizer": "tiktoken gpt2",
        "english": english,
        "turkish": turkish,
        "turkish_inefficiency_ratio": turkish["tokens_per_word"] / english["tokens_per_word"],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'language':10s} {'docs':>6s} {'words':>10s} {'tokens':>10s} {'tok/word':>10s} {'bytes/tok':>10s}")
    for name, stats in [("english", english), ("turkish", turkish)]:
        print(
            f"{name:10s} {stats['docs']:6d} {stats['words']:10d} {stats['tokens']:10d} "
            f"{stats['tokens_per_word']:10.3f} {stats['bytes_per_token']:10.3f}"
        )
    print(f"\nTurkish is {results['turkish_inefficiency_ratio']:.2f}x less token-efficient than English.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
