import argparse
import multiprocessing as mp
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# defining tokenizer here globally so that each process in multiproessing pool can access 
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
    """Tokenizes a single document into numpy.uint16."""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (
        0 <= tokens_np
    ).all() and tokens_np.max() < 2**16, "Token IDs exceed the uint16 limit!"
    return tokens_np.astype(np.uint16)


def write_data_file(filename, tokens_np):
    """Saves numpy array to given location."""
    np.save(filename, tokens_np)

def main():
    parser = argparse.ArgumentParser(description="FineWeb/Dataset Preprocessing Script")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="sample-10BT",
        help="Dataset config name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="edu_fineweb10B",
        help="Directory to save token shards",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=int(1e6),
        help="Max token size for each shard",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=10000,
        help="Max number of documents to process",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name} ({args.dataset_config})...")
    fw = load_dataset(
        args.dataset_name, name=args.dataset_config, split="train", streaming=True
    )
    fw = fw.take(args.max_docs)

    n_processes = max(1, os.cpu_count() // 2)
    print(f"Starting tokenization with {n_processes} processing cores..")

    with mp.Pool(n_processes) as pool:
        shard_index = 0
        all_tokens_np = np.empty(dtype=np.uint16, shape=(args.shard_size,))
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < args.shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)

                if progress_bar is None:
                    progress_bar = tqdm(
                        total=args.shard_size,
                        unit="tokens",
                        desc=f"Shard {shard_index}",
                    )
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    args.output_dir, f"edufineweb_{split}_{shard_index:06d}"
                )

                remainder = args.shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]

                write_data_file(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None

                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # Save remaining tokens if there is a remainder
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                args.output_dir, f"edufineweb_{split}_{shard_index:06d}"
            )
            write_data_file(filename, all_tokens_np[:token_count])

    print(
        f"\nDone! All shards are saved to : '{args.output_dir}'."
    )

if __name__ == "__main__":
    main()
