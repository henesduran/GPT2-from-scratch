import argparse
import torch
import torch.nn.functional as F
import tiktoken

from src.model import GPT
from src.utils import get_device, load_checkpoint, set_seed

def generate_text(
    model,
    enc,
    prompt: str,
    max_length: int = 32,
    num_return_sequences: int = 4,
    device: str = "cpu",
    seed: int = 42,
    ):
    "Generates text using Top-K sampling based on the provided prompt."
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed)

    tokens = enc.encode(prompt) #encode initial prompt
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) #arange shape so it becomes (1,T) since we expect tokens as Batch
    xgen = tokens.repeat(num_return_sequences, 1).to(device)

    while xgen.size(1) < max_length:
        with torch.no_grad(): # will not call .backward()
            with torch.autocast(device_type=device, dtype=torch.bfloat16): # for better performance
                logits, _ = model(xgen) # we are not interested in loss, doing only generation

                logits = logits[:, -1, :]  #only the prediction for last token(what should come next)
                probs = F.softmax(logits, dim=-1)


                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) #getting top50 probs
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix) # indices in topk_indices=vocab id of that token

                xgen = torch.cat((xgen, xcol), dim=1) #concatenate and keep generating with new context until given length

    print("\n" + "=" * 40)
    for i in range(num_return_sequences):
        out_tokens = xgen[i].tolist()
        decoded = enc.decode(out_tokens)
        print(f"Sampling {i+1}:\n{decoded}")
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Inference Script")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to the checkpoint file to be loaded (.pt or .zip)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, I'm a language model,",
        help="Starter text(prompt) for text generation",
    )
    parser.add_argument(
        "--max_length", type=int, default=32, help="Maximum number of tokens to be generated"
    )
    parser.add_argument(
        "--num_sequences", type=int, default=4, help="Number of sequences to be generated"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
        )
    parser.add_argument("--pretrained", type=str, default=None,
                    choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                    help="Load OpenAI pretrained weights instead of a checkpoint"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if args.pretrained is not None:
        model = GPT.from_pretrained(args.pretrained)
    elif args.checkpoint is not None:
        # load checkpoint
        checkpoint = load_checkpoint(args.checkpoint, device=device)
        config = checkpoint["config"]

        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
    else:
        parser.error("Either --pretrained or --checkpoint must be provided")
        
    model.to(device)
    model.eval() # put into evaluation mode,may affect behaviour

    enc = tiktoken.get_encoding("gpt2")

    print("\nGenerating...")
    generate_text(
        model=model,
        enc=enc,
        prompt=args.prompt,
        max_length=args.max_length,
        num_return_sequences=args.num_sequences,
        device=device,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
