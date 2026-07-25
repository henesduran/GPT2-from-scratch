import argparse
import math
import os
import time
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data import DataLoaderLite
from src.model import GPT, GPTConfig
from src.utils import get_device, save_checkpoint, set_seed

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate scheduler"""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Training Script")
    parser.add_argument("--data_dir", type=str, default="edu_fineweb10B", help="Folder with data shards")
    parser.add_argument("--log_dir", type=str, default="checkpoint", help="Folder for checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size")
    parser.add_argument("--block_size", type=int, default=1024, help="Sequence length")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="Total tokens for gradient accumulation")
    parser.add_argument("--max_steps", type=int, default=50, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--max_lr", type=float, default=6e-4, help="Maximum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "Cuda is required for DDP!"
        init_process_group(backend="nccl") #communication between gpus
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = get_device()

    set_seed(args.seed + ddp_rank)

    assert args.total_batch_size % (args.batch_size * args.block_size * ddp_world_size) == 0, "total_batch_size must divide evenly by batch_size * block_size * ddp_world_size"
    grad_accum_steps = args.total_batch_size // (args.batch_size * args.block_size * ddp_world_size)

    if master_process:
        print(f"Total batch size: {args.total_batch_size}")
        print(f"=> Number of gradient accumulation steps: {grad_accum_steps}")

    #data loaders

    train_loader = DataLoaderLite(
        B=args.batch_size,
        T=args.block_size,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train",
        data_root=args.data_dir,
    )
    val_loader = DataLoaderLite(
        B=args.batch_size,
        T=args.block_size,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        data_root=args.data_dir,
    )

    torch.set_float32_matmul_precision("high") # for better performance (very-high -> high)

    config = GPTConfig(vocab_size=50304) #choosing 'good numbers' for better performance
    model = GPT(config)
    model.to(device)

    if master_process:
        print("Compiling model (torch.compile)...")
    model = torch.compile(model)


    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.max_lr,
        device=device,
    )

    min_lr = args.max_lr * 0.1 # %10 of max lr

    for step in range(args.max_steps):
        t0 = time.time()

        # Validation
        if step % 100 == 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation loss (Step {step}): {val_loss_accum.item():.4f}")

        # Checkpointing
        if master_process and step > 0 and (step % 25 == 0 or step == args.max_steps - 1):
            checkpoint_path = os.path.join(args.log_dir, f"model_{step:05d}.pt")
            print(f"Saving Checkpoint: {checkpoint_path}...")
            save_checkpoint(
                model=raw_model,
                optimizer=optimizer,
                config=config,
                step=step,
                val_loss=val_loss_accum.item() if 'val_loss_accum' in locals() else 0.0,
                filepath=checkpoint_path,
            )

        # Gradient accumulation & forward-backward pass
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) #sync only if we are averaging the loss,otherwise do not sync its costly 
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        dt = time.time() - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            print(
                f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
                f"norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()


