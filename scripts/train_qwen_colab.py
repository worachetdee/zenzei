"""Train Qwen2.5-7B on Japanese Wikipedia data using LoRA on Colab A100.

This is a continued pretraining script that adapts Qwen2.5-7B for Japanese
using the Swallow methodology with LoRA for memory efficiency.

Usage (in Colab):
    !cd /content/zenzei && python scripts/train_qwen_colab.py
"""
import json
import os
import time
import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

os.chdir("/content/zenzei")

# ============================================================
# Config
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-7B"
DATA_BIN = "data/processed/ja_wiki.bin"
DATA_IDX = "data/processed/ja_wiki.idx"
OUTPUT_DIR = "checkpoints/zensei-7b-ja"
LOG_DIR = "logs"

# Training hyperparameters
MAX_SEQ_LEN = 512       # Short for Colab memory; increase if fits
BATCH_SIZE = 2           # Per-device batch size
GRAD_ACCUM_STEPS = 8     # Effective batch = 2 * 8 = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
MAX_STEPS = 1000         # Stop early for validation; set -1 for full epoch
WARMUP_STEPS = 50
SAVE_STEPS = 200
LOG_STEPS = 10

# LoRA config
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# Dataset
# ============================================================
class MemmapDataset(Dataset):
    """Memory-mapped token dataset for causal LM training."""

    def __init__(self, bin_path, idx_path, seq_len):
        self.seq_len = seq_len

        # Load index
        with open(idx_path, "r") as f:
            index = json.load(f)
        self.total_tokens = index["total_tokens"]

        # Load binary as memmap
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r",
                                shape=(self.total_tokens,))

        # Number of full sequences we can extract
        self.num_samples = self.total_tokens // seq_len
        print(f"  Dataset: {self.total_tokens:,} tokens -> "
              f"{self.num_samples:,} samples of {seq_len} tokens")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = self.tokens[start:end].astype(np.int64)
        x = torch.from_numpy(chunk)
        return x


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Zensei-7B Japanese Continued Pretraining (LoRA)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # --------------------------------------------------------
    # Step 1: Load model
    # --------------------------------------------------------
    print("\n[1/4] Loading Qwen2.5-7B...")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --------------------------------------------------------
    # Step 2: Apply LoRA
    # --------------------------------------------------------
    print("\n[2/4] Applying LoRA...")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    # --------------------------------------------------------
    # Step 3: Load dataset
    # --------------------------------------------------------
    print("\n[3/4] Loading dataset...")
    dataset = MemmapDataset(DATA_BIN, DATA_IDX, MAX_SEQ_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # --------------------------------------------------------
    # Step 4: Train
    # --------------------------------------------------------
    print(f"\n[4/4] Training...")
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} = "
          f"{BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Seq length: {MAX_SEQ_LEN}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # Warmup + cosine schedule
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0
    total_loss = 0.0
    log_losses = []
    best_loss = float("inf")
    t_start = time.time()

    print()
    for epoch in range(NUM_EPOCHS):
        for batch_idx, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device)

            # Forward pass (causal LM: labels = input_ids)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            total_loss += loss.item()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = total_loss
                log_losses.append(step_loss)
                total_loss = 0.0

                if global_step % LOG_STEPS == 0:
                    elapsed = time.time() - t_start
                    avg_recent = sum(log_losses[-LOG_STEPS:]) / min(LOG_STEPS, len(log_losses))
                    lr = scheduler.get_last_lr()[0]
                    tokens_per_sec = (global_step * BATCH_SIZE * GRAD_ACCUM_STEPS * MAX_SEQ_LEN) / elapsed
                    print(f"  Step {global_step:>5d} | "
                          f"loss={avg_recent:.4f} | "
                          f"lr={lr:.2e} | "
                          f"tok/s={tokens_per_sec:,.0f} | "
                          f"elapsed={elapsed:.0f}s")

                if global_step % SAVE_STEPS == 0:
                    ckpt_path = os.path.join(OUTPUT_DIR, f"step_{global_step}")
                    model.save_pretrained(ckpt_path)
                    print(f"  -> Saved checkpoint: {ckpt_path}")

                    if log_losses[-1] < best_loss:
                        best_loss = log_losses[-1]
                        best_path = os.path.join(OUTPUT_DIR, "best")
                        model.save_pretrained(best_path)
                        print(f"  -> New best model (loss={best_loss:.4f})")

                if MAX_STEPS > 0 and global_step >= MAX_STEPS:
                    break

        if MAX_STEPS > 0 and global_step >= MAX_STEPS:
            break

    # Save final
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    total_time = time.time() - t_start
    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"  Total steps:    {global_step}")
    print(f"  Final loss:     {log_losses[-1]:.4f}")
    print(f"  Best loss:      {best_loss:.4f}")
    print(f"  Total time:     {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Model saved to: {final_path}")


if __name__ == "__main__":
    main()
