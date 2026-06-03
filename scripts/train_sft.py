#!/usr/bin/env python3
"""
train_sft.py — Full SFT fine-tune of Qwen3-4B on Albedo duel traces.

Single A100 80 GB:
  python scripts/train_sft.py --data data/traces.jsonl --output checkpoints/v1

Multi-GPU 4×A100:
  torchrun --nproc_per_node=4 scripts/train_sft.py \
    --data data/traces.jsonl \
    --output checkpoints/v1 \
    --deepspeed configs/ds_zero2.json
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_data(path, tokenizer, max_length):
    records = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    kept, dropped = [], 0
    for r in records:
        ids = tokenizer.encode(r["text"], add_special_tokens=False)
        if len(ids) > max_length:
            dropped += 1
        else:
            kept.append({"text": r["text"]})
    print(f"Dataset: {len(kept)} examples kept, {dropped} dropped (too long for max_length={max_length})")
    if not kept:
        raise SystemExit("No examples after filtering — lower --max-length or re-collect traces")
    return Dataset.from_list(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",       required=True,  help="Path to traces.jsonl")
    ap.add_argument("--output",     required=True,  help="Checkpoint output directory")
    ap.add_argument("--base",       default="Qwen/Qwen3-4B",
                    help="Base model to fine-tune (default: Qwen/Qwen3-4B)")
    ap.add_argument("--epochs",     type=int,   default=3)
    ap.add_argument("--lr",         type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int,   default=4,
                    help="Per-device batch size")
    ap.add_argument("--grad-accum", type=int,   default=4,
                    help="Gradient accumulation steps. "
                         "Effective batch = batch_size × grad_accum × n_gpus")
    ap.add_argument("--max-length", type=int,   default=8192,
                    help="Max sequence length in tokens. Qwen3 supports up to 32768.")
    ap.add_argument("--deepspeed",  default=None,
                    help="Path to DeepSpeed config JSON (for multi-GPU)")
    ap.add_argument("--no-flash-attn", action="store_true",
                    help="Disable flash-attention (use if flash-attn is not installed)")
    args = ap.parse_args()

    print(f"Loading tokenizer from {args.base} …")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for SFT loss masking

    print(f"Loading model {args.base} …")
    attn = "eager" if args.no_flash_attn else "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
        trust_remote_code=False,
    )
    model.config.use_cache = False  # required with gradient_checkpointing=True

    dataset = load_data(args.data, tokenizer, args.max_length)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output,

            # Schedule
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,

            # Precision and memory
            bf16=True,
            gradient_checkpointing=True,
            max_seq_length=args.max_length,

            # Data
            dataset_text_field="text",
            packing=False,  # keep examples separate — cleaner SFT signal

            # Logging and saving
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",

            # Multi-GPU
            deepspeed=args.deepspeed,
        ),
    )

    print("Training …")
    trainer.train()

    final = Path(args.output) / "final"
    print(f"Saving to {final} …")
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print("Done.")


if __name__ == "__main__":
    main()
