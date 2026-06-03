# Training configs

DeepSpeed configs for `scripts/train_sft.py --deepspeed <config>`.

## Which config to use

| Config | When | VRAM per GPU | Models |
|---|---|---|---|
| `ds_zero2.json` | 2–8× GPU, model fits per-GPU | ~22 GB for 4B | Qwen3-4B, Qwen3-1.7B |
| `ds_zero3.json` | model does NOT fit per-GPU | ~12 GB for 7B | Qwen3-7B, Qwen3-14B |

**Rule of thumb:** Use ZeRO-2 first. Switch to ZeRO-3 only if you hit OOM with ZeRO-2.

## ds_zero2.json

ZeRO stage 2: shards optimizer states and gradients across GPUs.
Parameters stay on each GPU — forward/backward passes are fast.
Good default for Qwen3-4B on 4×A100 40 GB.

```bash
torchrun --nproc_per_node=4 scripts/train_sft.py \
  --base    Qwen/Qwen3-4B \
  --data    data/traces.jsonl \
  --output  checkpoints/v1 \
  --deepspeed configs/ds_zero2.json
```

## ds_zero3.json

ZeRO stage 3: shards parameters, gradients, AND optimizer states.
Each GPU holds only a slice of every weight — allows training models
that don't fit on a single GPU at all. Slower than ZeRO-2 due to
all-gather calls before every forward pass.

```bash
torchrun --nproc_per_node=4 scripts/train_sft.py \
  --base    Qwen/Qwen3-7B \
  --data    data/traces.jsonl \
  --output  checkpoints/v1 \
  --deepspeed configs/ds_zero3.json
```

## Single GPU

No DeepSpeed needed — just omit `--deepspeed`:

```bash
python scripts/train_sft.py \
  --base   Qwen/Qwen3-4B \
  --data   data/traces.jsonl \
  --output checkpoints/v1
```
