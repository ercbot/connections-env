# Training

Post-training pipeline: SFT warmup followed by RL.

All configs are designed for a machine with **4x A100 80GB VRAM**.

## Steps

### 1. SFT

Fine-tune the base model on gameplay examples to learn the format and basic strategy before RL.

```bash
uv run torchrun --nproc-per-node=4 src/prime_rl/trainer/sft/train.py @ configs/sft/train.toml
```

See [`configs/sft/train.toml`](configs/sft/train.toml).

### 2. RL

Run GRPO on the `train_rl` puzzle split, starting from the SFT checkpoint.

```bash
uv run rl @ configs/rl.toml
```

See [`configs/rl/train.toml`](configs/rl/train.toml). Update `[model].name` to point to your uploaded SFT checkpoint before running.

## Uploading configs to a remote machine

```bash
scp -r training/configs user@your-gpu-instance:/workspace/prime-rl/configs/
```
