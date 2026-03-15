# reward-model-train

> Qwen Vision-Language Model GRPO reward model training framework

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![uv](https://img.shields.io/badge/pkg-uv-blueviolet)

## Overview

A training framework for **Group Relative Policy Optimization (GRPO)** on Qwen2.5-VL vision-language models. Supports multi-modal (image + video) reward model training with DeepSpeed distributed training, LoRA fine-tuning, and optional vLLM-accelerated generation.

Key capabilities:
- **GRPO training** with composable reward functions (format validation, accuracy scoring)
- **Multi-modal inputs** — video comparison and image comparison tasks
- **Agent mode** — multi-turn tool-calling (frame selection, image cropping) during generation
- **Distributed training** — DeepSpeed ZeRO-2/3 with CPU offloading
- **LoRA fine-tuning** via PEFT for efficient parameter updates

## Architecture

```
CLI Entry Points          Trainers (vendored TRL fork)        Reward Functions
 rmt-grpo ──────────────> GRPOTrainer_qwen ──────────────> format_reward
 rmt-agent-grpo ────────> GRPOTrainer_agent_qwen ────────> accuracy_reward
 rmt-sft ───────────────> GRPOTrainer_qwen                  (custom via registry)
        |                        |
        |                        v
        v                   DeepSpeed / vLLM
  Data Pipelines            Multi-GPU Training
  (humanbody video)
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.x
- Docker (recommended) or Python 3.10+ with [uv](https://docs.astral.sh/uv/)

### With Docker

```bash
# Build
docker compose build

# Train
docker compose run train

# Interactive development
docker compose run dev
```

### With uv (local install)

```bash
# Install dependencies
uv sync

# Single-GPU debug run
bash scripts/grpo_humanbody_1gpu_debug.sh

# Multi-GPU training (8 GPUs)
bash scripts/grpo_humanbody_8gpu.sh

# Or use the CLI entry point directly
uv run accelerate launch --config_file configs/deepspeed/zero2_1gpu.yaml \
    -m reward_model_train.cli.grpo_train \
    --dataset_name /path/to/dataset \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir output/ \
    --do_train True \
    --report_to wandb
```

## Training Modes

| Mode | CLI | Trainer | Use Case |
|------|-----|---------|----------|
| **GRPO** | `rmt-grpo` | `GRPOTrainer_qwen` | Standard GRPO with think-format + accuracy rewards |
| **Agent GRPO** | `rmt-agent-grpo` | `GRPOTrainer_agent_qwen` | Multi-turn agent with tool use (frame selection, cropping) + vLLM |
| **SFT** | `rmt-sft` | `GRPOTrainer_qwen` | Supervised fine-tuning baseline |

## Configuration

### DeepSpeed

Pre-built configs in `configs/deepspeed/`:

| Config | GPUs | ZeRO Stage | CPU Offload |
|--------|------|------------|-------------|
| `zero2_1gpu.yaml` | 1 | 2 | No |
| `zero2_2gpu.yaml` | 2 | 2 | No |
| `zero2_4gpu.yaml` | 4 | 2 | No |
| `zero2_8gpu.yaml` | 8 | 2 | No |
| `zero3_8gpu_offload.yaml` | 8 | 3 | Optimizer |
| `zero3_2x8gpu_offload.yaml` | 2x8 (16) | 3 | Optimizer |

### wandb Logging

Automatic naming convention:
- **Group:** `{setting}-{MMDD}`
- **Project:** `{setting}-{MMDD}-{HHMM}`

The `setting` is auto-derived from `--data_pipeline` and `--data_source`, or set explicitly via `--wandb_setting`.

### LoRA

Default configuration used across all scripts:
```
--use_peft True
--lora_r 64
--lora_target_modules "q_proj" "v_proj"
--lora_alpha 32
--lora_dropout 0.1
```

## Project Structure

```
reward-model-train/
├── pyproject.toml                          # Dependencies & build config
├── Dockerfile                              # Multi-stage build with uv
├── docker-compose.yml                      # GPU-enabled services
├── configs/deepspeed/                      # DeepSpeed ZeRO configs
├── scripts/                                # Pre-built training shell scripts
│   ├── grpo_humanbody_*gpu*.sh            #   GRPO variants (1/6/7 GPU)
│   ├── agent_grpo_*gpu*.sh               #   Agent GRPO with vLLM
│   └── pipeline_*.sh                      #   Data/training pipeline steps
├── src/reward_model_train/                 # Main package
│   ├── cli/                               #   Training entry points
│   │   ├── grpo_train.py                  #     Standard GRPO
│   │   ├── agent_grpo_train.py            #     Agent GRPO + vLLM
│   │   └── sft_train.py                   #     SFT baseline
│   ├── rewards/                           #   Reward functions
│   │   ├── format_rewards.py              #     <think>/<answer> format check
│   │   ├── accuracy_rewards.py            #     Image/video comparison accuracy
│   │   └── registry.py                    #     Pluggable reward registry
│   ├── data/                              #   Dataset loading & preprocessing
│   │   ├── pipelines.py                   #     Pipeline registry
│   │   └── humanbody.py                   #     Video comparison dataset handler
│   ├── agents/                            #   Multi-turn tool-use generation
│   │   ├── tools.py                       #     SelectFrames, CropImageNormalized
│   │   ├── tool_execution.py              #     Parsing, termination, processing
│   │   └── multi_turn.py                  #     vLLM & Transformer generation loops
│   ├── models/                            #   Model wrappers & OpenRLHF integration
│   ├── vision/                            #   Qwen VL vision processing
│   ├── logging.py                         #   wandb + console logging setup
│   └── utils.py                           #   Shared helpers
└── trl_fork/                                   # Forked TRL with custom Qwen trainers
```

## Custom Reward Functions

Add new reward functions via the registry or dotted import paths:

```bash
# Use a built-in reward
--reward_funcs "think_format_reward" "accuracy_reward"

# Use a custom reward function from your code
--reward_funcs "think_format_reward" "my_package.rewards.custom_reward"
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

## License

Apache 2.0. Vendored TRL is under its own Apache 2.0 license.
