# CLAUDE.md

## Project: reward-model-train

Qwen VLM GRPO reward model training framework for video/image comparison tasks.

## Build & Run

- Package manager: **uv**
- Install: `uv sync`
- Install with vLLM: `uv sync --extra vllm`
- Train (GRPO): `uv run accelerate launch --config_file configs/deepspeed/zero2_1gpu.yaml -m reward_model_train.cli.grpo_train --dataset_name <path> --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct --do_train True`
- Train (Agent): `uv run accelerate launch --config_file configs/deepspeed/zero2_1gpu.yaml -m reward_model_train.cli.agent_grpo_train --dataset_name <path> --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct --do_train True --use_vllm True`
- Lint: `uv run ruff check src/`
- Format: `uv run ruff format src/`

## Architecture

- `src/reward_model_train/` — Main package
  - `cli/` — Training entry points (grpo_train, agent_grpo_train, sft_train)
  - `rewards/` — Reward functions and pluggable registry
  - `data/` — Dataset loading and preprocessing pipelines
  - `agents/` — Multi-turn tool-use generation (vLLM and Transformer backends)
  - `models/` — Model wrappers (agent wrapper, OpenRLHF integration)
  - `vision/` — Qwen VL vision processing (image/video loading, resizing)
  - `logging.py` — wandb + console logging with auto naming convention
  - `utils.py` — Shared helpers (dist_debug)
- `trl_fork/trl/` — Forked TRL v0.19.0.dev0 with custom Qwen GRPO trainers
- `configs/deepspeed/` — DeepSpeed ZeRO-2/3 accelerate configs
- `scripts/` — Pre-built shell scripts for different GPU configurations

## Key Conventions

- **wandb logging**: Group=`{setting}-{MMDD}`, Project=`{setting}-{MMDD}-{HHMM}`, Entity=`KwaiAiTraining`
- All training uses `accelerate launch` with DeepSpeed configs
- LoRA fine-tuning is the default (r=64, target=q_proj+v_proj)
- The vendored TRL in `trl_fork/trl/` is a fork with custom Qwen VL trainers — do NOT replace with upstream TRL
- Entry points are invoked as modules: `-m reward_model_train.cli.grpo_train`

## Code Style

- Python 3.10+
- Ruff for linting (line-length 120)
- Type hints on public functions
- No global mutable state in library code — use registries/classes
- English comments and docstrings

## Important Files

- `trl_fork/trl/trainer/grpo_trainer_qwen_wo_vllm.py` — Main Qwen GRPO trainer (heavily customized)
- `trl_fork/trl/trainer/grpo_trainer_qwen_agent_vllm.py` — Agent-based trainer with vLLM
- `src/reward_model_train/data/humanbody.py` — Video comparison dataset handler
- `src/reward_model_train/agents/multi_turn.py` — Multi-turn tool-use generation loop
- `src/reward_model_train/rewards/registry.py` — Reward function registry with system prompt
