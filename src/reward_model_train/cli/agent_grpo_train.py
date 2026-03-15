"""Agent-based GRPO training entry point with vLLM and tool-use support."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForSequenceClassification, AutoProcessor

from trl import GRPOConfig, GRPOTrainer_agent_qwen, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from reward_model_train.data.pipelines import select_data_pipeline
from reward_model_train.logging import configure_wandb, setup_logging
from reward_model_train.rewards.registry import SYSTEM_PROMPT, RewardRegistry
from reward_model_train.utils import dist_debug

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the agent GRPO training script."""

    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Reward model id or local path."},
    )
    reward_funcs: Optional[list[str]] = field(
        default_factory=lambda: ["think_format_reward", "accuracy_reward"],
        metadata={"help": "Reward functions to use. Supports built-in names or dotted import paths."},
    )
    save_last_checkpoint: bool = False
    debug_entry_point: bool = False
    data_source: str = "image"
    data_pipeline: str = "qwen2.5-humanbody-grpo"
    data_select_ratio: float = 0.1
    cache_dir: Optional[str] = None
    fps: float = 1.0
    prompt_data: str = "test"
    eval_data: str = "test"
    wandb_setting: Optional[str] = None


def main(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig) -> None:
    rank = 0
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            rank = dist.get_rank()
    except Exception:
        pass

    setup_logging(rank=rank)

    # Configure wandb with user's preferred naming convention
    if training_args.report_to and "wandb" in training_args.report_to:
        setting = script_args.wandb_setting or f"{script_args.data_pipeline}-{script_args.data_source}"
        configure_wandb(training_args, setting=setting)

    # Build reward functions
    registry = RewardRegistry(data_source=script_args.data_source)
    reward_funcs = []

    if script_args.reward_model_name_or_path:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            num_labels=1,
        )
        reward_funcs.append(reward_model)

    if script_args.reward_funcs:
        reward_funcs.extend(registry.resolve(script_args.reward_funcs))

    logger.info("Using %d reward functions", len(reward_funcs))

    # Load and preprocess dataset
    loader, preprocessor = select_data_pipeline(script_args.data_pipeline)
    train_dataset, test_dataset = loader(script_args.dataset_name, script_args)

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    train_dataset = preprocessor(train_dataset, processor, SYSTEM_PROMPT, fps=script_args.fps)
    test_dataset = preprocessor(test_dataset, processor, SYSTEM_PROMPT, fps=script_args.fps)
    del processor

    # Initialize agent trainer
    trainer = GRPOTrainer_agent_qwen(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    if script_args.debug_entry_point:
        dist_debug()

    if training_args.do_train:
        logger.info("Starting Agent GRPO training")
        trainer.train()

    if training_args.do_eval:
        logger.info("Running evaluation")
        trainer.evaluate()

    if script_args.save_last_checkpoint:
        trainer.save_model(training_args.output_dir)
        logger.info("Model saved to %s", training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction | None = None) -> TrlParser:
    dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig)
    if subparsers is not None:
        return subparsers.add_parser("agent-grpo", help="Run Agent GRPO training", dataclass_types=dataclass_types)
    return TrlParser(dataclass_types)


def main_cli() -> None:
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_init_kwargs = {}
    for name in ("torch_dtype", "cache_dir"):
        if hasattr(script_args, name):
            model_init_kwargs[name] = getattr(script_args, name)
        elif hasattr(training_args, name):
            model_init_kwargs[name] = getattr(training_args, name)
    training_args.model_init_kwargs = model_init_kwargs
    training_args.prompt_data = script_args.prompt_data
    training_args.eval_data = script_args.eval_data
    training_args.zero_stage = 2
    training_args.advantage_estimator = "group"
    main(script_args, training_args, model_args)


if __name__ == "__main__":
    main_cli()
