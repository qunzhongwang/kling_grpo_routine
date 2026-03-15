"""Verbose logging and wandb configuration.

wandb naming convention (user preference):
    Group:   {setting}-{MMDD}
    Project: {setting}-{MMDD}-{HHMM}

Usage::

    from reward_model_train.logging import setup_logging, configure_wandb

    logger = setup_logging()
    configure_wandb(training_args, setting="grpo-humanbody-video")
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


def setup_logging(level: str = "INFO", rank: int = 0) -> logging.Logger:
    """Configure verbose console logging.

    Only rank 0 logs at the requested level; other ranks log WARNING+.
    """
    logger = logging.getLogger("reward_model_train")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    effective_level = level if rank == 0 else "WARNING"
    logger.setLevel(getattr(logging, effective_level))
    return logger


def configure_wandb(
    training_args,
    setting: str,
    entity: str = "KwaiAiTraining",
    now: datetime | None = None,
) -> None:
    """Set wandb group and project following the naming convention.

    Group:   ``{setting}-{MMDD}``
    Project: ``{setting}-{MMDD}-{HHMM}``
    """
    if now is None:
        now = datetime.now()

    mmdd = now.strftime("%m%d")
    hhmm = now.strftime("%H%M")

    group = f"{setting}-{mmdd}"
    project = f"{setting}-{mmdd}-{hhmm}"

    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_RUN_GROUP"] = group
    os.environ["WANDB_ENTITY"] = entity

    if not training_args.run_name:
        training_args.run_name = f"{setting}-{mmdd}-{hhmm}"

    logger = logging.getLogger("reward_model_train")
    logger.info("wandb configured: entity=%s project=%s group=%s", entity, project, group)
