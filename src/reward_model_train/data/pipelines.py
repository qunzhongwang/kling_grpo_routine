"""Data pipeline registry for loading and preprocessing datasets."""

from __future__ import annotations

from typing import Callable

from reward_model_train.data.humanbody import load_human_body, human_body_preprocess_handler


def select_data_pipeline(name: str) -> tuple[Callable, Callable]:
    """Return (loader, preprocessor) for the given pipeline name."""
    pipelines = {
        "qwen2.5-humanbody-grpo": (load_human_body, human_body_preprocess_handler),
    }
    if name not in pipelines:
        raise NotImplementedError(f"Unknown data pipeline '{name}'. Available: {list(pipelines.keys())}")
    return pipelines[name]
