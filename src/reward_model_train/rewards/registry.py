"""Registry for reward functions — replaces global mutable state."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any, Callable

from reward_model_train.rewards.format_rewards import format_reward
from reward_model_train.rewards.accuracy_rewards import (
    pick_correct_image_reward,
    pick_correct_video_reward,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


class RewardRegistry:
    """Manages available reward functions.

    Usage::

        registry = RewardRegistry(data_source="video")
        funcs = registry.resolve(["think_format_reward", "accuracy_reward"])
    """

    def __init__(self, data_source: str = "image") -> None:
        self._registry: dict[str, Callable] = {
            "think_format_reward": format_reward,
        }
        if data_source == "image":
            self._registry["accuracy_reward"] = pick_correct_image_reward
        else:
            self._registry["accuracy_reward"] = pick_correct_video_reward

    def resolve(self, func_names: list[str]) -> list[Callable]:
        """Resolve a list of reward function names to callables.

        Supports built-in names and dotted import paths (e.g. ``my_lib.rewards.custom``).
        """
        funcs: list[Any] = []
        for name in func_names:
            if name in self._registry:
                funcs.append(self._registry[name])
            elif "." in name:
                module_path, func_name = name.rsplit(".", 1)
                sys.path.insert(0, os.getcwd())
                module = importlib.import_module(module_path)
                funcs.append(getattr(module, func_name))
            else:
                raise ValueError(
                    f"Unknown reward function '{name}'. "
                    f"Available: {list(self._registry.keys())} or use a dotted import path."
                )
        return funcs
