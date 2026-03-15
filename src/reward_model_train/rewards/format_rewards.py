"""Reward functions for validating completion format (think/answer tags)."""

import re


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Check if the completion uses the <think>...</think><answer>...</answer> format.

    Returns 1.0 if the format matches, 0.0 otherwise.
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    matches = [re.match(pattern, c, re.DOTALL) for c in completions]
    return [1.0 if m else 0.0 for m in matches]
