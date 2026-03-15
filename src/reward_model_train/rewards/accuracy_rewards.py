"""Reward functions for evaluating image/video comparison accuracy."""

from __future__ import annotations


def pick_correct_video_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward based on whether the model correctly picks the better video.

    Expects ``kwargs["selections"]`` — a list of ground-truth labels where
    1 means "video 1 is better" and 0 means "video 2 is better".
    """
    selections = kwargs["selections"]
    rewards = []
    for content, selection in zip(completions, selections):
        text = content.lower()
        if "video 1 is better" in text:
            prediction = 1
        elif "video 2 is better" in text:
            prediction = 0
        else:
            prediction = -1
        rewards.append(1.0 if prediction == selection else 0.0)
    return rewards


def pick_correct_image_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    """Reward based on whether the model correctly picks the better image.

    Expects ``kwargs["selections"]`` — a list of ground-truth labels where
    1 means "image 1 is better" and 0 means "image 2 is better".
    """
    selections = kwargs["selections"]
    rewards = []
    for completion, selection in zip(completions, selections):
        text = completion[0]["content"].lower()
        if "image 1 is better" in text:
            prediction = 1
        elif "image 2 is better" in text:
            prediction = 0
        else:
            prediction = -1
        rewards.append(1.0 if prediction == selection else 0.0)
    return rewards
