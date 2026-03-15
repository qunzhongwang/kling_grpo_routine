"""Human body video comparison dataset handler."""

from __future__ import annotations

import logging
import os
import random

import datasets
from datasets import load_dataset

logger = logging.getLogger(__name__)

MAX_PIXELS = 14 * 14 * 80
TOTAL_PIXELS = 1024 * 28 * 28

QUESTION_TEMPLATE = (
    "Given a caption and two videos generated based on this caption, please analyze in detail the two provided videos. "
    "Evaluate them on various dimensions such as semantic consistency (how closely the video content aligns with the caption), "
    "temporal coherence (smoothness and logical flow of motion across frames), authenticity (realism and attention to detail), "
    "coordination of human movement(with emphasis on unrealistic limbs movements and distortions),and any other factors you deem relevant. "
    "For each evaluation dimension, provide a score between 1-10 for both videos (e.g., Video 1: 8/10, Video 2: 6/10) and provide a concise rationale for the score. "
    "Calculate the total score for each video by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. "
    "Then, in the <answer> tag, output exactly one of the following strings: 'Video 1 is better' or 'Video 2 is better' based on the total scores. No additional text is allowed in the <answer> section. "
    "\n\nExample output format:\n<think>1. Semantic consistency: Video 1 (9/10) - ...; Video 2 (7/10) - ...\n2. Temporal coherence: Video 1 (8/10) - ...; Video 2 (6/10) - ...\n3. Authenticity: Video 1 (7/10) - ...; Video 2 (5/10) "
    "- ...\n4. Coordination of human movement: Video 1 (6/10) - ...; Video 2 (8/10) - ... \n[Additional dimensions if any]: Video 1 (6/10) - ...;  Video 2 (8/10) - ...\nTotal score:\nVideo 1: 9+8+7+6+6=36\nVideo 2: 7+6+5+8+8=34</think><answer>Video 1 is better</answer> "
    "Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given videos.* "
    "\n\nYour task is provided as follows:\nText Caption: [{prompt}]\n"
)


def _selection_identify(selection: list, default_selection: list | None = None) -> int:
    if default_selection is None:
        default_selection = ["chosen_video_path", "rejected_video_path"]
    return int(selection == default_selection)


def generate_prompts(
    processor, sample: dict, video_paths: list[str], fps: float, system_prompt: str | None = None
) -> tuple[str, list[dict]]:
    """Build a chat-template prompt comparing two videos."""
    left_video, right_video = video_paths
    message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUESTION_TEMPLATE.format(prompt=sample["caption"])},
                {"type": "text", "text": "This is the start of Video 1:\n"},
                {
                    "type": "video",
                    "video": sample[left_video],
                    "max_pixels": MAX_PIXELS,
                    "total_pixels": TOTAL_PIXELS,
                    "fps": fps,
                },
                {"type": "text", "text": "\nThis is the end of Video 1.\n\nThis is the start of Video 2:\n"},
                {
                    "type": "video",
                    "video": sample[right_video],
                    "max_pixels": MAX_PIXELS,
                    "total_pixels": TOTAL_PIXELS,
                    "fps": fps,
                },
                {"type": "text", "text": "\nThis is the end of Video 2.\n\n"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return prompt, message


def load_human_body(data_path: str, args=None) -> tuple:
    """Load the human body video comparison dataset, split into train/val."""
    fps = getattr(args, "fps", 1.0)
    select_ratio = getattr(args, "data_select_ratio", 0.1)

    logger.info("Loading dataset from %s (fps=%.1f, select_ratio=%.2f)", data_path, fps, select_ratio)

    dataset = load_dataset(path=data_path)["train"]
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"].shuffle(seed=42).select(range(int(len(split["train"]) * select_ratio)))
    val_dataset = split["test"].shuffle(seed=42).select(range(int(len(split["test"]) * select_ratio)))
    train_dataset = train_dataset.select_columns(["chosen_video_path", "rejected_video_path", "caption"])
    val_dataset = val_dataset.select_columns(["chosen_video_path", "rejected_video_path", "caption"])

    logger.info("Dataset loaded: %d train, %d val samples", len(train_dataset), len(val_dataset))
    return train_dataset, val_dataset


def _filter_missing_files(sample: dict) -> dict | None:
    for key in ("chosen_video_path", "rejected_video_path"):
        if not os.path.exists(sample[key]):
            logger.warning("Missing file: %s", sample[key])
            return None
    return sample


def _preprocess_sample(sample: dict, processor=None, system_prompt: str | None = None, fps: float = 1.0) -> dict | None:
    try:
        video_paths = ["chosen_video_path", "rejected_video_path"]
        random.shuffle(video_paths)
        selection = _selection_identify(video_paths)

        prompt, message = generate_prompts(processor, sample, video_paths, fps, system_prompt)
        return {
            "selections": selection,
            "prompts_text": prompt,
            "message": message,
        }
    except Exception as e:
        logger.warning("Preprocessing failed: %s", e)
        return None


def human_body_preprocess_handler(
    dataset: datasets.Dataset, processor, system_prompt: str | None = None, fps: float = 1.0
) -> datasets.Dataset:
    """Filter missing files and preprocess the dataset."""
    dataset = dataset.map(_filter_missing_files, batched=False, load_from_cache_file=False)
    dataset = dataset.map(
        _preprocess_sample,
        fn_kwargs={"processor": processor, "system_prompt": system_prompt, "fps": fps},
        batched=False,
        load_from_cache_file=False,
    )
    return dataset
