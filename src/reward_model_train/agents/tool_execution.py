"""Tool call parsing, termination checks, and result processing."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Parse the last ``<tool_call>...</tool_call>`` block from the response text."""
    if not text.endswith("</tool_call>"):
        return None
    try:
        matches = re.findall(r"<tool_call>(.*?)</tool_call>", text)
        if matches:
            return json.loads(matches[-1])
    except (json.JSONDecodeError, IndexError):
        return None
    return None


def check_termination_conditions(
    response: str,
    num_tool_calls: int,
    num_images: int,
    total_tokens: int,
    max_tools: int = 3,
    max_images: int = 16,
    max_tokens: int = 12000,
) -> tuple[bool, bool]:
    """Check whether the generation should continue or be terminated.

    Returns:
        (requires_tool_call, force_terminate)
    """
    requires_tool = response.endswith("</tool_call>")
    force_terminate = num_tool_calls > max_tools or num_images > max_images or total_tokens > max_tokens - 200
    return requires_tool, force_terminate


def resize_cropped(image: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    """Resize a cropped image to fit within pixel bounds."""
    w, h = image.size
    total = w * h
    if total < min_pixels:
        scale = np.sqrt(min_pixels / total)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif total > max_pixels:
        scale = np.sqrt(max_pixels / total)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def process_tool_result(
    tool_name: str,
    tool_args: dict[str, Any],
    images: list[Image.Image],
    raw_images: list[Image.Image],
    is_video: bool,
    operations: dict[str, Any],
    image_size_config: dict[str, int],
) -> tuple[list[Image.Image], str, bool]:
    """Execute a tool call and return (added_images, message, had_error)."""
    try:
        if tool_name == "select_frames":
            if not is_video:
                return (
                    [],
                    "Execution error:\nYou attempted to select frames from an **image**, "
                    "but this operation is only designed for analyzing videos. Think again.\n",
                    True,
                )

            selected_frames, info = operations[tool_name].call(images, raw_images, tool_args)

            if isinstance(info, str):
                message = f"\n{info}"
                if not selected_frames:
                    return [], message, False
            else:
                message = ""

            added_images = [
                resize_cropped(
                    frame,
                    min_pixels=image_size_config["select_min_pixels"],
                    max_pixels=image_size_config["select_max_pixels"],
                )
                for frame in selected_frames
            ]

            if added_images:
                size = added_images[0].size
                message = (
                    f"\nHere are the selected frames "
                    f"(Frame Size: {size[0]}x{size[1]}, "
                    f"Numbered {len(images)} to {len(selected_frames) + len(images) - 1}):"
                )
        else:
            cropped = operations[tool_name].call(images, raw_images, tool_args)
            processed = resize_cropped(
                cropped,
                min_pixels=image_size_config["crop_min_pixels"],
                max_pixels=image_size_config["crop_max_pixels"],
            )
            added_images = [processed]
            size = processed.size
            message = f"\nHere is the cropped image (Image Size: {size[0]}x{size[1]}):"

        return added_images, message, False

    except Exception as e:
        logger.warning("Tool execution error: %s", e)
        return [], f"\nExecution error:\n{e}\n", True


def create_tool_response_message(message: str, images: list[Image.Image]) -> dict[str, Any]:
    """Create a user message containing tool output text and images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": message}]
    content.extend({"type": "image", "image": img} for img in images)
    return {"role": "user", "content": content}
