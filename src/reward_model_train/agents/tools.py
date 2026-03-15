"""Vision tool definitions for multi-turn agent interaction."""

from __future__ import annotations

from PIL import Image


class SelectFrames:
    """Select specific frames from a video by index."""

    name = "select_frames"
    parameters = {
        "type": "object",
        "properties": {
            "target_frames": {
                "type": "array",
                "description": "List of frame indices to select from the video (no more than 8 frames in total).",
                "items": {"type": "integer", "description": "Frame index from 1 to 240."},
            }
        },
        "required": ["target_frames"],
    }

    @property
    def description(self) -> str:
        return "Select frames from a video."

    def call(self, images: list[Image.Image], target_frames: list[int]) -> list[Image.Image]:
        return [images[idx] for idx in target_frames]


class CropImageNormalized:
    """Crop and zoom into a region of an image using normalized bounding box coordinates."""

    name = "crop_image_normalized"
    parameters = {
        "type": "object",
        "properties": {
            "bbox_2d": {
                "type": "array",
                "description": "Bounding box coordinates [x1, y1, x2, y2]. Values should be within [0.0, 1.0].",
                "items": {"type": "number"},
            },
            "target_image": {
                "type": "number",
                "description": "Index of the image to crop (1-based).",
            },
        },
        "required": ["bbox_2d", "target_image"],
    }

    @property
    def description(self) -> str:
        return "Zoom in on the image based on the bounding box coordinates."

    def call(self, image: Image.Image, bbox_2d: list[float], padding: float = 0.1) -> Image.Image:
        img_x, img_y = image.size

        if all(v < 1 for v in bbox_2d):
            normalized = (bbox_2d[0] - padding, bbox_2d[1] - padding, bbox_2d[2] + padding, bbox_2d[3] + padding)
        else:
            normalized = (
                bbox_2d[0] / img_x - padding,
                bbox_2d[1] / img_y - padding,
                bbox_2d[2] / img_x + padding,
                bbox_2d[3] / img_y + padding,
            )

        x1 = min(max(0, normalized[0]), 1)
        y1 = min(max(0, normalized[1]), 1)
        x2 = min(max(0, normalized[2]), 1)
        y2 = min(max(0, normalized[3]), 1)

        cropped = image.crop((x1 * img_x, y1 * img_y, x2 * img_x, y2 * img_y))
        w, h = cropped.size
        assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"
        return cropped
