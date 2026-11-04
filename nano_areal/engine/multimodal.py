"""Local-only single-image preparation for exact backend prompt reconciliation."""

from __future__ import annotations

import base64
import io


def has_images(messages):
    return any(
        isinstance(message.get("content"), list)
        and any(
            isinstance(block, dict) and block.get("type") == "image_url"
            for block in message["content"]
        )
        for message in messages
    )


def prepare_single_image(processor, messages, tools=None):
    from PIL import Image

    converted = []
    image_count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message.copy())
            continue
        blocks = []
        for block in content:
            if block.get("type") == "image_url":
                url = block["image_url"]["url"]
                if not isinstance(url, str) or len(url) > 20_000_000:
                    raise ValueError("Image data URI is missing or exceeds 20 MB")
                header, separator, encoded = url.partition(",")
                if not separator or header not in {
                    "data:image/png;base64",
                    "data:image/jpeg;base64",
                    "data:image/webp;base64",
                }:
                    raise ValueError("Only inline PNG/JPEG/WebP images are accepted")
                image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
                if (
                    image.width * image.height > 4_194_304
                    or getattr(image, "n_frames", 1) != 1
                ):
                    raise ValueError("Expected a still image of at most 4 megapixels")
                image.load()
                blocks.append({"type": "image", "image": image.convert("RGB")})
                image_count += 1
            elif block.get("type") == "text":
                blocks.append(block)
            else:
                raise ValueError("Unsupported multimodal content block")
        converted.append({**message, "content": blocks})
    if image_count != 1:
        raise ValueError("Exactly one image is supported per request")
    inputs = processor.apply_chat_template(
        converted,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return (
        inputs["input_ids"][0].tolist(),
        inputs["pixel_values"],
        inputs["image_grid_thw"],
    )
