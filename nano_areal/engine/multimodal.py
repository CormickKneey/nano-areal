"""
Multimodal prompt handling shared by the proxy and the local inference server.

* `extract_images(messages)` pulls OpenAI-style `image_url` parts (http(s) URL,
  `data:` URI or local `file://` path) out of a message list and returns
  decoded PIL images plus replay metadata.
* `PromptProcessor` turns a chat message list into model inputs
  (`input_ids`, `pixel_values`, `image_grid_thw`) using the HF processor when
  the checkpoint has one, and falls back to the plain tokenizer for text-only
  models. Both paths use the model's chat template so tokenization is identical
  to what the inference engine sees.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ImageInput:
    image: object            # PIL.Image.Image
    meta: dict               # {"source": ..., "sha1": ..., "width": ..., "height": ...}


def _decode_image_url(url: str):
    from PIL import Image

    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        raw = base64.b64decode(payload)
        src = "data:" + header.split(";")[0][5:]
    elif url.startswith("file://"):
        raw = Path(url[len("file://"):]).read_bytes()
        src = url
    elif url.startswith(("http://", "https://")):
        import httpx
        raw = httpx.get(url, timeout=30.0, follow_redirects=True).content
        src = url
    else:
        raw = Path(url).read_bytes()
        src = f"file://{url}"
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img, {
        "source": src, "sha1": hashlib.sha1(raw).hexdigest(),
        "width": img.width, "height": img.height,
    }


def extract_images(messages: list[dict]) -> list[ImageInput]:
    out: list[ImageInput] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "input_image"):
                url = part.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url", "")
                if not url:
                    continue
                img, meta = _decode_image_url(url)
                out.append(ImageInput(img, meta))
            elif part.get("type") == "image" and isinstance(part.get("image"), str):
                img, meta = _decode_image_url(part["image"])
                out.append(ImageInput(img, meta))
    return out


def text_of(content) -> str:
    """Flatten an OpenAI content field (str or parts list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "".join(parts)


def normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """
    Convert OpenAI `image_url` parts into the `{"type": "image"}` parts HF chat
    templates understand, and drop fields templates cannot serialise.
    """
    norm = []
    for m in messages:
        mm = {"role": m.get("role", "user")}
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") in ("image_url", "input_image", "image"):
                    parts.append({"type": "image"})
                elif p.get("type") == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
            mm["content"] = parts
        else:
            mm["content"] = content if content is not None else ""
        if m.get("tool_calls"):
            mm["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.get("id"),
                    "function": {
                        "name": tc.get("function", {}).get("name", tc.get("name", "")),
                        "arguments": _as_json_obj(tc.get("function", {}).get("arguments", tc.get("arguments", "{}"))),
                    },
                }
                for tc in m["tool_calls"]
            ]
        if m.get("tool_call_id"):
            mm["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            mm["name"] = m["name"]
        if m.get("reasoning_content"):
            mm["reasoning_content"] = m["reasoning_content"]
        norm.append(mm)
    return norm


def _as_json_obj(args):
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"_raw": args}
    return args


@dataclass
class ProcessedPrompt:
    input_ids: list[int]
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    images: list[dict]


class ImageOnlyProcessor:
    """Minimal stand-in for the HF `Qwen3VLProcessor` when the full processor
    cannot be built (e.g. its video processor needs torchvision): tokenizer
    chat template + `AutoImageProcessor`, expanding each `<|image_pad|>` to
    the number of merged patches of the corresponding image."""

    def __init__(self, tokenizer, image_processor, image_token: str = "<|image_pad|>"):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token = image_token
        self.merge_size = int(getattr(image_processor, "merge_size", 2))

    @classmethod
    def from_pretrained(cls, path: str, tokenizer) -> "ImageOnlyProcessor":
        from transformers import AutoImageProcessor
        return cls(tokenizer, AutoImageProcessor.from_pretrained(path))

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, text: list[str], images: list, return_tensors: str = "pt"):
        enc = self.image_processor(images=images, return_tensors=return_tensors)
        grid = enc["image_grid_thw"]
        out = text[0]
        for g in grid.tolist():
            n = int(g[0] * g[1] * g[2]) // (self.merge_size ** 2)
            out = out.replace(self.image_token, "<|placeholder|>" * n, 1)
        out = out.replace("<|placeholder|>", self.image_token)
        ids = self.tokenizer(out, add_special_tokens=False, return_tensors=return_tensors)["input_ids"]
        return {"input_ids": ids, "pixel_values": enc["pixel_values"], "image_grid_thw": grid}


class PromptProcessor:
    """Chat messages (+ images) -> model inputs, via HF processor or tokenizer."""

    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor

    @classmethod
    def from_pretrained(cls, path: str, tokenizer=None) -> "PromptProcessor":
        from transformers import AutoTokenizer
        tok = tokenizer or AutoTokenizer.from_pretrained(path)
        proc = None
        try:
            from transformers import AutoProcessor
            cand = AutoProcessor.from_pretrained(path)
            if hasattr(cand, "image_processor"):
                proc = cand
        except Exception:
            try:
                proc = ImageOnlyProcessor.from_pretrained(path, tok)
            except Exception:
                proc = None
        return cls(tok, proc)

    @property
    def supports_images(self) -> bool:
        return self.processor is not None

    def __call__(self, messages: list[dict], tools: list[dict] | None = None,
                 add_generation_prompt: bool = True) -> ProcessedPrompt:
        images = extract_images(messages)
        norm = normalize_messages_for_template(messages)
        kwargs = {"tools": tools} if tools else {}
        if images and self.processor is not None:
            text = self.processor.apply_chat_template(
                norm, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
            )
            enc = self.processor(text=[text], images=[im.image for im in images], return_tensors="pt")
            return ProcessedPrompt(
                input_ids=enc["input_ids"][0].tolist(),
                pixel_values=enc["pixel_values"],
                image_grid_thw=enc["image_grid_thw"],
                images=[im.meta for im in images],
            )
        # text-only path (also used when images are present but no processor: text parts only)
        if images and self.processor is None:
            norm = [
                {**m, "content": text_of(m["content"]) if isinstance(m.get("content"), list) else m.get("content", "")}
                for m in norm
            ]
        try:
            ids = self.tokenizer.apply_chat_template(
                norm, tokenize=True, add_generation_prompt=add_generation_prompt, **kwargs
            )
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.flatten().tolist()
            elif ids and isinstance(ids[0], list):
                ids = ids[0]
        except Exception:
            text = " ".join(text_of(m.get("content")) for m in norm)
            ids = self.tokenizer(text, add_special_tokens=True).input_ids
        return ProcessedPrompt(list(ids), None, None, [im.meta for im in images])

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def image_to_data_url(image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode("ascii")
