"""Download an allowlisted S1 checkpoint from ModelScope with SHA256 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

MODELS = {
    "infer": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "train": "Qwen/Qwen3-VL-4B-Instruct",
}


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="infer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = MODELS[args.model]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "s1-download-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["model"] != model:
            raise ValueError("Manifest model does not match requested checkpoint")
    else:
        url = (
            f"https://modelscope.cn/api/v1/models/{model}/repo/files"
            "?Revision=master&Recursive=true"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            files = json.load(response)["Data"]["Files"]
        manifest = {"model": model, "source": "ModelScope", "files": files}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for entry in manifest["files"]:
        relative = Path(entry["Path"])
        if entry.get("Type") == "tree":
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe checkpoint file path")
        target = args.output / relative
        expected = entry["Sha256"]
        if target.exists() and sha256(target) == expected:
            print(f"Verified {relative}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        url = (
            f"https://modelscope.cn/models/{model}/resolve/master/"
            + urllib.parse.quote(entry["Path"])
        )
        print(f"Downloading {relative}: {entry['Size']} bytes", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            with partial.open("wb") as destination:
                while chunk := response.read(8 * 1024 * 1024):
                    destination.write(chunk)
        if partial.stat().st_size != entry["Size"] or sha256(partial) != expected:
            raise ValueError(f"Checkpoint checksum/size mismatch: {relative}")
        partial.replace(target)
        print(f"Verified {relative}", flush=True)


if __name__ == "__main__":
    main()
