"""Immutable local policy identities for owned, sequential Metal probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_files(directory: Path) -> dict[str, str]:
    paths = sorted(directory.glob("*.safetensors"))
    paths += sorted(directory.glob("*.json"))
    paths += sorted(directory.glob("*.jinja"))
    if (not (directory / "config.json").is_file()
            or not any(path.suffix == ".safetensors" for path in paths)):
        raise ValueError("Checkpoint must contain config and safetensors")
    return {str(path.resolve()): file_digest(path) for path in paths}


class PolicyBinding:
    def __init__(self, manifest: dict):
        self.manifest = manifest
        identity = {key: value for key, value in manifest.items() if key != "model"}
        encoded = json.dumps(identity, sort_keys=True).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        self.model = f"s4-policy-v{manifest['version']}-{digest[:16]}"
        if manifest.get("model", self.model) != self.model:
            raise ValueError("Policy manifest identity mismatch")
        self.version = manifest["version"]

    @classmethod
    def read(cls, path: Path) -> PolicyBinding:
        return cls(json.loads(path.read_text()))

    def verify_files(self):
        for section in ("inference_files", "training_files", "adapter_files"):
            if not self.manifest[section]:
                raise ValueError(f"Empty policy section: {section}")
            for name, expected in self.manifest[section].items():
                if file_digest(Path(name)) != expected:
                    raise ValueError(f"Policy file changed: {name}")

    def verify_catalog(self, catalog: dict):
        bases = [row for row in catalog["data"]
                 if row["id"] == self.manifest["base_model"]]
        roots = {Path(name).parent.resolve()
                 for name in self.manifest["inference_files"]}
        if (len(bases) != 1 or roots != {Path(bases[0]["root"]).resolve()}):
            raise ValueError("Served base path differs from policy manifest")
        entries = [row for row in catalog["data"] if row["id"] == self.model]
        if len(entries) != 1:
            raise ValueError("Bound adapter is missing or ambiguous")
        entry = entries[0]
        if Path(entry["root"]).resolve() != Path(self.manifest["adapter"]).resolve():
            raise ValueError("Served adapter path differs from policy manifest")
        if entry.get("parent") != self.manifest["base_model"]:
            raise ValueError("Served adapter parent differs from policy manifest")

    def verify_response(self, model: str, version: int | None):
        if model != self.model or version != self.version:
            raise ValueError(
                "Backend model or rollout version differs from bound policy"
            )

    def write(self, path: Path):
        payload = {**self.manifest, "model": self.model}
        path.write_text(json.dumps(payload, indent=2) + "\n")
