"""Export Hugging Face-compatible tokenizer bundle artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zvezda.tokenizer.special_tokens import flatten_special_tokens


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def build_tokenizer_config(config: dict[str, Any], chat_template: str) -> dict[str, Any]:
    tokenizer_cfg = config["tokenizer"]
    return {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "tokenizer_file": "tokenizer.json",
        "model_max_length": int(tokenizer_cfg.get("model_max_length", 32768)),
        "clean_up_tokenization_spaces": bool(tokenizer_cfg.get("clean_up_tokenization_spaces", False)),
        "add_prefix_space": bool(tokenizer_cfg.get("add_prefix_space", False)),
        "add_bos_token": bool(tokenizer_cfg.get("add_bos_token", False)),
        "eos_token": tokenizer_cfg.get("eos_token"),
        "pad_token": tokenizer_cfg.get("pad_token"),
        "unk_token": tokenizer_cfg.get("unk_token"),
        "chat_template": chat_template,
    }


def build_special_tokens_map(config: dict[str, Any]) -> dict[str, Any]:
    tokenizer_cfg = config["tokenizer"]
    special_tokens = flatten_special_tokens(config)
    control = set(config.get("special_tokens", {}).get("control", []))
    additional = [token for token in special_tokens if token not in control]
    return {
        "eos_token": tokenizer_cfg.get("eos_token"),
        "pad_token": tokenizer_cfg.get("pad_token"),
        "unk_token": tokenizer_cfg.get("unk_token"),
        "additional_special_tokens": additional,
    }


def write_bundle(
    config: dict[str, Any],
    output_dir: Path,
    *,
    repo_root: Path,
    actual_vocab_size: int,
) -> dict[str, Any]:
    tokenizer_cfg = config["tokenizer"]
    template_path = _resolve_repo_path(str(tokenizer_cfg["chat_template_file"]), repo_root)
    if not template_path.exists():
        raise FileNotFoundError(template_path)
    chat_template = template_path.read_text(encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, output_dir / "chat_template.jinja")

    tokenizer_config = build_tokenizer_config(config, chat_template)
    with (output_dir / "tokenizer_config.json").open("w", encoding="utf-8") as handle:
        json.dump(tokenizer_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    special_tokens_map = build_special_tokens_map(config)
    with (output_dir / "special_tokens_map.json").open("w", encoding="utf-8") as handle:
        json.dump(special_tokens_map, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    required_files = list(config.get("bundle_output", {}).get("required_files", []))
    file_hashes: dict[str, str] = {}
    for filename in required_files:
        file_path = output_dir / filename
        if file_path.exists():
            file_hashes[filename] = _sha256(file_path)

    binding = config.get("model_binding", {})
    manifest = {
        "program": config.get("program"),
        "revision": config.get("revision"),
        "tokenizer_name": tokenizer_cfg.get("name"),
        "target_vocab_size": int(tokenizer_cfg["vocab_size"]),
        "actual_vocab_size": actual_vocab_size,
        "compatible_configs": binding.get("compatible_configs", []),
        "special_tokens": flatten_special_tokens(config),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": file_hashes,
    }
    with (output_dir / "bundle_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
