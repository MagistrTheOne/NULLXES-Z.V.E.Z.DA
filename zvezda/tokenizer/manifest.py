"""Corpus manifest loading and validation for tokenizer training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.tokenizer.trainer import CorpusFile


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def load_schema(path: Path) -> dict[str, Any]:
    return _load_mapping(path)


def validate_corpus_manifest(manifest: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for field in schema.get("required_fields", []):
        if field not in manifest:
            errors.append(f"manifest missing required field: {field}")

    allowed_segments = set(schema.get("allowed_segments", []))
    report_segments = set(schema.get("report_segments", []))
    all_segments = allowed_segments | report_segments

    files = manifest.get("files")
    if not isinstance(files, list):
        errors.append("manifest.files must be a list")
        return errors

    rules = schema.get("validation_rules", {})
    if rules.get("files_must_be_non_empty") and not files:
        errors.append("manifest.files must be non-empty (no mock data)")

    seen_paths: set[str] = set()
    for index, entry in enumerate(files):
        prefix = f"files[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue

        for field in schema.get("file_entry_required_fields", []):
            if field not in entry:
                errors.append(f"{prefix} missing required field: {field}")

        segment = entry.get("segment")
        if isinstance(segment, str) and all_segments and segment not in all_segments:
            errors.append(f"{prefix}.segment '{segment}' is not allowed")

        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        if raw_path in seen_paths:
            errors.append(f"{prefix}.path duplicate: {raw_path}")
        seen_paths.add(raw_path)

        if rules.get("license_must_not_be_unknown") and entry.get("license") in {None, "", "unknown"}:
            errors.append(f"{prefix}.license must be set to a known license")

        content_hash = entry.get("content_hash")
        if isinstance(content_hash, dict):
            algorithm = content_hash.get("algorithm")
            allowed = set(rules.get("content_hash_algorithm", []))
            if allowed and algorithm not in allowed:
                errors.append(f"{prefix}.content_hash.algorithm must be one of {sorted(allowed)}")

        file_path = Path(raw_path)
        if rules.get("paths_must_exist") and not file_path.exists():
            errors.append(f"{prefix}.path does not exist: {raw_path}")
        elif rules.get("paths_must_be_files") and file_path.exists() and not file_path.is_file():
            errors.append(f"{prefix}.path is not a file: {raw_path}")

    return errors


def load_corpus_manifest(path: Path, schema_path: Path | None = None) -> list[CorpusFile]:
    manifest = _load_mapping(path)
    if schema_path is not None:
        schema = load_schema(schema_path)
        errors = validate_corpus_manifest(manifest, schema)
        if errors:
            raise ValueError("invalid corpus manifest:\n- " + "\n- ".join(errors))

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("corpus manifest must contain non-empty 'files' list")

    corpus_files: list[CorpusFile] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("each corpus manifest entry must be an object")
        segment = entry.get("segment")
        raw_path = entry.get("path")
        if not isinstance(segment, str) or not segment:
            raise ValueError("each corpus file must define non-empty segment")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("each corpus file must define non-empty path")
        file_path = Path(raw_path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        if not file_path.is_file():
            raise ValueError(f"corpus path is not a file: {file_path}")
        corpus_files.append(CorpusFile(segment=segment, path=file_path))
    return corpus_files
