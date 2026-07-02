"""Cloud-only corpus manifest loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.data.cloud import is_cloud_uri, parse_cloud_uri, probe_cloud_uri, reject_local_path
from zvezda.data.corpus import CloudCorpusFile


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


def validate_corpus_manifest(
    manifest: dict[str, Any],
    schema: dict[str, Any],
    *,
    storage_config: dict[str, Any] | None = None,
    probe_cloud: bool = False,
) -> list[str]:
    errors: list[str] = []

    for field in schema.get("required_fields", []):
        if field not in manifest:
            errors.append(f"manifest missing required field: {field}")

    storage_policy = schema.get("storage_policy", {})
    if storage_policy.get("runpod_only") and manifest.get("storage_policy") not in (None, "runpod_only", "cloud_only"):
        errors.append("manifest must declare storage_policy: runpod_only")

    allowed_segments = set(schema.get("allowed_segments", []))
    report_segments = set(schema.get("report_segments", []))
    all_segments = allowed_segments | report_segments

    files = manifest.get("files")
    if not isinstance(files, list):
        errors.append("manifest.files must be a list")
        return errors

    rules = schema.get("validation_rules", {})
    if rules.get("files_must_be_non_empty") and not files:
        errors.append("manifest.files must be non-empty (populate runpod:// URIs before pod launch)")

    seen_uris: set[str] = set()
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

        uri = entry.get("uri")
        if not isinstance(uri, str) or not uri:
            errors.append(f"{prefix}.uri must be a non-empty cloud URI")
            continue

        if rules.get("local_paths_allowed") is False:
            try:
                reject_local_path(uri)
            except ValueError as exc:
                errors.append(f"{prefix}.uri: {exc}")

        if rules.get("cloud_uri_required") and not is_cloud_uri(uri):
            errors.append(f"{prefix}.uri must use runpod:// volume paths on RunPod launch")

        if uri in seen_uris:
            errors.append(f"{prefix}.uri duplicate: {uri}")
        seen_uris.add(uri)

        if rules.get("license_must_not_be_unknown") and entry.get("license") in {None, "", "unknown"}:
            errors.append(f"{prefix}.license must be set to a known license")

        content_hash = entry.get("content_hash")
        if isinstance(content_hash, dict):
            algorithm = content_hash.get("algorithm")
            allowed = set(rules.get("content_hash_algorithm", []))
            if allowed and algorithm not in allowed:
                errors.append(f"{prefix}.content_hash.algorithm must be one of {sorted(allowed)}")
            if not content_hash.get("value"):
                errors.append(f"{prefix}.content_hash.value is required")
            else:
                value = str(content_hash.get("value"))
                if value.startswith("<") or value.endswith(">"):
                    errors.append(f"{prefix}.content_hash.value is a placeholder stub")
                if "your-bucket" in uri.lower() or "example" in uri.lower():
                    errors.append(f"{prefix}.uri looks like a template stub")

        if probe_cloud:
            try:
                parse_cloud_uri(uri)
                probe_cloud_uri(uri, storage_config)
            except Exception as exc:  # noqa: BLE001 — surface cloud access failures in preflight
                errors.append(f"{prefix}.uri cloud probe failed: {exc}")

    return errors


def load_corpus_manifest(
    path: Path,
    schema_path: Path | None = None,
    *,
    storage_config: dict[str, Any] | None = None,
    probe_cloud: bool = False,
) -> list[CloudCorpusFile]:
    manifest = _load_mapping(path)
    if schema_path is not None:
        schema = load_schema(schema_path)
        errors = validate_corpus_manifest(
            manifest,
            schema,
            storage_config=storage_config,
            probe_cloud=probe_cloud,
        )
        if errors:
            raise ValueError("invalid corpus manifest:\n- " + "\n- ".join(errors))

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("corpus manifest must contain non-empty 'files' list")

    corpus_files: list[CloudCorpusFile] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("each corpus manifest entry must be an object")
        segment = entry.get("segment")
        uri = entry.get("uri")
        license_name = entry.get("license")
        content_hash = entry.get("content_hash")
        if not isinstance(segment, str) or not segment:
            raise ValueError("each corpus file must define non-empty segment")
        if not isinstance(uri, str) or not uri:
            raise ValueError("each corpus file must define non-empty uri")
        reject_local_path(uri)
        if not isinstance(license_name, str) or not license_name:
            raise ValueError(f"{uri}: license is required")
        if not isinstance(content_hash, dict):
            raise ValueError(f"{uri}: content_hash is required")
        corpus_files.append(
            CloudCorpusFile(
                segment=segment,
                uri=uri,
                license=license_name,
                content_hash={
                    "algorithm": str(content_hash.get("algorithm", "")),
                    "value": str(content_hash.get("value", "")),
                },
            )
        )
    return corpus_files
