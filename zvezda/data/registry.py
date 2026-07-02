"""Pretraining data source registry validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_source_registry(registry: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    sources = registry.get("sources")
    if not isinstance(sources, list):
        return ["registry must contain a sources list"]

    required = schema.get("required_source_fields")
    if not isinstance(required, list):
        return ["schema must contain required_source_fields list"]

    allowed_status = set(schema.get("allowed_status", []))
    seen_ids: set[str] = set()

    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            errors.append(f"sources[{index}] must be an object")
            continue
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"sources[{index}].source_id must be non-empty string")
            continue
        if source_id in seen_ids:
            errors.append(f"duplicate source_id: {source_id}")
        seen_ids.add(source_id)

        for field in required:
            if field not in source:
                errors.append(f"{source_id}: missing required field {field}")

        status = source.get("status")
        if status not in allowed_status:
            errors.append(f"{source_id}: invalid status {status!r}")

        if status == "accepted":
            errors.extend(_validate_accepted_source(source_id, source))
        if status == "rejected" and not source.get("rejected_reason"):
            errors.append(f"{source_id}: rejected source must include rejected_reason")

    return errors


def _validate_accepted_source(source_id: str, source: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if source.get("license") in (None, "", "unknown"):
        errors.append(f"{source_id}: accepted source must have known license")

    content_hash = source.get("content_hash")
    if not isinstance(content_hash, Mapping) or not content_hash.get("algorithm") or not content_hash.get("value"):
        errors.append(f"{source_id}: accepted source must have content_hash.algorithm and content_hash.value")

    dedup = source.get("dedup")
    if not isinstance(dedup, Mapping) or dedup.get("exact") is not True or dedup.get("minhash") is not True:
        errors.append(f"{source_id}: accepted source requires exact and minhash dedup")

    pii_scrub = source.get("pii_scrub")
    if not isinstance(pii_scrub, Mapping) or pii_scrub.get("completed") is not True:
        errors.append(f"{source_id}: accepted source requires completed PII scrub")

    fertility = source.get("tokenizer_fertility")
    if not isinstance(fertility, Mapping) or fertility.get("measured") is not True:
        errors.append(f"{source_id}: accepted source requires measured tokenizer fertility")

    return errors
