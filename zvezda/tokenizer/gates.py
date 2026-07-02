"""Fertility gate validation for trained tokenizer bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _segment_ratio(report: dict[str, Any], numerator: str, denominator: str, field: str) -> float | None:
    segments = report.get("segments", {})
    num = segments.get(numerator, {}).get(field)
    den = segments.get(denominator, {}).get(field)
    if not isinstance(num, (int, float)) or not isinstance(den, (int, float)) or den <= 0:
        return None
    return float(num) / float(den)


def _require_segments(report: dict[str, Any], names: list[str]) -> list[str]:
    segments = report.get("segments", {})
    return [name for name in names if name not in segments]


def validate_gates(
    config: dict[str, Any],
    report: dict[str, Any],
    *,
    baseline_report: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    gates = config.get("gates")
    if not isinstance(gates, dict):
        return ["config must define gates mapping"]

    segments = report.get("segments", {})
    if not isinstance(segments, dict) or not segments:
        return ["fertility report must contain non-empty segments"]

    missing_required = _require_segments(report, list(gates.get("required_segments", [])))
    for segment in missing_required:
        errors.append(f"required segment missing from fertility report: {segment}")

    missing_report = _require_segments(report, list(gates.get("report_required", [])))
    for segment in missing_report:
        errors.append(f"report-required segment missing from fertility report: {segment}")

    unk_rate_max = gates.get("unk_rate_max")
    if isinstance(unk_rate_max, (int, float)):
        unk_rate = report.get("unk_rate")
        if not isinstance(unk_rate, (int, float)):
            errors.append("fertility report missing unk_rate")
        elif unk_rate > unk_rate_max:
            errors.append(f"unk_rate {unk_rate} exceeds max {unk_rate_max}")

    if gates.get("utf8_roundtrip_required"):
        if report.get("utf8_roundtrip") is not True:
            errors.append("utf8 roundtrip check failed or missing")

    if gates.get("whitespace_sensitive_roundtrip_required"):
        if report.get("whitespace_roundtrip") is not True:
            errors.append("whitespace-sensitive roundtrip check failed or missing")

    collision_max = gates.get("special_token_collision_max")
    if isinstance(collision_max, int):
        collisions = report.get("special_token_collisions", 0)
        if collisions > collision_max:
            errors.append(f"special token collisions {collisions} exceed max {collision_max}")

    ru_code_max = gates.get("ru_fertility_ratio_vs_code_max")
    if isinstance(ru_code_max, (int, float)):
        ratio = _segment_ratio(report, "ru_curated", "code_verified", "tokens_per_char")
        if ratio is None:
            ratio = _segment_ratio(report, "ru_clean", "code_python", "tokens_per_char")
        if ratio is None:
            errors.append("cannot compute RU/code fertility ratio")
        elif ratio > ru_code_max:
            errors.append(f"RU/code fertility ratio {ratio:.4f} exceeds max {ru_code_max}")

    code_baseline_max = gates.get("code_fertility_ratio_vs_baseline_max")
    if isinstance(code_baseline_max, (int, float)):
        code_ratio = _segment_ratio(report, "code_verified", "ru_curated", "tokens_per_byte")
        if code_ratio is None:
            code_ratio = _segment_ratio(report, "code_python", "ru_clean", "tokens_per_byte")
        baseline_ratio = None
        if baseline_report is not None:
            baseline_ratio = _segment_ratio(baseline_report, "code_verified", "ru_curated", "tokens_per_byte")
            if baseline_ratio is None:
                baseline_ratio = _segment_ratio(baseline_report, "code_python", "ru_clean", "tokens_per_byte")
        if code_ratio is None:
            errors.append("cannot compute code fertility ratio")
        elif baseline_ratio is None:
            errors.append("code_fertility_ratio_vs_baseline gate requires --baseline-fertility-report")
        elif code_ratio / baseline_ratio > code_baseline_max:
            errors.append(
                f"code/baseline fertility ratio {code_ratio / baseline_ratio:.4f} exceeds max {code_baseline_max}"
            )

    json_tool_max = gates.get("json_tool_overhead_ratio_vs_baseline_max")
    if isinstance(json_tool_max, (int, float)):
        tool_ratio = segments.get("tool_action_traces", {}).get("tokens_per_char")
        if tool_ratio is None:
            tool_ratio = segments.get("json_tool_calls", {}).get("tokens_per_char")
        baseline_tool = None
        if baseline_report is not None:
            baseline_tool = baseline_report.get("segments", {}).get("json_tool_calls", {}).get("tokens_per_char")
            if baseline_tool is None:
                baseline_tool = baseline_report.get("segments", {}).get("tool_action_traces", {}).get("tokens_per_char")
        if tool_ratio is None:
            errors.append("cannot compute tool overhead: missing tool_action_traces or json_tool_calls segment")
        elif baseline_tool is None:
            errors.append("json_tool_overhead gate requires --baseline-fertility-report")
        elif float(tool_ratio) / float(baseline_tool) > json_tool_max:
            errors.append(
                f"json/tool overhead ratio {float(tool_ratio) / float(baseline_tool):.4f} exceeds max {json_tool_max}"
            )

    return errors


def load_fertility_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data
