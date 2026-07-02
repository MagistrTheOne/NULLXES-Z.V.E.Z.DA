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
        roundtrip = report.get("utf8_roundtrip")
        if roundtrip is not True:
            errors.append("utf8 roundtrip check failed or missing")

    if gates.get("whitespace_sensitive_roundtrip_required"):
        ws_roundtrip = report.get("whitespace_roundtrip")
        if ws_roundtrip is not True:
            errors.append("whitespace-sensitive roundtrip check failed or missing")

    collision_max = gates.get("special_token_collision_max")
    if isinstance(collision_max, int):
        collisions = report.get("special_token_collisions", 0)
        if collisions > collision_max:
            errors.append(f"special token collisions {collisions} exceed max {collision_max}")

    ru_en_max = gates.get("ru_fertility_ratio_vs_english_max")
    if isinstance(ru_en_max, (int, float)):
        ratio = _segment_ratio(report, "russian_high_quality", "english_web", "tokens_per_char")
        if ratio is None:
            ratio = _segment_ratio(report, "ru_clean", "en_clean", "tokens_per_char")
        if ratio is None:
            errors.append("cannot compute RU/EN fertility ratio: missing russian_high_quality or english_web segments")
        elif ratio > ru_en_max:
            errors.append(f"RU/EN fertility ratio {ratio:.4f} exceeds max {ru_en_max}")

    code_qwen_max = gates.get("code_fertility_ratio_vs_qwen2_5_max")
    if isinstance(code_qwen_max, (int, float)):
        code_ratio = _segment_ratio(report, "code", "english_web", "tokens_per_byte")
        if code_ratio is None:
            code_ratio = _segment_ratio(report, "code_python", "en_clean", "tokens_per_byte")
        baseline_ratio = None
        if baseline_report is not None:
            baseline_ratio = _segment_ratio(baseline_report, "code", "english_web", "tokens_per_byte")
            if baseline_ratio is None:
                baseline_ratio = _segment_ratio(baseline_report, "code_python", "en_clean", "tokens_per_byte")
        if code_ratio is None:
            errors.append("cannot compute code fertility ratio: missing code segment")
        elif baseline_ratio is None:
            errors.append("code_fertility_ratio_vs_qwen2_5 gate requires --baseline-fertility-report")
        elif code_ratio / baseline_ratio > code_qwen_max:
            errors.append(
                f"code/Qwen fertility ratio {code_ratio / baseline_ratio:.4f} exceeds max {code_qwen_max}"
            )

    json_tool_max = gates.get("json_tool_overhead_ratio_vs_qwen_mimo_max")
    if isinstance(json_tool_max, (int, float)):
        tool_ratio = segments.get("tool_action_state", {}).get("tokens_per_char")
        if tool_ratio is None:
            tool_ratio = segments.get("json_tool_calls", {}).get("tokens_per_char")
        baseline_tool = None
        if baseline_report is not None:
            baseline_tool = baseline_report.get("segments", {}).get("json_tool_calls", {}).get("tokens_per_char")
            if baseline_tool is None:
                baseline_tool = baseline_report.get("segments", {}).get("tool_action_state", {}).get("tokens_per_char")
        if tool_ratio is None:
            errors.append("cannot compute tool overhead: missing tool_action_state or json_tool_calls segment")
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
