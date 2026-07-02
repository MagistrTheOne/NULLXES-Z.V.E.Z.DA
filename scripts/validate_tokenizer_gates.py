from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.tokenizer.gates import load_fertility_report, validate_gates


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tokenizer fertility gates for a trained bundle.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fertility-report", type=Path, required=True)
    parser.add_argument("--baseline-fertility-report", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    report = load_fertility_report(args.fertility_report)
    baseline = load_fertility_report(args.baseline_fertility_report) if args.baseline_fertility_report else None
    errors = validate_gates(config, report, baseline_report=baseline)
    if errors:
        print("tokenizer gate validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"tokenizer gates passed: {args.fertility_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
