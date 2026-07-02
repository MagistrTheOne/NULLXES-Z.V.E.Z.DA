from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.config.validation import validate


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Z.V.E.Z.D.A model config accounting.")
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()

    failed = False
    for config_path in args.configs:
        errors = validate(load_yaml(config_path))
        if errors:
            failed = True
            print(f"{config_path}: FAIL", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        else:
            print(f"{config_path}: OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
