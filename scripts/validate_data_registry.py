from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.data.registry import validate_source_registry


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Z.V.E.Z.D.A data source registry.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("configs/data/source-registry.schema.yaml"))
    args = parser.parse_args()

    errors = validate_source_registry(load_yaml(args.registry), load_yaml(args.schema))
    if errors:
        print(f"{args.registry}: FAIL", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"{args.registry}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
