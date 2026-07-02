from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.tokenizer.manifest import load_schema, validate_corpus_manifest


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a tokenizer corpus manifest against schema.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=REPO_ROOT / "configs/tokenizer/corpus-manifest.schema.yaml")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    schema = load_schema(args.schema)
    errors = validate_corpus_manifest(manifest, schema)
    if errors:
        print("corpus manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"corpus manifest valid: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
