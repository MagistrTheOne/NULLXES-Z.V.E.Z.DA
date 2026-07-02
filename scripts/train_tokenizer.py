from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.tokenizer.gates import load_fertility_report, validate_gates
from zvezda.tokenizer.manifest import load_corpus_manifest, load_schema, validate_corpus_manifest
from zvezda.tokenizer.trainer import fertility_report, save_outputs, train_tokenizer


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Z.V.E.Z.D.A BPE tokenizer on a real corpus manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--manifest-schema", type=Path, default=REPO_ROOT / "configs/tokenizer/corpus-manifest.schema.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-fertility-report", type=Path, default=None)
    parser.add_argument("--skip-gates", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    corpus_files = load_corpus_manifest(args.corpus_manifest, schema_path=args.manifest_schema)
    tokenizer = train_tokenizer(config, corpus_files)
    report = fertility_report(tokenizer, corpus_files, config=config)
    save_outputs(tokenizer, args.output_dir, report, config=config, repo_root=REPO_ROOT)

    if not args.skip_gates:
        baseline = load_fertility_report(args.baseline_fertility_report) if args.baseline_fertility_report else None
        gate_errors = validate_gates(config, report, baseline_report=baseline)
        if gate_errors:
            print("tokenizer gate validation failed:", file=sys.stderr)
            for error in gate_errors:
                print(f"- {error}", file=sys.stderr)
            return 1

    print(f"tokenizer bundle written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
