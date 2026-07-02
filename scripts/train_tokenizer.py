from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.tokenizer.trainer import fertility_report, load_corpus_manifest, save_outputs, train_tokenizer


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    corpus_files = load_corpus_manifest(args.corpus_manifest)
    tokenizer = train_tokenizer(config, corpus_files)
    report = fertility_report(tokenizer, corpus_files)
    save_outputs(tokenizer, args.output_dir, report)
    print(f"tokenizer written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
