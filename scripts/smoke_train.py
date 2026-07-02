from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.tokenizer.manifest import load_corpus_manifest
from zvezda.train.loop import run_smoke_train


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Z.V.E.Z.D.A smoke training on cloud corpus.")
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()

    launch = load_yaml(args.launch)
    repo_root = Path(launch.get("paths", {}).get("repo_root", REPO_ROOT))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    model_config = load_yaml(repo_root / launch["configs"]["model"])
    storage_key = "storage" if "storage" in launch.get("configs", {}) else "cloud_storage"
    storage_config = load_yaml(repo_root / launch["configs"][storage_key])
    corpus_files = load_corpus_manifest(
        repo_root / launch["configs"]["corpus_manifest"],
        schema_path=repo_root / "configs/tokenizer/corpus-manifest.schema.yaml",
        storage_config=storage_config,
        probe_cloud=True,
    )

    result = run_smoke_train(
        model_config,
        launch,
        corpus_files=corpus_files,
        storage_config=storage_config,
        checkpoint_dir=Path(launch["paths"]["init_checkpoint_dir"]),
        output_dir=Path(launch["paths"]["smoke_output_dir"]),
    )
    print(f"smoke train complete: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
