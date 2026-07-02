from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.config.validation import validate
from zvezda.data.cloud import required_env_present
from zvezda.tokenizer.manifest import load_schema, validate_corpus_manifest


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _reject_aws_env() -> list[str]:
    return [name for name in os.environ if name.startswith("AWS_")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks before Z.V.E.Z.D.A RunPod launch.")
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--probe-storage", action="store_true")
    parser.add_argument("--require-artifacts", action="store_true")
    args = parser.parse_args()

    launch = load_yaml(args.launch)
    repo_root = Path(launch.get("paths", {}).get("repo_root", REPO_ROOT))
    errors: list[str] = []

    if launch.get("pod", {}).get("provider") != "runpod":
        errors.append("pod.provider must be runpod")

    aws_env = _reject_aws_env()
    if aws_env:
        errors.append(f"AWS env vars forbidden on RunPod launch: {', '.join(aws_env)}")

    if not torch.cuda.is_available():
        errors.append("CUDA is not available on this pod")

    model_path = repo_root / launch["configs"]["model"]
    model_config = load_yaml(model_path)
    errors.extend(validate(model_config))

    storage_key = "storage" if "storage" in launch.get("configs", {}) else "cloud_storage"
    storage_path = repo_root / launch["configs"][storage_key]
    if "cloud-storage-v0" in storage_path.as_posix():
        errors.append("cloud-storage-v0.yaml is retired; use runpod-storage-v0.yaml")

    storage_config = load_yaml(storage_path)
    errors.extend(required_env_present(storage_config))

    manifest = load_yaml(repo_root / launch["configs"]["corpus_manifest"])
    schema = load_schema(repo_root / "configs/tokenizer/corpus-manifest.schema.yaml")
    errors.extend(
        validate_corpus_manifest(
            manifest,
            schema,
            storage_config=storage_config,
            probe_cloud=args.probe_storage,
        )
    )

    if args.require_artifacts:
        tokenizer_bundle = Path(launch["paths"]["tokenizer_bundle_dir"])
        for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            if not (tokenizer_bundle / filename).exists():
                errors.append(f"tokenizer bundle missing {filename}; run tokenizer_train phase first")
        init_dir = Path(launch["paths"]["init_checkpoint_dir"])
        if not any(init_dir.glob("model-*-of-*.safetensors")):
            errors.append("init checkpoint shards missing; run init_checkpoint phase first")

    if errors:
        print("pod preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("pod preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
