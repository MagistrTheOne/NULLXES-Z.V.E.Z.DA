from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # pyright: ignore[reportMissingModuleSource]

from zvezda.config.validation import validate
from zvezda.init.writer import write_init_checkpoint


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a streaming BF16 init checkpoint for Z.V.E.Z.D.A.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-seed", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    args = parser.parse_args()

    config = load_yaml(args.config)
    policy = load_yaml(args.init_policy)
    errors = validate(config)
    if errors:
        print("config validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    dtype_name = policy.get("canonical_checkpoint", {}).get("dtype", "bf16")
    if dtype_name != "bf16":
        print("only bf16 canonical init is implemented", file=sys.stderr)
        return 1

    manifest = write_init_checkpoint(
        config,
        args.output_dir,
        global_seed=args.global_seed,
        shard_count=args.shard_count,
    )
    print(f"init checkpoint written to {args.output_dir} ({manifest['tensor_count']} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
