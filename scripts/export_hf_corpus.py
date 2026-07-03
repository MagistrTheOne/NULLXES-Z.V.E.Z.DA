"""Export Hugging Face curated sources to RunPod volume corpus + manifest."""

from __future__ import annotations

import argparse
import hashlib
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # pyright: ignore[reportMissingModuleSource]
from huggingface_hub import hf_hub_download

from zvezda.tokenizer.manifest import load_schema, validate_corpus_manifest


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.file_digest(path.open("rb"), "sha256")
    return digest.hexdigest()


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _iter_paragraphs(raw: str, *, separator: str, flatten: bool) -> list[str]:
    chunks = raw.split(separator)
    paragraphs: list[str] = []
    for chunk in chunks:
        text = _normalize(chunk.strip())
        if not text:
            continue
        if flatten:
            text = " ".join(text.split())
        paragraphs.append(text)
    return paragraphs


def _write_shards(
    paragraphs: list[str],
    *,
    output_dir: Path,
    prefix: str,
    shard_max_bytes: int,
    max_total_bytes: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    total_bytes = 0
    shard_index = 0
    current_lines: list[str] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal shard_index, current_lines, current_bytes, total_bytes
        if not current_lines:
            return
        shard_path = output_dir / f"{prefix}-{shard_index:04d}.txt"
        body = "\n".join(current_lines) + "\n"
        shard_path.write_text(body, encoding="utf-8")
        written.append(shard_path)
        total_bytes += shard_path.stat().st_size
        shard_index += 1
        current_lines = []
        current_bytes = 0

    for paragraph in paragraphs:
        line_bytes = len(paragraph.encode("utf-8")) + 1
        if current_lines and current_bytes + line_bytes > shard_max_bytes:
            flush()
        if total_bytes + line_bytes > max_total_bytes and not current_lines:
            break
        if total_bytes >= max_total_bytes:
            break
        current_lines.append(paragraph)
        current_bytes += line_bytes
        if current_bytes >= shard_max_bytes:
            flush()
            if total_bytes >= max_total_bytes:
                break

    flush()
    return written


def _runpod_uri(corpus_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(corpus_root).as_posix()
    return f"runpod://corpus/{rel}"


def export_source(
    source: dict,
    *,
    corpus_root: Path,
    export_cfg: dict,
) -> list[dict]:
    repo = str(source["hf_repo"])
    filename = str(source["hf_filename"])
    local_path = Path(
        hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
    )
    raw = local_path.read_text(encoding="utf-8")
    paragraphs = _iter_paragraphs(
        raw,
        separator=str(export_cfg.get("paragraph_separator", "\n\n")),
        flatten=bool(export_cfg.get("flatten_paragraphs", True)),
    )
    if not paragraphs:
        raise RuntimeError(f"{repo}/{filename}: no paragraphs extracted")

    output_dir = corpus_root / str(source["output_dir"])
    shards = _write_shards(
        paragraphs,
        output_dir=output_dir,
        prefix=str(source.get("output_prefix", "part")),
        shard_max_bytes=int(export_cfg.get("shard_max_bytes", 33_554_432)),
        max_total_bytes=int(export_cfg.get("max_total_bytes", 104_857_600)),
    )
    if not shards:
        raise RuntimeError(f"{source['id']}: export produced no shard files")

    entries: list[dict] = []
    for shard in shards:
        entries.append(
            {
                "segment": source["segment"],
                "uri": _runpod_uri(corpus_root, shard),
                "license": source["license"],
                "content_hash": {
                    "algorithm": "sha256",
                    "value": _sha256_file(shard),
                },
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Export HF curated corpus shards to RunPod volume.")
    parser.add_argument("--export", type=Path, required=True, help="Export recipe YAML.")
    parser.add_argument("--write-manifest", action="store_true", help="Write manifest (default if export.write_manifest=true).")
    parser.add_argument("--dry-run", action="store_true", help="Validate export config only.")
    args = parser.parse_args()

    cfg = load_yaml(args.export)
    paths = cfg["paths"]
    corpus_root = Path(paths["corpus_root"])
    repo_root = Path(paths.get("repo_root", REPO_ROOT))
    manifest_out = repo_root / paths["manifest_out"]
    export_cfg = cfg.get("export", {})
    write_manifest = args.write_manifest or bool(export_cfg.get("write_manifest", False))

    sources = cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        print("export config must contain non-empty sources list", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"export config OK: {len(sources)} source(s), corpus_root={corpus_root}")
        return 0

    manifest_files: list[dict] = []
    for source in sources:
        if not isinstance(source, dict):
            print("each source must be a mapping", file=sys.stderr)
            return 1
        print(f"exporting {source.get('id', source.get('hf_repo'))} ...")
        manifest_files.extend(export_source(source, corpus_root=corpus_root, export_cfg=export_cfg))

    manifest = {
        "schema_version": 1,
        "program": "Z.V.E.Z.D.A",
        "purpose": cfg.get("purpose", "runpod_init_curriculum"),
        "storage_policy": "runpod_only",
        "launch_provider": "runpod",
        "no_mock_data": True,
        "no_stubs": True,
        "files": manifest_files,
    }

    schema = load_schema(repo_root / "configs/tokenizer/corpus-manifest.schema.yaml")
    errors = validate_corpus_manifest(manifest, schema, probe_cloud=False)
    if errors:
        print("manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    if write_manifest:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with manifest_out.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)
        print(f"manifest written: {manifest_out}")

    print(f"exported {len(manifest_files)} shard(s) under {corpus_root}")
    for entry in manifest_files:
        print(f"  {entry['uri']}  sha256={entry['content_hash']['value'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
