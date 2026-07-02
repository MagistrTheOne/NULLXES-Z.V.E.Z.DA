"""BPE tokenizer training for Z.V.E.Z.D.A."""

from __future__ import annotations

import importlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import unicodedata


@dataclass(frozen=True)
class CorpusFile:
    segment: str
    path: Path


def flatten_special_tokens(config: dict[str, Any]) -> list[str]:
    special = config.get("special_tokens")
    if not isinstance(special, dict):
        raise ValueError("tokenizer config must define special_tokens mapping")
    tokens: list[str] = []
    for group_tokens in special.values():
        if not isinstance(group_tokens, list):
            raise ValueError("special token groups must be lists")
        tokens.extend(str(token) for token in group_tokens)
    if len(tokens) != len(set(tokens)):
        duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
        raise ValueError(f"duplicate special tokens: {duplicates}")
    return tokens


def load_corpus_manifest(path: Path) -> list[CorpusFile]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("corpus manifest must contain non-empty 'files' list")

    corpus_files: list[CorpusFile] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("each corpus manifest entry must be an object")
        segment = entry.get("segment")
        raw_path = entry.get("path")
        if not isinstance(segment, str) or not segment:
            raise ValueError("each corpus file must define non-empty segment")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("each corpus file must define non-empty path")
        file_path = Path(raw_path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        if not file_path.is_file():
            raise ValueError(f"corpus path is not a file: {file_path}")
        corpus_files.append(CorpusFile(segment=segment, path=file_path))
    return corpus_files


def iter_segmented_normalized_lines(files: Iterable[CorpusFile]) -> Iterable[tuple[str, str]]:
    for corpus_file in files:
        with corpus_file.path.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                normalized = unicodedata.normalize("NFKC", line.rstrip("\n"))
                if normalized:
                    yield corpus_file.segment, normalized


def iter_normalized_lines(files: Iterable[CorpusFile]) -> Iterable[str]:
    for _, normalized in iter_segmented_normalized_lines(files):
        yield normalized


def _tokenizers() -> tuple[Any, Any, Any, Any, Any, Any]:
    tokenizers_module = importlib.import_module("tokenizers")
    decoders_module = importlib.import_module("tokenizers.decoders")
    models_module = importlib.import_module("tokenizers.models")
    normalizers_module = importlib.import_module("tokenizers.normalizers")
    pre_tokenizers_module = importlib.import_module("tokenizers.pre_tokenizers")
    trainers_module = importlib.import_module("tokenizers.trainers")
    return tokenizers_module.Tokenizer, decoders_module, models_module, normalizers_module, pre_tokenizers_module, trainers_module


def build_tokenizer(config: dict[str, Any]) -> Any:
    tokenizer_cfg = config.get("tokenizer")
    if not isinstance(tokenizer_cfg, dict):
        raise ValueError("config must contain tokenizer mapping")
    if tokenizer_cfg.get("algorithm") != "bpe":
        raise ValueError("only BPE tokenizer training is implemented")

    tokenizer_cls, decoders, models, normalizers, pre_tokenizers, _ = _tokenizers()
    tokenizer = tokenizer_cls(models.BPE(byte_fallback=bool(tokenizer_cfg.get("byte_fallback", True)), unk_token=None))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=bool(tokenizer_cfg.get("add_prefix_space", False)))
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(config: dict[str, Any], corpus_files: list[CorpusFile]) -> Any:
    tokenizer_cfg = config["tokenizer"]
    special_tokens = flatten_special_tokens(config)
    tokenizer = build_tokenizer(config)
    _, _, _, _, _, trainers = _tokenizers()
    trainer = trainers.BpeTrainer(
        vocab_size=int(tokenizer_cfg["vocab_size"]),
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_normalized_lines(corpus_files), trainer=trainer)
    return tokenizer


def fertility_report(tokenizer: Any, corpus_files: list[CorpusFile]) -> dict[str, Any]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"chars": 0, "bytes": 0, "tokens": 0, "records": 0})
    for segment, normalized in iter_segmented_normalized_lines(corpus_files):
        encoded = tokenizer.encode(normalized)
        segment_stats = stats[segment]
        segment_stats["chars"] += len(normalized)
        segment_stats["bytes"] += len(normalized.encode("utf-8"))
        segment_stats["tokens"] += len(encoded.ids)
        segment_stats["records"] += 1

    segments: dict[str, Any] = {}
    for segment, segment_stats in sorted(stats.items()):
        chars = max(segment_stats["chars"], 1)
        byte_count = max(segment_stats["bytes"], 1)
        tokens = segment_stats["tokens"]
        segments[segment] = {
            **segment_stats,
            "tokens_per_char": tokens / chars,
            "tokens_per_byte": tokens / byte_count,
            "chars_per_token": chars / max(tokens, 1),
        }
    return {"segments": segments}


def save_outputs(tokenizer: Any, output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    model = tokenizer.model
    if hasattr(model, "save"):
        model.save(str(output_dir))
    with (output_dir / "fertility_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
