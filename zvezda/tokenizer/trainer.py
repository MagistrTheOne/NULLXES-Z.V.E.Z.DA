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

from zvezda.tokenizer.special_tokens import flatten_special_tokens


@dataclass(frozen=True)
class CorpusFile:
    segment: str
    path: Path


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


def _quality_checks(tokenizer: Any, corpus_files: list[CorpusFile], config: dict[str, Any]) -> dict[str, Any]:
    special_tokens = flatten_special_tokens(config)
    unk_tokens = 0
    total_tokens = 0
    collisions = 0
    utf8_ok = True
    whitespace_ok = True

    for token in special_tokens:
        probe = tokenizer.encode(token)
        if len(probe.ids) != 1:
            collisions += 1

    for _, normalized in iter_segmented_normalized_lines(corpus_files):
        encoded = tokenizer.encode(normalized)
        total_tokens += len(encoded.ids)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        if decoded.encode("utf-8") != normalized.encode("utf-8"):
            utf8_ok = False
        if "  " in normalized or normalized.startswith(" ") or normalized.endswith(" "):
            if decoded != normalized:
                whitespace_ok = False

    return {
        "unk_rate": unk_tokens / max(total_tokens, 1),
        "utf8_roundtrip": utf8_ok,
        "whitespace_roundtrip": whitespace_ok,
        "special_token_collisions": collisions,
    }


def fertility_report(tokenizer: Any, corpus_files: list[CorpusFile], config: dict[str, Any] | None = None) -> dict[str, Any]:
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

    report: dict[str, Any] = {"segments": segments}
    if config is not None:
        report.update(_quality_checks(tokenizer, corpus_files, config))
    return report


def save_outputs(
    tokenizer: Any,
    output_dir: Path,
    report: dict[str, Any],
    *,
    config: dict[str, Any],
    repo_root: Path,
) -> None:
    from zvezda.tokenizer.bundle import write_bundle

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    model = tokenizer.model
    if hasattr(model, "save"):
        model.save(str(output_dir))
    with (output_dir / "fertility_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    actual_vocab_size = tokenizer.get_vocab_size()
    write_bundle(config, output_dir, repo_root=repo_root, actual_vocab_size=actual_vocab_size)
