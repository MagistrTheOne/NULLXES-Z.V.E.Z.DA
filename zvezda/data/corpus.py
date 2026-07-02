"""Corpus iteration from RunPod storage manifests."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any
import unicodedata

from zvezda.data.cloud import iter_storage_text_lines


@dataclass(frozen=True)
class CloudCorpusFile:
    segment: str
    uri: str
    license: str
    content_hash: dict[str, str]


def normalize_line(line: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", line.rstrip("\n\r"))
    return normalized if normalized else None


def lines_to_document(lines: Iterable[str]) -> str | None:
    """Build one training/fertility document from a file's lines."""
    parts: list[str] = []
    for line in lines:
        normalized = normalize_line(line)
        if normalized is not None:
            parts.append(normalized)
    if not parts:
        return None
    return "\n".join(parts)


def iter_segmented_normalized_lines(
    files: list[CloudCorpusFile],
    storage_config: dict[str, Any] | None = None,
) -> Iterator[tuple[str, str]]:
    for corpus_file in files:
        for line in iter_storage_text_lines(corpus_file.uri, storage_config):
            normalized = normalize_line(line)
            if normalized is not None:
                yield corpus_file.segment, normalized


def iter_segmented_normalized_documents(
    files: list[CloudCorpusFile],
    storage_config: dict[str, Any] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield one normalized document per manifest file for file-level tokenization."""
    for corpus_file in files:
        document = lines_to_document(iter_storage_text_lines(corpus_file.uri, storage_config))
        if document is not None:
            yield corpus_file.segment, document


def iter_normalized_lines(
    files: list[CloudCorpusFile],
    storage_config: dict[str, Any] | None = None,
) -> Iterator[str]:
    for _, normalized in iter_segmented_normalized_lines(files, storage_config):
        yield normalized


def iter_normalized_documents(
    files: list[CloudCorpusFile],
    storage_config: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Yield one document per manifest file — matches fertility_report encoding unit."""
    for _, document in iter_segmented_normalized_documents(files, storage_config):
        yield document
