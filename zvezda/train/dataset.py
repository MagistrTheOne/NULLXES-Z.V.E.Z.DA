"""Cloud-backed tokenized dataset for smoke training."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset

from zvezda.data.corpus import CloudCorpusFile, iter_normalized_lines


@dataclass(frozen=True)
class PackedBatch:
    input_ids: torch.Tensor


class CloudTokenStream(IterableDataset):
    def __init__(
        self,
        corpus_files: list[CloudCorpusFile],
        tokenizer_json: Path,
        *,
        seq_len: int,
        storage_config: dict[str, Any] | None = None,
    ) -> None:
        self.corpus_files = corpus_files
        self.tokenizer_json = tokenizer_json
        self.seq_len = seq_len
        self.storage_config = storage_config

    def _encode_lines(self) -> Iterator[list[int]]:
        from tokenizers import Tokenizer  # pyright: ignore[reportMissingImports]

        tokenizer = Tokenizer.from_file(str(self.tokenizer_json))
        buffer: list[int] = []
        for line in iter_normalized_lines(self.corpus_files, self.storage_config):
            ids = tokenizer.encode(line).ids
            buffer.extend(ids)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                yield chunk

    def __iter__(self) -> Iterator[PackedBatch]:
        for chunk in self._encode_lines():
            tensor = torch.tensor(chunk, dtype=torch.long)
            yield PackedBatch(input_ids=tensor.unsqueeze(0))
