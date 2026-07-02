"""Special token helpers for Z.V.E.Z.D.A tokenizer."""

from __future__ import annotations

from typing import Any


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
