"""RunPod network volume access for Z.V.E.Z.D.A corpus (no AWS, no S3 API)."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


RUNPOD_SCHEME = "runpod"
FORBIDDEN_SCHEMES = frozenset({"file", "", "s3", "gs", "r2", "https", "runpod-s3"})
AWS_ENV_PREFIXES = ("AWS_",)


@dataclass(frozen=True)
class StorageUri:
    raw: str
    scheme: str
    key: str


def _reject_aws_env() -> None:
    for name in os.environ:
        if name.startswith(AWS_ENV_PREFIXES):
            raise RuntimeError(f"AWS credentials are not allowed on RunPod launch: {name}")


def parse_storage_uri(uri: str) -> StorageUri:
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("corpus uri must be a non-empty string")
    parsed = urlparse(uri.strip())
    scheme = parsed.scheme.lower()
    if scheme in FORBIDDEN_SCHEMES:
        if scheme == "runpod-s3":
            raise ValueError(f"runpod-s3:// is not supported on pod; use runpod:// volume paths only: {uri}")
        raise ValueError(f"forbidden uri scheme {scheme!r} on RunPod launch: {uri}")
    if scheme != RUNPOD_SCHEME:
        raise ValueError(f"unsupported uri scheme {scheme!r}: use runpod://<volume-relative-path>")
    if parsed.netloc and parsed.path:
        key = f"{parsed.netloc}/{parsed.path.lstrip('/')}"
    elif parsed.netloc:
        key = parsed.netloc
    else:
        key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"runpod:// uri must include a volume-relative path: {uri}")
    return StorageUri(raw=uri, scheme=scheme, key=key)


def is_storage_uri(uri: str) -> bool:
    try:
        parse_storage_uri(uri)
        return True
    except ValueError:
        return False


def reject_local_path(uri: str) -> None:
    if re.match(r"^[A-Za-z]:\\", uri) or uri.startswith(("/", "./", "../", "\\")):
        raise ValueError(f"local machine paths are not allowed: {uri}")
    parse_storage_uri(uri)


def volume_root(storage_config: dict[str, Any] | None) -> Path:
    providers = (storage_config or {}).get("providers", {})
    runpod = providers.get("runpod", {})
    env_name = runpod.get("volume_root_env", "RUNPOD_VOLUME_ROOT")
    default = runpod.get("default_volume_root", "/workspace")
    return Path(os.environ.get(env_name, default))


def resolve_runpod_volume_path(uri: StorageUri, storage_config: dict[str, Any] | None) -> Path:
    path = volume_root(storage_config) / uri.key
    if not path.is_file():
        raise FileNotFoundError(f"runpod volume object not found: {path} (uri={uri.raw})")
    return path


def open_storage_binary(uri: str, storage_config: dict[str, Any] | None = None):
    _reject_aws_env()
    storage_uri = parse_storage_uri(uri)
    path = resolve_runpod_volume_path(storage_uri, storage_config)
    return path.open("rb")


def iter_storage_text_lines(uri: str, storage_config: dict[str, Any] | None = None) -> Iterator[str]:
    with open_storage_binary(uri, storage_config) as handle:
        for raw in handle:
            yield raw.decode("utf-8")


def probe_storage_uri(uri: str, storage_config: dict[str, Any] | None = None, nbytes: int = 4096) -> int:
    with open_storage_binary(uri, storage_config) as handle:
        return len(handle.read(nbytes))


def required_env_present(storage_config: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    _reject_aws_env()

    preflight = (storage_config or {}).get("preflight", {})
    if preflight.get("require_runpod_pod"):
        runpod_signals = preflight.get("runpod_env_any", [])
        if runpod_signals and not any(os.environ.get(name) for name in runpod_signals):
            errors.append(f"not on RunPod pod: set one of {runpod_signals}")

    root = volume_root(storage_config)
    if not root.exists():
        errors.append(f"RunPod volume root missing: {root}")
    return errors


# Backward-compatible aliases used during module migration.
parse_cloud_uri = parse_storage_uri
is_cloud_uri = is_storage_uri
open_cloud_binary = open_storage_binary
iter_cloud_text_lines = iter_storage_text_lines
probe_cloud_uri = probe_storage_uri
