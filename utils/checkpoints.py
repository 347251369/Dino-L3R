"""Checkpoint integrity and immutable final-selection helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path, expected_sha256: str) -> Path:
    checkpoint = Path(path)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch for {checkpoint}: {actual} != {expected_sha256}"
        )
    return checkpoint
