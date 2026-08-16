"""Crash-safe, resumable JSONL logging.

Colab sessions die mid-sweep. Every generation is appended to disk as soon as
it exists, and each record carries a key so a restarted run can skip work it
has already done (risk table, section 7).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: Path | str) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A truncated final line is the expected signature of a killed
                # session; drop it rather than losing the whole file.
                continue


def write_jsonl(path: Path | str, records: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    return path


class JsonlWriter:
    """Append-only writer that fsyncs each record and tracks completed keys."""

    def __init__(self, path: Path | str, key_fields: tuple[str, ...]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_fields = key_fields
        self.done: set[tuple] = {self._key(r) for r in read_jsonl(self.path)}
        self._fh = self.path.open("a")

    def _key(self, rec: dict) -> tuple:
        return tuple(_norm(rec.get(f)) for f in self.key_fields)

    def has(self, **fields: Any) -> bool:
        return tuple(_norm(fields.get(f)) for f in self.key_fields) in self.done

    def write(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.done.add(self._key(rec))

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _norm(v: Any) -> Any:
    """Floats round-trip through JSON with noise; bucket keys to 6 decimals."""
    return round(v, 6) if isinstance(v, float) else v
