"""Resumable, checksummed downloads from the manifest (specs/01-data.md DATA-11, DATA-12, DATA-13).

Network access goes through an injected ``Transport`` so the resume/verify logic is testable offline. Default
transports (requests for HTTP, unsigned boto3 for S3) are thin and exercised only in T1.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from outrun.data.manifest import Source


class DownloadError(OSError):
    """Bad HTTP status, size mismatch or an existing file of the wrong size (DATA-11)."""


class ChecksumError(OSError):
    """sha256 mismatch (DATA-11, DATA-12)."""


@dataclass
class RangeResponse:
    """status 206 = range honoured (append), 200 = whole body from byte 0 (restart)."""

    status: int
    chunks: Iterable[bytes]


class Transport(Protocol):
    def get(self, url: str, start: int = 0) -> RangeResponse: ...


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    bytes_downloaded: int
    skipped: bool


@dataclass(frozen=True)
class PlannedFile:
    key: str
    size: int
    dest: Path


@dataclass(frozen=True)
class VerifyReport:
    ok: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unrecorded: list[str] = field(default_factory=list)


def download(
    url: str,
    dest: Path | str,
    transport: Transport,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> DownloadResult:
    """DATA-11. Resume from ``dest.name + '.part'``, verify, atomically rename. See spec steps 1-5."""
    raise NotImplementedError


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """DATA-12. Lowercase hex sha256 of the file, read in chunks."""
    raise NotImplementedError


class FetchState:
    """DATA-12. Persistent JSON ``{source_id: {relpath: {"sha256": str, "bytes": int}}}``; each record is written
    atomically. Missing file = empty state; invalid JSON -> ValueError."""

    def __init__(self, path: Path | str) -> None:
        raise NotImplementedError

    def record(self, source_id: str, relpath: str, sha256: str, nbytes: int) -> None:
        raise NotImplementedError

    def get(self, source_id: str, relpath: str) -> dict | None:
        raise NotImplementedError


def verify_source(source: Source, raw_root: Path | str, state: FetchState) -> VerifyReport:
    """DATA-12. Re-hash recorded files under ``raw_root/source.id``; classify ok/mismatched/missing/unrecorded."""
    raise NotImplementedError


def fetch_source(
    source: Source,
    raw_root: Path | str,
    transport: Transport,
    state: FetchState,
    lister: Callable[[str], Iterable[tuple[str, int]]] | None = None,
) -> list[DownloadResult]:
    """DATA-12. Download a manifest source into ``raw_root/source.id`` and record hashes; stream sources -> ValueError."""
    raise NotImplementedError


def expand_braces(url: str) -> list[str]:
    """DATA-13. Expand ``{a,b}`` groups (cartesian product, left slowest); unbalanced/nested -> ValueError."""
    raise NotImplementedError


def plan_downloads(remote: Iterable[tuple[str, int]], prefix: str, dest_dir: Path | str) -> list[PlannedFile]:
    """DATA-13. Map keys under ``prefix`` to ``dest_dir``; skip files already present with the same size."""
    raise NotImplementedError


def planned_bytes(plan: Sequence[PlannedFile]) -> int:
    """DATA-13. Sum of planned sizes."""
    raise NotImplementedError
