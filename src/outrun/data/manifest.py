"""``data/manifest.yaml`` parsing and validation (specs/01-data.md DATA-10)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ACCESS: tuple[str, ...] = ("s3_unsigned", "http", "kaggle", "herbie_byte_range")
TARGETS: tuple[str, ...] = ("hdd", "ssd", "stream")
REQUIRED_KEYS: tuple[str, ...] = (
    "id",
    "desc",
    "url",
    "access",
    "approx_bytes",
    "license",
    "target",
    "sha256",
)


class ManifestError(ValueError):
    """Invalid manifest; the message names the offending source id (and key)."""


@dataclass(frozen=True)
class Source:
    """One manifest entry. ``sha256`` is None when the manifest says TBD/null (not yet verified)."""

    id: str
    desc: str
    url: str
    access: str
    approx_bytes: int
    license: str
    target: str
    sha256: str | None

    @property
    def is_single_file(self) -> bool:
        """True when the URL neither ends with '/' nor contains a brace group."""
        return not self.url.endswith("/") and "{" not in self.url


@dataclass(frozen=True)
class Manifest:
    version: int
    sources: tuple[Source, ...]

    def get(self, source_id: str) -> Source:
        """Source by id; KeyError if absent."""
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)

    def by_target(self, target: str) -> tuple[Source, ...]:
        """Sources with this target, in file order."""
        return tuple(s for s in self.sources if s.target == target)


def parse_manifest(data: Mapping) -> Manifest:
    """DATA-10. Validate a loaded YAML mapping (rules in the spec); raises ManifestError."""
    raise NotImplementedError


def load_manifest(path: Path | str) -> Manifest:
    """DATA-10. Read YAML at ``path`` and ``parse_manifest`` it."""
    raise NotImplementedError
