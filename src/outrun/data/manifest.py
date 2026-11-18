"""``data/manifest.yaml`` parsing and validation (specs/01-data.md DATA-10)."""

from __future__ import annotations

import re
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


_ID_RE = re.compile(r"^[a-z0-9_]+$")
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


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


def _parse_sha256(raw: object, target: str, sid: str) -> str | None:
    """TBD / null / (stream only) n/a -> None; 64 hex chars -> lowercased; anything else is an error."""
    if raw is None or raw == "TBD":
        return None
    if raw == "n/a":
        if target == "stream":
            return None
        raise ManifestError(f"source {sid!r}: key 'sha256' may be 'n/a' only for target 'stream'")
    if isinstance(raw, str) and _HEX64_RE.match(raw):
        return raw.lower()
    raise ManifestError(f"source {sid!r}: key 'sha256' must be 'TBD', null or 64 hex characters, got {raw!r}")


def _parse_source(entry: object, index: int) -> Source:
    if not isinstance(entry, Mapping):
        raise ManifestError(f"source #{index}: entry must be a mapping")
    sid = entry.get("id", f"#{index}")
    for key in REQUIRED_KEYS:
        if key not in entry:
            raise ManifestError(f"source {sid!r}: missing key {key!r}")
    if not isinstance(sid, str) or not _ID_RE.match(sid):
        raise ManifestError(f"source {sid!r}: key 'id' must match ^[a-z0-9_]+$")
    access, target = entry["access"], entry["target"]
    if access not in ACCESS:
        raise ManifestError(f"source {sid!r}: key 'access' must be one of {ACCESS}, got {access!r}")
    if target not in TARGETS:
        raise ManifestError(f"source {sid!r}: key 'target' must be one of {TARGETS}, got {target!r}")
    if (target == "stream") != (access == "herbie_byte_range"):
        raise ManifestError(
            f"source {sid!r}: keys 'target'/'access': target 'stream' goes with access 'herbie_byte_range' only"
        )
    nbytes = entry["approx_bytes"]
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        raise ManifestError(f"source {sid!r}: key 'approx_bytes' must be a non-negative int, got {nbytes!r}")
    sha = _parse_sha256(entry["sha256"], target, sid)
    source = Source(
        id=sid,
        desc=str(entry["desc"]),
        url=str(entry["url"]),
        access=access,
        approx_bytes=nbytes,
        license=str(entry["license"]),
        target=target,
        sha256=sha,
    )
    if sha is not None and not source.is_single_file:
        raise ManifestError(
            f"source {sid!r}: key 'sha256' must be TBD for a multi-file source (per-file hashes live in the fetch state)"
        )
    return source


def parse_manifest(data: Mapping) -> Manifest:
    """DATA-10. Validate a loaded YAML mapping (rules in the spec); raises ManifestError."""
    if not isinstance(data, Mapping):
        raise ManifestError("manifest: top level must be a mapping")
    version = data.get("version")
    if isinstance(version, bool) or version != 1:
        raise ManifestError(f"manifest: key 'version' must be 1, got {version!r}")
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ManifestError("manifest: key 'sources' must be a non-empty list")
    sources: list[Source] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw_sources):
        src = _parse_source(entry, i)
        if src.id in seen:
            raise ManifestError(f"source {src.id!r}: duplicate key 'id'")
        seen.add(src.id)
        sources.append(src)
    return Manifest(version=1, sources=tuple(sources))


def load_manifest(path: Path | str) -> Manifest:
    """DATA-10. Read YAML at ``path`` and ``parse_manifest`` it."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_manifest(data)
