"""Resumable, checksummed downloads from the manifest (specs/01-data.md DATA-11, DATA-12, DATA-13).

Network access goes through an injected ``Transport`` so the resume/verify logic is testable offline. Default
transports (requests for HTTP, unsigned boto3 for S3) are thin and exercised only in T1.
"""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlsplit

from outrun.data.manifest import Source


class DownloadError(OSError):
    """Bad HTTP status, size mismatch or an existing file of the wrong size (DATA-11)."""


class TransientStatusError(DownloadError):
    """An HTTP status that means "busy, try again later" (429, 500, 502, 503, 504). Still a DownloadError, but
    ``download(max_retries > 0)`` retries it, because servers such as NCEI often answer 503 under load."""


TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


class ChecksumError(OSError):
    """sha256 mismatch (DATA-11, DATA-12)."""


class KaggleCredentialsError(RuntimeError):
    """No Kaggle API credentials found (needed only for ``access: kaggle`` sources)."""


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


def transient_errors() -> tuple[type[BaseException], ...]:
    """Network errors worth retrying: the connection dropped or timed out, so the server may answer next time.

    Deliberately NOT ``OSError``: ``DownloadError`` (404 and other bad statuses, size mismatch) and
    ``ChecksumError`` are OSErrors too, and retrying those cannot help. The one DownloadError that is retried is
    ``TransientStatusError`` (429/5xx: the server is busy). requests/urllib3/botocore are imported lazily, like
    the default transports.
    """
    import http.client

    errors: list[type[BaseException]] = [
        ConnectionError,
        TimeoutError,
        http.client.IncompleteRead,
        TransientStatusError,
    ]
    try:
        import requests

        errors += [
            requests.exceptions.ConnectionError,  # not a subclass of the builtin ConnectionError
            requests.exceptions.ChunkedEncodingError,  # what urllib3's IncompleteRead turns into mid-body
            requests.exceptions.Timeout,  # ReadTimeout and ConnectTimeout
        ]
    except ImportError:
        pass
    try:
        import urllib3.exceptions

        errors += [urllib3.exceptions.ProtocolError, urllib3.exceptions.ReadTimeoutError]
    except ImportError:
        pass
    try:
        import botocore.exceptions as be

        # S3 downloads: botocore wraps urllib3's errors in its own classes while the body is being read.
        errors += [
            be.ResponseStreamingError,
            be.ReadTimeoutError,
            be.ConnectTimeoutError,
            be.EndpointConnectionError,
            be.ConnectionClosedError,
            be.IncompleteReadError,
        ]
    except ImportError:
        pass
    return tuple(errors)


def download(
    url: str,
    dest: Path | str,
    transport: Transport,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    *,
    max_retries: int = 0,
    backoff: float = 2.0,
    max_backoff: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadResult:
    """DATA-11. Resume from ``dest.name + '.part'``, verify, atomically rename. See spec steps 1-5.

    Retry (off by default, so a bare call behaves exactly as DATA-11 says): with ``max_retries > 0`` a transient
    network error or a busy-server status 429/5xx (see ``transient_errors``) is caught, we wait, and the single attempt runs again. It re-reads the
    ``.part`` size, so it resumes with a Range request. The bound counts failures *in a row without progress*: an
    attempt that added bytes to the ``.part`` file resets the count, so a long file that drops every few minutes
    still finishes, while a dead server gives up after ``max_retries`` retries. The wait doubles each time
    (``backoff``, ``2*backoff``, ...) up to ``max_backoff`` seconds. Other errors (404 and other bad statuses, size
    or checksum mismatch) are raised at once. ``sleep`` is injectable so tests do not really wait.
    """
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    retryable = transient_errors() if max_retries > 0 else ()
    best = part.stat().st_size if part.exists() else 0  # most bytes the .part file has ever held
    failures = 0
    while True:
        try:
            return _download_once(url, dest, transport, expected_sha256, expected_size)
        except retryable:
            after = part.stat().st_size if part.exists() else 0
            if after > best:  # new bytes beyond anything seen before: real progress, restart the count
                best, failures = after, 1
            else:
                failures += 1
            if failures > max_retries:
                raise
            sleep(min(backoff * 2 ** (failures - 1), max_backoff))


def _download_once(
    url: str,
    dest: Path,
    transport: Transport,
    expected_sha256: str | None,
    expected_size: int | None,
) -> DownloadResult:
    """One attempt of DATA-11 steps 1-5 (one transport request at most)."""
    expected = expected_sha256.lower() if expected_sha256 else None

    # 1. A finished file only ever comes from step 5, so it is trusted (or checked against the hash).
    if dest.exists():
        if expected is not None:
            actual = sha256_file(dest)
            if actual != expected:
                raise ChecksumError(f"{dest}: sha256 {actual} != expected {expected} (file left untouched)")
            return DownloadResult(dest, actual, 0, True)
        size = dest.stat().st_size
        if expected_size is not None and size != expected_size:
            raise DownloadError(f"{dest}: existing file has {size} bytes, expected {expected_size}")
        return DownloadResult(dest, sha256_file(dest), 0, True)

    # 2. Partial data lives next to dest.
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if expected_size is not None and offset > expected_size:
        part.write_bytes(b"")
        offset = 0

    received = 0
    if expected_size is None or offset != expected_size:
        # 3. One request, from the current offset.
        resp = transport.get(url, start=offset)
        if resp.status == 206:
            mode = "ab"
        elif resp.status == 200:
            mode = "wb"  # the server ignored the range: restart from byte 0
        elif resp.status in TRANSIENT_STATUSES:
            raise TransientStatusError(f"{url}: HTTP status {resp.status} (server busy, worth retrying)")
        else:
            raise DownloadError(f"{url}: unexpected HTTP status {resp.status}")
        with open(part, mode) as f:
            for chunk in resp.chunks:  # an exception here keeps what was received; next call resumes
                f.write(chunk)
                received += len(chunk)

    # 4. Size check.
    final_size = part.stat().st_size if part.exists() else 0
    if expected_size is not None and final_size != expected_size:
        raise DownloadError(
            f"{url}: got {final_size} bytes, expected {expected_size} ({part} kept for resume)"
        )

    # 5. Hash, then atomic rename.
    actual = sha256_file(part)
    if expected is not None and actual != expected:
        corrupt = dest.with_name(dest.name + ".corrupt")
        os.replace(part, corrupt)
        raise ChecksumError(f"{url}: sha256 {actual} != expected {expected}; moved to {corrupt}")
    os.replace(part, dest)
    return DownloadResult(dest, actual, received, False)


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """DATA-12. Lowercase hex sha256 of the file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class FetchState:
    """DATA-12. Persistent JSON ``{source_id: {relpath: {"sha256": str, "bytes": int}}}``, always saved atomically
    (temp file + rename). By default every ``record`` saves at once. With ``save_every_s`` > 0, records made since
    the last save are only in memory until the next timed save or ``flush()``: call ``flush()`` in a ``finally``,
    or a crash loses them and those files show as "unrecorded". Missing file = empty state; invalid JSON ->
    ValueError."""

    def __init__(self, path: Path | str, *, save_every_s: float = 0.0) -> None:
        """``save_every_s`` > 0 batches writes: ``record`` saves at most that often and ``flush`` saves the rest.
        Use it for sources with tens of thousands of files, where rewriting the whole JSON per file is slow."""
        self.path = Path(path)
        self.save_every_s = float(save_every_s)
        self._last_save = 0.0
        self._dirty = False
        self._data: dict[str, dict[str, dict]] = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"{self.path}: invalid fetch-state JSON ({e})") from e
            if not isinstance(data, dict):
                raise ValueError(f"{self.path}: fetch state must be a JSON object")
            self._data = data

    def record(self, source_id: str, relpath: str, sha256: str, nbytes: int) -> None:
        self._data.setdefault(source_id, {})[relpath] = {"sha256": sha256, "bytes": int(nbytes)}
        self._dirty = True
        if self.save_every_s <= 0 or time.monotonic() - self._last_save >= self.save_every_s:
            self.flush()

    def flush(self) -> None:
        """Write the state atomically if anything changed since the last save."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._dirty = False
        self._last_save = time.monotonic()

    def get(self, source_id: str, relpath: str) -> dict | None:
        rec = self._data.get(source_id, {}).get(relpath)
        return dict(rec) if rec is not None else None

    def records(self, source_id: str) -> dict[str, dict]:
        """All records of one source (copy)."""
        return {k: dict(v) for k, v in self._data.get(source_id, {}).items()}


_IGNORED_SUFFIXES = (".part", ".corrupt", ".kaggle-partial")


def _local_files(folder: Path) -> list[str]:
    """Relpaths (``/`` separators) of regular files under ``folder``, ignoring partial/corrupt files."""
    if not folder.is_dir():
        return []
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and not p.is_symlink() and not p.name.endswith(_IGNORED_SUFFIXES):
            out.append(p.relative_to(folder).as_posix())
    return sorted(out)


def _url_basename(url: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
    if not name:
        raise ValueError(f"cannot derive a file name from URL {url!r}")
    return name


def verify_source(source: Source, raw_root: Path | str, state: FetchState) -> VerifyReport:
    """DATA-12. Re-hash recorded files under ``raw_root/source.id``; classify ok/mismatched/missing/unrecorded."""
    folder = Path(raw_root) / source.id
    reference = {rel: rec["sha256"] for rel, rec in state.records(source.id).items()}
    if source.is_single_file and source.sha256 is not None:
        reference[_url_basename(source.url)] = source.sha256  # the manifest hash wins
    report = VerifyReport()
    for rel in sorted(reference):
        path = folder / rel
        if not path.is_file():
            report.missing.append(rel)
        elif sha256_file(path) == reference[rel].lower():
            report.ok.append(rel)
        else:
            report.mismatched.append(rel)
    report.unrecorded.extend(rel for rel in _local_files(folder) if rel not in reference)
    return report


def fetch_source(
    source: Source,
    raw_root: Path | str,
    transport: Transport,
    state: FetchState,
    lister: Callable[[str], Iterable[tuple[str, int]]] | None = None,
    *,
    max_retries: int = 8,
    sleep: Callable[[float], None] = time.sleep,
) -> list[DownloadResult]:
    """DATA-12. Download a manifest source into ``raw_root/source.id`` and record hashes; stream sources -> ValueError.

    Multi-file sources: ``lister(prefix_url)`` yields ``(file_url, size)`` for every remote file under one expanded
    prefix (S3: ``s3://bucket/key``; see ``s3_lister`` / ``zenodo_lister`` / ``http_index_lister``). Files already
    present with the right size are never downloaded again (they are only hashed into the state if the state has no
    record of them). Every file download retries dropped connections (HTTP and S3) and busy-server statuses
    (``max_retries``, see ``download``).
    """
    if source.target == "stream":
        raise ValueError(f"source {source.id!r} is streamed (target 'stream'); it is never stored raw")
    folder = Path(raw_root) / source.id
    if source.access == "kaggle":
        return fetch_kaggle(source, raw_root, state)

    if source.is_single_file:
        name = _url_basename(source.url)
        size = _remote_size(transport, source.url)
        result = download(
            source.url,
            folder / name,
            transport,
            expected_sha256=source.sha256,
            expected_size=size,
            max_retries=max_retries,
            sleep=sleep,
        )
        _record(state, source.id, name, result)
        return [result]

    if lister is None:
        raise ValueError(f"source {source.id!r} has several files; a lister is required (see default_lister)")
    results: list[DownloadResult] = []
    for prefix in expand_braces(source.url):
        sub = _brace_relative(source.url, prefix)
        dest_dir = folder / sub if sub else folder
        remote = list(lister(prefix))
        planned = {p.key for p in plan_downloads(remote, prefix, dest_dir)}
        for key, size in remote:
            dest = dest_dir / key[len(prefix) :].lstrip("/")
            rel = dest.relative_to(folder).as_posix()
            if key not in planned:  # already complete on disk: no network
                if state.get(source.id, rel) is None:
                    state.record(source.id, rel, sha256_file(dest), dest.stat().st_size)
                continue
            result = download(key, dest, transport, expected_size=size, max_retries=max_retries, sleep=sleep)
            _record(state, source.id, rel, result)
            results.append(result)
    return results


def _record(state: FetchState, source_id: str, rel: str, result: DownloadResult) -> None:
    rec = state.get(source_id, rel)
    nbytes = result.path.stat().st_size
    if rec is None or rec["sha256"] != result.sha256 or rec["bytes"] != nbytes:
        state.record(source_id, rel, result.sha256, nbytes)


def _remote_size(transport: Transport, url: str) -> int | None:
    """Optional ``transport.size(url)`` (default transports provide it); fakes without it give None."""
    size_fn = getattr(transport, "size", None)
    return size_fn(url) if callable(size_fn) else None


def _brace_relative(template: str, expanded: str) -> str:
    """Part of ``expanded`` below the folder that holds the first brace group of ``template`` ('' if none)."""
    i = template.find("{")
    if i < 0:
        return ""
    base = template[: template.rfind("/", 0, i) + 1]
    return expanded[len(base) :].strip("/")


def expand_braces(url: str) -> list[str]:
    """DATA-13. Expand ``{a,b}`` groups (cartesian product, left slowest); unbalanced/nested -> ValueError."""
    parts: list[list[str]] = []
    literal = []
    i = 0
    while i < len(url):
        c = url[i]
        if c == "}":
            raise ValueError(f"unbalanced '}}' in {url!r}")
        if c == "{":
            j = url.find("}", i + 1)
            if j < 0:
                raise ValueError(f"unbalanced '{{' in {url!r}")
            inner = url[i + 1 : j]
            if "{" in inner:
                raise ValueError(f"nested braces in {url!r}")
            parts.append(["".join(literal)])
            literal = []
            parts.append(inner.split(","))
            i = j + 1
            continue
        literal.append(c)
        i += 1
    parts.append(["".join(literal)])
    return ["".join(combo) for combo in itertools.product(*parts)]


def plan_downloads(remote: Iterable[tuple[str, int]], prefix: str, dest_dir: Path | str) -> list[PlannedFile]:
    """DATA-13. Map keys under ``prefix`` to ``dest_dir``; skip files already present with the same size."""
    dest_dir = Path(dest_dir)
    plan = []
    for key, size in remote:
        if not key.startswith(prefix):
            raise ValueError(f"remote key {key!r} is not under prefix {prefix!r}")
        dest = dest_dir / key[len(prefix) :].lstrip("/")
        if dest.is_file() and dest.stat().st_size == size:
            continue
        plan.append(PlannedFile(key, int(size), dest))
    return plan


def planned_bytes(plan: Sequence[PlannedFile]) -> int:
    """DATA-13. Sum of planned sizes."""
    return sum(p.size for p in plan)


# ---------------------------------------------------------------------------------------------------------------
# Default transports and listers (real network; not unit-tested, exercised in T1).
_CHUNK = 1 << 20
_TIMEOUT = 60


def _split_s3(url: str) -> tuple[str, str]:
    if not url.startswith("s3://"):
        raise ValueError(f"not an s3:// URL: {url!r}")
    bucket, _, key = url[5:].partition("/")
    return bucket, key


def _s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 10}))


class S3Transport:
    """Unsigned (public-bucket) S3 GETs with an optional byte range."""

    def __init__(self) -> None:
        self._client = _s3_client()

    def size(self, url: str) -> int | None:
        bucket, key = _split_s3(url)
        return int(self._client.head_object(Bucket=bucket, Key=key)["ContentLength"])

    def get(self, url: str, start: int = 0) -> RangeResponse:
        from botocore.exceptions import ClientError

        bucket, key = _split_s3(url)
        kwargs = {"Range": f"bytes={start}-"} if start > 0 else {}
        try:
            obj = self._client.get_object(Bucket=bucket, Key=key, **kwargs)
        except ClientError as e:
            status = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
            return RangeResponse(status, iter(()))
        status = int(obj["ResponseMetadata"]["HTTPStatusCode"])
        return RangeResponse(status, obj["Body"].iter_chunks(_CHUNK))


class HttpTransport:
    """HTTP(S) GETs with ``Range: bytes=<start>-`` via requests."""

    def __init__(self) -> None:
        import requests

        self._session = requests.Session()
        self._session.headers["User-Agent"] = "outrun-fetch/0.1"

    def size(self, url: str) -> int | None:
        r = self._session.head(url, allow_redirects=True, timeout=_TIMEOUT)
        length = r.headers.get("Content-Length")
        return (
            int(length) if r.ok and length and "gzip" not in r.headers.get("Content-Encoding", "") else None
        )

    def get(self, url: str, start: int = 0) -> RangeResponse:
        headers = {"Accept-Encoding": "identity"}  # byte offsets must refer to the stored bytes
        if start > 0:
            headers["Range"] = f"bytes={start}-"
        r = self._session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=_TIMEOUT)
        if r.status_code not in (200, 206):
            r.close()
            return RangeResponse(r.status_code, iter(()))
        return RangeResponse(r.status_code, r.iter_content(_CHUNK))


def s3_lister(prefix_url: str) -> Iterator[tuple[str, int]]:
    """Every object under an ``s3://bucket/prefix/`` as ``(s3 url, size)`` (unsigned)."""
    bucket, prefix = _split_s3(prefix_url)
    paginator = _s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                yield f"s3://{bucket}/{obj['Key']}", int(obj["Size"])


_ZENODO_RE = re.compile(r"^https://zenodo\.org/records/(\d+)/(?:files/)?$")


def zenodo_lister(prefix_url: str) -> Iterator[tuple[str, int]]:
    """Files of a Zenodo record given as ``https://zenodo.org/records/<id>/files/`` -> ``(file url, size)``."""
    import requests

    m = _ZENODO_RE.match(prefix_url)
    if not m:
        raise ValueError(f"expected https://zenodo.org/records/<id>/files/, got {prefix_url!r}")
    r = requests.get(f"https://zenodo.org/api/records/{m.group(1)}", timeout=_TIMEOUT)
    r.raise_for_status()
    base = f"https://zenodo.org/records/{m.group(1)}/files/"
    for f in r.json().get("files", []):
        yield base + f["key"], int(f["size"])


class _LinkParser(HTMLParser):
    """Collects the ``href`` of every ``<a>`` tag of an HTML page, in page order."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.hrefs.append(value)


def parse_index_links(html: str, page_url: str) -> list[str]:
    """Absolute URLs of all ``<a href>`` links in ``html``, resolved against ``page_url``; duplicates dropped."""
    parser = _LinkParser()
    parser.feed(html)
    parser.close()
    out: list[str] = []
    for href in parser.hrefs:
        url = urljoin(page_url, href).split("#", 1)[0]
        if url not in out:
            out.append(url)
    return out


def _http_get_text(url: str) -> str:
    import requests

    r = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "outrun-fetch/0.1"})
    r.raise_for_status()
    return r.text


def _http_head_size(url: str) -> int | None:
    import requests

    r = requests.head(url, allow_redirects=True, timeout=_TIMEOUT, headers={"User-Agent": "outrun-fetch/0.1"})
    length = r.headers.get("Content-Length")
    return int(length) if r.ok and length else None


def http_index_lister(
    prefix_url: str,
    patterns: Sequence[str] = ("*",),
    *,
    index_page: str = "",
    get_text: Callable[[str], str] = _http_get_text,
    head_size: Callable[[str], int | None] = _http_head_size,
) -> Iterator[tuple[str, int]]:
    """Files linked from a plain HTML index page (Apache ``Index of /...``) as ``(file url, size)``.

    1. Fetch ``prefix_url + index_page`` (``index_page`` is '' for a real directory index, or a page such as
       ``download_files.html`` when the server forbids directory listings) and collect its ``<a href>`` links.
    2. Keep only files directly under ``prefix_url`` (so no sort links ``?C=N;O=D``, parent or sub-directories)
       whose file name matches one of the shell-style ``patterns`` (case-sensitive ``fnmatchcase``).
    3. Ask the server for each kept file's exact size with a HEAD request: the "29M" on the page is rounded, and
       ``fetch_source`` needs exact sizes to skip finished files and to check downloads.
    Nothing matching, or a file without a Content-Length, raises ``DownloadError`` (never a silent empty fetch).
    """
    if not prefix_url.endswith("/"):
        raise ValueError(f"an index prefix must end with '/', got {prefix_url!r}")
    page_url = prefix_url + index_page
    found = 0
    for url in parse_index_links(get_text(page_url), page_url):
        if not url.startswith(prefix_url) or urlsplit(url).query:
            continue
        name = url[len(prefix_url) :]
        if not name or "/" in name:  # the directory itself, or something below a sub-directory
            continue
        if not any(fnmatch.fnmatchcase(name, p) for p in patterns):
            continue
        size = head_size(url)
        if size is None:
            raise DownloadError(f"{url}: server gave no Content-Length, cannot plan the download")
        found += 1
        yield url, int(size)
    if found == 0:
        raise DownloadError(f"{page_url}: no linked file matches {list(patterns)}")


# File-name globs for http directory sources, keyed by source id. The manifest has no field for this (see NOTES),
# and an unfiltered directory would pull far too much (Storm Events also has fatalities/locations files back to
# 1950). A directory source that is not listed here is refused by default_lister.
HTTP_INDEX_RULES: dict[str, dict] = {
    "storm_events": {
        "patterns": (
            "StormEvents_details-ftp_v1.0_d2019_c*.csv.gz",
            "StormEvents_details-ftp_v1.0_d202[0-6]_c*.csv.gz",
        ),
    },
    "aqs_pm25_hourly": {
        # aqs.epa.gov/aqsweb/airdata/ answers 403 to a directory listing (checked Nov 18); its file links live on
        # download_files.html, relative to the same folder.
        "index_page": "download_files.html",
        "patterns": ("hourly_88101_202[2-5].zip",),
    },
    "hms_smoke_polygons": {
        # One daily shapefile zip per month folder (Shapefile/YYYY/MM/hms_smokeYYYYMMDD.zip).
        "patterns": ("hms_smoke*.zip",),
    },
    "hms_fire_points": {
        # T20 replay: daily HMS fire-detection point shapefile zips of March 2025 (Shapefile/2025/03/hms_fireYYYYMMDD.zip).
        "patterns": ("hms_fire202503*.zip",),
    },
}


def default_transport(source: Source) -> Transport:
    """S3Transport for ``s3://`` URLs, HttpTransport otherwise."""
    return S3Transport() if source.url.startswith("s3://") else HttpTransport()


def default_lister(source: Source) -> Callable[[str], Iterable[tuple[str, int]]] | None:
    """s3_lister / zenodo_lister / http_index_lister by URL, None for single-file sources.

    An http(s) URL ending in '/' uses ``http_index_lister`` with the source's rule in ``HTTP_INDEX_RULES``; a
    directory source without a rule raises ValueError rather than download the whole directory.
    """
    if source.is_single_file:
        return None
    if source.url.startswith("s3://"):
        return s3_lister
    if source.url.startswith("https://zenodo.org/records/"):
        return zenodo_lister
    if (
        source.url.startswith(("http://", "https://"))
        and source.url.endswith("/")
        and source.id in HTTP_INDEX_RULES
    ):
        rule = HTTP_INDEX_RULES[source.id]
        return partial(http_index_lister, patterns=rule["patterns"], index_page=rule.get("index_page", ""))
    raise ValueError(f"source {source.id!r}: no lister for {source.url!r}")


# ---------------------------------------------------------------------------------------------------------------
# Kaggle (NDWS). Token: https://www.kaggle.com/settings -> API -> "Create New Token" saves kaggle.json.
def check_kaggle_credentials(env: Mapping[str, str] | None = None, home: Path | None = None) -> str:
    """Return where Kaggle credentials were found; raise KaggleCredentialsError with instructions if none.

    Accepted: ``$KAGGLE_CONFIG_DIR/kaggle.json`` or ``~/.kaggle/kaggle.json``, ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``,
    ``KAGGLE_API_TOKEN``, or ``~/.kaggle/access_token`` (newer Kaggle tokens).
    """
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    config_dir = (
        Path(env["KAGGLE_CONFIG_DIR"]).expanduser() if env.get("KAGGLE_CONFIG_DIR") else home / ".kaggle"
    )
    kaggle_json = config_dir / "kaggle.json"
    if kaggle_json.is_file():
        try:
            data = json.loads(kaggle_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise KaggleCredentialsError(f"{kaggle_json} is not valid JSON ({e}); download it again") from e
        if not (isinstance(data, dict) and data.get("username") and data.get("key")):
            raise KaggleCredentialsError(f'{kaggle_json} must contain {{"username": ..., "key": ...}}')
        return str(kaggle_json)
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return "KAGGLE_USERNAME/KAGGLE_KEY"
    if env.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN"
    if (config_dir / "access_token").is_file():
        return str(config_dir / "access_token")
    raise KaggleCredentialsError(
        f"Kaggle credentials not found: {kaggle_json} is missing.\n"
        "  1. Sign in at https://www.kaggle.com, open Settings (https://www.kaggle.com/settings), section 'API'.\n"
        "  2. Click 'Create New Token' (legacy API key): the browser downloads kaggle.json.\n"
        f"  3. mkdir -p {config_dir} && mv ~/Downloads/kaggle.json {kaggle_json} && chmod 600 {kaggle_json}\n"
        "  (A newer-style API token also works: export KAGGLE_API_TOKEN=<token> or save it to "
        f"{config_dir / 'access_token'}.)\n"
        "  Also accept the dataset's terms once on its Kaggle page if asked."
    )


def fetch_kaggle(source: Source, raw_root: Path | str, state: FetchState) -> list[DownloadResult]:
    """Download a ``kaggle://owner/dataset`` source as one zip into ``raw_root/source.id`` and record its hash.

    A zip already recorded in the state with the same size is not requested again; otherwise the Kaggle client
    resumes a partial download itself.
    """
    if not source.url.startswith("kaggle://"):
        raise ValueError(f"source {source.id!r}: kaggle URL must look like kaggle://owner/dataset")
    dataset = source.url[len("kaggle://") :].strip("/")
    folder = Path(raw_root) / source.id
    zip_name = dataset.rsplit("/", 1)[-1] + ".zip"
    zip_path = folder / zip_name
    rec = state.get(source.id, zip_name)
    if rec is not None and zip_path.is_file() and zip_path.stat().st_size == rec["bytes"]:
        return [DownloadResult(zip_path, rec["sha256"], 0, True)]

    check_kaggle_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    folder.mkdir(parents=True, exist_ok=True)
    before = zip_path.stat().st_size if zip_path.is_file() else 0
    api.dataset_download_files(dataset, path=str(folder), force=False, quiet=False, unzip=False)
    if not zip_path.is_file():
        raise DownloadError(f"Kaggle download of {dataset!r} did not produce {zip_path}")
    sha = sha256_file(zip_path)
    if source.sha256 is not None and sha != source.sha256:
        raise ChecksumError(f"{zip_path}: sha256 {sha} != manifest {source.sha256}")
    nbytes = zip_path.stat().st_size
    state.record(source.id, zip_name, sha, nbytes)
    return [DownloadResult(zip_path, sha, max(nbytes - before, 0), False)]
