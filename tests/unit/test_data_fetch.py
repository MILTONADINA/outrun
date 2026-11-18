"""specs/01-data.md DATA-11 (resumable download), DATA-12 (sha256 + fetch state), DATA-13 (planning).

All network I/O goes through a fake in-memory Transport.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from outrun.data.fetch import (
    ChecksumError,
    DownloadError,
    FetchState,
    RangeResponse,
    download,
    expand_braces,
    fetch_source,
    plan_downloads,
    planned_bytes,
    sha256_file,
    verify_source,
)
from outrun.data.manifest import Source

CONTENT = bytes(range(256)) * 40  # 10240 bytes
SHA = hashlib.sha256(CONTENT).hexdigest()


class FakeTransport:
    """Serves ``content`` in 1000-byte chunks. honour_range=False mimics a server that ignores Range.
    fail_after_chunks=k raises ConnectionError after k chunks (once)."""

    def __init__(self, content=CONTENT, honour_range=True, fail_after_chunks=None, status=None):
        self.content = content
        self.honour_range = honour_range
        self.fail_after_chunks = fail_after_chunks
        self.status = status
        self.calls: list[tuple[str, int]] = []

    def get(self, url, start=0):
        self.calls.append((url, start))
        if self.status is not None:
            return RangeResponse(self.status, iter([]))
        if start > 0 and self.honour_range:
            status, body = 206, self.content[start:]
        else:
            status, body = 200, self.content
        fail = self.fail_after_chunks
        self.fail_after_chunks = None

        def chunks():
            for i in range(0, len(body), 1000):
                if fail is not None and i // 1000 == fail:
                    raise ConnectionError("dropped")
                yield body[i : i + 1000]

        return RangeResponse(status, chunks())


# ---------------------------------------------------------------- DATA-11
@pytest.mark.req("DATA-11")
def test_fresh_download_verifies_and_renames(tmp_path):
    dest = tmp_path / "sub" / "f.bin"
    t = FakeTransport()
    r = download("u", dest, t, expected_sha256=SHA, expected_size=len(CONTENT))
    assert dest.read_bytes() == CONTENT
    assert not (tmp_path / "sub" / "f.bin.part").exists()
    assert r.sha256 == SHA and r.bytes_downloaded == len(CONTENT) and not r.skipped and r.path == dest
    assert t.calls == [("u", 0)]


@pytest.mark.req("DATA-11")
def test_existing_complete_file_is_skipped_without_network(tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(CONTENT)
    t = FakeTransport()
    assert download("u", dest, t, expected_sha256=SHA).skipped
    assert download("u", dest, t, expected_size=len(CONTENT)).skipped
    assert download("u", dest, t).skipped
    assert t.calls == []


@pytest.mark.req("DATA-11")
def test_existing_file_wrong_hash_or_size(tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"junk")
    with pytest.raises(ChecksumError):
        download("u", dest, FakeTransport(), expected_sha256=SHA)
    assert dest.read_bytes() == b"junk"  # left untouched
    with pytest.raises(DownloadError):
        download("u", dest, FakeTransport(), expected_size=len(CONTENT))


@pytest.mark.req("DATA-11")
def test_interrupted_download_resumes_from_offset(tmp_path):
    dest = tmp_path / "f.bin"
    part = tmp_path / "f.bin.part"
    t = FakeTransport(fail_after_chunks=3)
    with pytest.raises(ConnectionError):
        download("u", dest, t, expected_sha256=SHA, expected_size=len(CONTENT))
    assert part.read_bytes() == CONTENT[:3000]
    assert not dest.exists()
    r = download("u", dest, t, expected_sha256=SHA, expected_size=len(CONTENT))
    assert t.calls == [("u", 0), ("u", 3000)]
    assert dest.read_bytes() == CONTENT and r.bytes_downloaded == len(CONTENT) - 3000
    assert not part.exists()


@pytest.mark.req("DATA-11")
def test_server_ignoring_range_restarts(tmp_path):
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(CONTENT[:2500])
    t = FakeTransport(honour_range=False)
    r = download("u", dest, t, expected_sha256=SHA)
    assert t.calls == [("u", 2500)]
    assert dest.read_bytes() == CONTENT and r.bytes_downloaded == len(CONTENT)


@pytest.mark.req("DATA-11")
def test_complete_part_needs_no_request(tmp_path):
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(CONTENT)
    t = FakeTransport()
    r = download("u", dest, t, expected_sha256=SHA, expected_size=len(CONTENT))
    assert t.calls == [] and dest.read_bytes() == CONTENT and r.bytes_downloaded == 0


@pytest.mark.req("DATA-11")
def test_oversized_part_is_truncated(tmp_path):
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(b"z" * (len(CONTENT) + 5))
    t = FakeTransport()
    download("u", dest, t, expected_size=len(CONTENT))
    assert t.calls == [("u", 0)] and dest.read_bytes() == CONTENT


@pytest.mark.req("DATA-11")
def test_checksum_mismatch_moves_part_to_corrupt(tmp_path):
    dest = tmp_path / "f.bin"
    with pytest.raises(ChecksumError):
        download("u", dest, FakeTransport(), expected_sha256="0" * 64)
    assert not dest.exists() and not (tmp_path / "f.bin.part").exists()
    assert (tmp_path / "f.bin.corrupt").read_bytes() == CONTENT


@pytest.mark.req("DATA-11")
def test_bad_status_and_short_body(tmp_path):
    with pytest.raises(DownloadError):
        download("u", tmp_path / "a.bin", FakeTransport(status=404))
    with pytest.raises(DownloadError):
        download("u", tmp_path / "b.bin", FakeTransport(content=CONTENT[:100]), expected_size=len(CONTENT))
    assert (tmp_path / "b.bin.part").exists() and not (tmp_path / "b.bin").exists()
    for exc in (DownloadError, ChecksumError):
        assert issubclass(exc, OSError) and not issubclass(exc, RuntimeError)


# ---------------------------------------------------------------- DATA-12
@pytest.mark.req("DATA-12")
def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(CONTENT)
    assert sha256_file(p) == SHA
    assert sha256_file(p, chunk_size=7) == SHA
    (tmp_path / "empty").write_bytes(b"")
    assert sha256_file(tmp_path / "empty") == hashlib.sha256(b"").hexdigest()


@pytest.mark.req("DATA-12")
def test_fetch_state_persists(tmp_path):
    p = tmp_path / "state.json"
    s = FetchState(p)
    assert s.get("src", "a.h5") is None
    s.record("src", "a.h5", SHA, 10)
    assert FetchState(p).get("src", "a.h5") == {"sha256": SHA, "bytes": 10}
    assert json.loads(p.read_text())["src"]["a.h5"]["bytes"] == 10
    p.write_text("{not json")
    with pytest.raises(ValueError):
        FetchState(p)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.mark.req("DATA-12")
def test_verify_source_classifies_files(tmp_path):
    raw = tmp_path / "raw"
    src = Source("sevir_vil", "d", "s3://sevir/data/vil/{2018,2019}/", "s3_unsigned", 1, "x", "hdd", None)
    state = FetchState(tmp_path / "state.json")
    good, bad = b"good", b"bad!"
    _write(raw / "sevir_vil" / "2018" / "a.h5", good)
    _write(raw / "sevir_vil" / "2018" / "b.h5", bad)
    _write(raw / "sevir_vil" / "2019" / "new.h5", b"n")
    _write(raw / "sevir_vil" / "2019" / "x.h5.part", b"p")
    state.record("sevir_vil", "2018/a.h5", hashlib.sha256(good).hexdigest(), 4)
    state.record("sevir_vil", "2018/b.h5", hashlib.sha256(b"original").hexdigest(), 8)
    state.record("sevir_vil", "2019/gone.h5", SHA, 1)
    rep = verify_source(src, raw, state)
    assert rep.ok == ["2018/a.h5"]
    assert rep.mismatched == ["2018/b.h5"]
    assert rep.missing == ["2019/gone.h5"]
    assert rep.unrecorded == ["2019/new.h5"]


@pytest.mark.req("DATA-12")
def test_verify_single_file_against_manifest_hash(tmp_path):
    raw = tmp_path / "raw"
    src = Source(
        "cat", "d", "https://e.org/CATALOG.csv", "http", 1, "x", "hdd", hashlib.sha256(b"abc").hexdigest()
    )
    _write(raw / "cat" / "CATALOG.csv", b"abd")
    rep = verify_source(src, raw, FetchState(tmp_path / "s.json"))
    assert rep.mismatched == ["CATALOG.csv"] and rep.ok == []


@pytest.mark.req("DATA-12")
def test_fetch_source_single_file_records_hash(tmp_path):
    raw = tmp_path / "raw"
    src = Source("cat", "d", "https://e.org/path/CATALOG.csv", "http", len(CONTENT), "x", "hdd", None)
    state = FetchState(tmp_path / "s.json")
    t = FakeTransport()
    results = fetch_source(src, raw, t, state)
    assert (raw / "cat" / "CATALOG.csv").read_bytes() == CONTENT
    assert t.calls == [("https://e.org/path/CATALOG.csv", 0)]
    assert len(results) == 1 and results[0].sha256 == SHA
    assert FetchState(tmp_path / "s.json").get("cat", "CATALOG.csv") == {"sha256": SHA, "bytes": len(CONTENT)}


@pytest.mark.req("DATA-12")
def test_fetch_source_refuses_stream(tmp_path):
    src = Source("hrrr", "d", "s3://noaa-hrrr-bdp-pds/", "herbie_byte_range", 0, "x", "stream", None)
    t = FakeTransport()
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError):
        fetch_source(src, raw, t, FetchState(tmp_path / "s.json"))
    assert t.calls == [] and list(raw.iterdir()) == []


# ---------------------------------------------------------------- DATA-13
@pytest.mark.req("DATA-13")
def test_expand_braces():
    assert expand_braces("s3://sevir/data/vil/{2018,2019}/") == [
        "s3://sevir/data/vil/2018/",
        "s3://sevir/data/vil/2019/",
    ]
    assert expand_braces("a{1,2}b{x,y}") == ["a1bx", "a1by", "a2bx", "a2by"]
    assert expand_braces("plain") == ["plain"]
    for bad in ("a{1,2", "a}1{", "a{1,{2,3}}"):
        with pytest.raises(ValueError):
            expand_braces(bad)


@pytest.mark.req("DATA-13")
def test_plan_downloads_skips_existing_same_size(tmp_path):
    prefix = "data/vil/2018/"
    _write(tmp_path / "a.h5", b"x" * 10)  # same size -> skip
    _write(tmp_path / "b.h5", b"x" * 3)  # different size -> plan again
    remote = [(prefix + "a.h5", 10), (prefix + "b.h5", 20), (prefix + "sub/c.h5", 30)]
    plan = plan_downloads(remote, prefix, tmp_path)
    assert [p.key for p in plan] == [prefix + "b.h5", prefix + "sub/c.h5"]
    assert [p.dest for p in plan] == [tmp_path / "b.h5", tmp_path / "sub" / "c.h5"]
    assert planned_bytes(plan) == 50
    with pytest.raises(ValueError):
        plan_downloads([("other/x.h5", 1)], prefix, tmp_path)
