"""Retry of dropped downloads and the HTTP directory-index lister (data/fetch.py).

Offline only: fake transports, fake HTML pages, and a fake ``sleep`` that records the waits instead of waiting.
"""

from __future__ import annotations

import hashlib

import pytest
import requests

from outrun.data.fetch import (
    ChecksumError,
    DownloadError,
    FetchState,
    RangeResponse,
    TransientStatusError,
    default_lister,
    download,
    fetch_source,
    http_index_lister,
    parse_index_links,
)
from outrun.data.manifest import Source

CONTENT = bytes(range(256)) * 40  # 10240 bytes, 11 chunks of 1000
SHA = hashlib.sha256(CONTENT).hexdigest()


class FlakyTransport:
    """Serves CONTENT with Range support. ``failures`` is a list, one entry per call: None = clean response,
    ``(k, exc)`` = raise ``exc`` after k chunks of 1000 bytes. Calls beyond the list are clean."""

    def __init__(self, failures=(), content=CONTENT, status=None):
        self.failures = list(failures)
        self.content = content
        self.status = status
        self.calls: list[tuple[str, int]] = []

    def get(self, url, start=0):
        self.calls.append((url, start))
        if self.status is not None:
            return RangeResponse(self.status, iter(()))
        plan = self.failures.pop(0) if self.failures else None
        body = self.content[start:]

        def chunks():
            for n, i in enumerate(range(0, len(body), 1000)):
                if plan is not None and n == plan[0]:
                    raise plan[1]
                yield body[i : i + 1000]

        return RangeResponse(206 if start > 0 else 200, chunks())


class SleepRecorder:
    def __init__(self):
        self.waits: list[float] = []

    def __call__(self, seconds):
        self.waits.append(seconds)


# ---------------------------------------------------------------- retry
def test_default_download_does_not_retry(tmp_path):
    t = FlakyTransport([(3, requests.exceptions.ChunkedEncodingError("IncompleteRead"))])
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        download("u", tmp_path / "f.bin", t, expected_size=len(CONTENT))
    assert len(t.calls) == 1
    assert (tmp_path / "f.bin.part").stat().st_size == 3000


def test_dropped_connections_are_retried_and_resumed(tmp_path):
    dest = tmp_path / "f.bin"
    t = FlakyTransport(
        [
            (3, requests.exceptions.ChunkedEncodingError("IncompleteRead")),
            (2, requests.exceptions.ReadTimeout("read timed out")),
            (1, requests.exceptions.ConnectionError("reset")),
        ]
    )
    sleep = SleepRecorder()
    r = download("u", dest, t, expected_sha256=SHA, expected_size=len(CONTENT), max_retries=3, sleep=sleep)
    assert dest.read_bytes() == CONTENT and not r.skipped
    assert t.calls == [("u", 0), ("u", 3000), ("u", 5000), ("u", 6000)]
    assert r.bytes_downloaded == len(CONTENT) - 6000  # only the last attempt's bytes
    assert sleep.waits == [2.0, 2.0, 2.0]  # every attempt made progress, so the backoff restarts each time


def test_backoff_doubles_up_to_the_cap_when_no_progress(tmp_path):
    t = FlakyTransport([(0, requests.exceptions.ConnectionError("refused"))] * 5)
    sleep = SleepRecorder()
    r = download(
        "u",
        tmp_path / "f.bin",
        t,
        expected_size=len(CONTENT),
        max_retries=5,
        backoff=1.0,
        max_backoff=5.0,
        sleep=sleep,
    )
    assert r.path.read_bytes() == CONTENT
    assert sleep.waits == [1.0, 2.0, 4.0, 5.0, 5.0]
    assert len(t.calls) == 6


def test_gives_up_after_max_retries_without_progress_and_keeps_part(tmp_path):
    dest = tmp_path / "f.bin"
    part = tmp_path / "f.bin.part"
    part.write_bytes(CONTENT[:4000])
    t = FlakyTransport([(0, requests.exceptions.ReadTimeout("timed out"))] * 10)
    sleep = SleepRecorder()
    with pytest.raises(requests.exceptions.ReadTimeout):
        download("u", dest, t, expected_size=len(CONTENT), max_retries=2, sleep=sleep)
    assert len(t.calls) == 3 and len(sleep.waits) == 2
    assert part.read_bytes() == CONTENT[:4000] and not dest.exists()


def test_builtin_connection_error_is_transient(tmp_path):
    t = FlakyTransport([(1, ConnectionError("dropped"))])
    r = download("u", tmp_path / "f.bin", t, expected_sha256=SHA, max_retries=1, sleep=SleepRecorder())
    assert r.sha256 == SHA and t.calls == [("u", 0), ("u", 1000)]


def test_bad_status_is_not_retried(tmp_path):
    t = FlakyTransport(status=404)
    sleep = SleepRecorder()
    with pytest.raises(DownloadError):
        download("u", tmp_path / "f.bin", t, max_retries=8, sleep=sleep)
    assert len(t.calls) == 1 and sleep.waits == []


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_busy_server_status_is_retried(tmp_path, status):
    t = FlakyTransport(status=status)
    sleep = SleepRecorder()
    with pytest.raises(TransientStatusError) as info:
        download("u", tmp_path / "f.bin", t, max_retries=2, sleep=sleep)
    assert isinstance(info.value, DownloadError)  # still a DownloadError for DATA-11 callers
    assert len(t.calls) == 3 and len(sleep.waits) == 2


def test_busy_server_status_is_not_retried_by_default(tmp_path):
    t = FlakyTransport(status=503)
    with pytest.raises(TransientStatusError):
        download("u", tmp_path / "f.bin", t)
    assert len(t.calls) == 1


def test_busy_server_then_success_finishes(tmp_path):
    class BusyOnce(FlakyTransport):
        def get(self, url, start=0):
            if not self.calls:
                self.calls.append((url, start))
                return RangeResponse(503, iter(()))
            return super().get(url, start)

    t = BusyOnce()
    r = download("u", tmp_path / "f.bin", t, expected_sha256=SHA, max_retries=1, sleep=SleepRecorder())
    assert r.sha256 == SHA and len(t.calls) == 2


def test_dropped_s3_stream_is_retried_and_resumed(tmp_path):
    be = pytest.importorskip("botocore.exceptions")
    t = FlakyTransport(
        [
            (2, be.ResponseStreamingError(error="connection reset")),
            (1, be.ReadTimeoutError(endpoint_url="https://example.org")),
        ]
    )
    sleep = SleepRecorder()
    r = download("u", tmp_path / "f.bin", t, expected_sha256=SHA, max_retries=3, sleep=sleep)
    assert r.sha256 == SHA
    assert t.calls == [("u", 0), ("u", 2000), ("u", 3000)]
    assert len(sleep.waits) == 2


def test_size_mismatch_is_not_retried(tmp_path):
    t = FlakyTransport()
    sleep = SleepRecorder()
    with pytest.raises(DownloadError):
        download("u", tmp_path / "f.bin", t, expected_size=len(CONTENT) + 1, max_retries=8, sleep=sleep)
    assert len(t.calls) == 1 and sleep.waits == []


def test_checksum_mismatch_is_not_retried(tmp_path):
    t = FlakyTransport()
    sleep = SleepRecorder()
    with pytest.raises(ChecksumError):
        download("u", tmp_path / "f.bin", t, expected_sha256="0" * 64, max_retries=8, sleep=sleep)
    assert len(t.calls) == 1 and sleep.waits == []
    assert (tmp_path / "f.bin.corrupt").read_bytes() == CONTENT


def test_fetch_source_retries_by_default(tmp_path):
    src = Source("one", "d", "https://example.org/data/f.bin", "http", 1, "PD", "hdd", SHA)
    t = FlakyTransport([(4, requests.exceptions.ChunkedEncodingError("IncompleteRead"))])
    sleep = SleepRecorder()
    results = fetch_source(src, tmp_path / "raw", t, FetchState(tmp_path / "s.json"), sleep=sleep)
    assert results[0].sha256 == SHA and t.calls == [(src.url, 0), (src.url, 4000)]
    assert len(sleep.waits) == 1


# ---------------------------------------------------------------- HTTP index lister
PREFIX = "https://www.example.gov/pub/csvfiles/"
APACHE_INDEX = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html><head><title>Index of /pub/csvfiles</title></head><body><h1>Index of /pub/csvfiles</h1>
<table>
<tr><th><a href="?C=N;O=D">Name</a></th><th><a href="?C=M;O=A">Last modified</a></th></tr>
<tr><td><a href="/pub/">Parent Directory</a></td></tr>
<tr><td><a href="legacy/">legacy/</a></td></tr>
<tr><td><a href="StormEvents_details-ftp_v1.0_d2018_c20260323.csv.gz">StormEvents_details-ftp_v1.0_d2018_c..&gt;</a>
</td><td>29M</td></tr>
<tr><td><a href="StormEvents_details-ftp_v1.0_d2019_c20260323.csv.gz">StormEvents_details-ftp_v1.0_d2019_c..&gt;</a>
</td><td>12M</td></tr>
<tr><td><a href="StormEvents_fatalities-ftp_v1.0_d2019_c20260323.csv.gz">fatalities</a></td></tr>
<tr><td><a href="StormEvents_details-ftp_v1.0_d2026_c20261026.csv.gz">2026</a></td></tr>
<tr><td><a href="https://www.example.gov/pub/csvfiles/StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz">abs</a>
</td></tr>
<tr><td><a href="https://other.example.org/StormEvents_details-ftp_v1.0_d2020_c1.csv.gz">elsewhere</a></td></tr>
</table></body></html>
"""
STORM_PATTERNS = (
    "StormEvents_details-ftp_v1.0_d2019_c*.csv.gz",
    "StormEvents_details-ftp_v1.0_d202[0-6]_c*.csv.gz",
)


class FakeWeb:
    def __init__(self, pages, sizes=None):
        self.pages = pages
        self.sizes = sizes or {}
        self.fetched: list[str] = []
        self.headed: list[str] = []

    def get_text(self, url):
        self.fetched.append(url)
        return self.pages[url]

    def head_size(self, url):
        self.headed.append(url)
        return self.sizes.get(url, 100)


def test_parse_index_links_resolves_relative_urls():
    html = '<a href="a.zip">a</a> <a href="./b.zip#top">b</a> <a href="../up/">up</a> <a>no href</a> <a href="a.zip">'
    assert parse_index_links(html, PREFIX) == [
        PREFIX + "a.zip",
        PREFIX + "b.zip",
        "https://www.example.gov/pub/up/",
    ]


def test_http_index_lister_filters_by_pattern_and_prefix():
    web = FakeWeb(
        {PREFIX: APACHE_INDEX}, sizes={PREFIX + "StormEvents_details-ftp_v1.0_d2019_c20260323.csv.gz": 1234}
    )
    out = list(http_index_lister(PREFIX, STORM_PATTERNS, get_text=web.get_text, head_size=web.head_size))
    assert out == [
        (PREFIX + "StormEvents_details-ftp_v1.0_d2019_c20260323.csv.gz", 1234),
        (PREFIX + "StormEvents_details-ftp_v1.0_d2026_c20261026.csv.gz", 100),
        (PREFIX + "StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz", 100),
    ]
    assert web.fetched == [PREFIX]
    assert web.headed == [u for u, _ in out]  # only matching files are sized


def test_http_index_lister_reads_a_named_index_page():
    page = PREFIX + "download_files.html"
    html = '<a href="hourly_88101_2021.zip">x</a><a href="hourly_88101_2024.zip">x</a><a href="daily_88101_2024.zip">'
    web = FakeWeb({page: html})
    out = list(
        http_index_lister(
            PREFIX,
            ("hourly_88101_202[2-5].zip",),
            index_page="download_files.html",
            get_text=web.get_text,
            head_size=web.head_size,
        )
    )
    assert out == [(PREFIX + "hourly_88101_2024.zip", 100)]
    assert web.fetched == [page]


def test_http_index_lister_matching_nothing_raises():
    web = FakeWeb({PREFIX: APACHE_INDEX})
    with pytest.raises(DownloadError):
        list(http_index_lister(PREFIX, ("*.nc",), get_text=web.get_text, head_size=web.head_size))


def test_http_index_lister_needs_an_exact_size():
    web = FakeWeb({PREFIX: '<a href="a.zip">a</a>'})
    with pytest.raises(DownloadError):
        list(http_index_lister(PREFIX, ("*.zip",), get_text=web.get_text, head_size=lambda url: None))


def _http_source(sid, url):
    return Source(sid, "d", url, "http", 1, "PD", "hdd", None)


def test_default_lister_wires_http_directories_by_source_id():
    storm = default_lister(
        _http_source("storm_events", "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/")
    )
    aqs = default_lister(_http_source("aqs_pm25_hourly", "https://aqs.epa.gov/aqsweb/airdata/"))
    assert storm.func is http_index_lister and storm.keywords["patterns"] == STORM_PATTERNS
    assert aqs.func is http_index_lister and aqs.keywords["index_page"] == "download_files.html"


def test_default_lister_refuses_an_http_directory_without_a_rule():
    with pytest.raises(ValueError):
        default_lister(_http_source("hms_smoke", "https://example.gov/pub/FIRE/Shapefile/"))


@pytest.mark.req("DATA-12")
def test_fetch_source_downloads_every_listed_http_file(tmp_path):
    files = {PREFIX + "a_2019.csv.gz": b"alpha" * 300, PREFIX + "a_2020.csv.gz": b"beta" * 500}
    html = (
        "".join(f'<a href="{u.rsplit("/", 1)[1]}">x</a>' for u in files) + '<a href="b_2019.csv.gz">skip</a>'
    )

    class Transport:
        def get(self, url, start=0):
            return RangeResponse(206 if start else 200, iter([files[url][start:]]))

    def lister(prefix):
        return http_index_lister(
            prefix, ("a_*.csv.gz",), get_text=lambda u: html, head_size=lambda u: len(files[u])
        )

    src = _http_source("dir", PREFIX)
    state = FetchState(tmp_path / "s.json")
    results = fetch_source(src, tmp_path / "raw", Transport(), state, lister=lister)
    assert sorted(r.path.name for r in results) == ["a_2019.csv.gz", "a_2020.csv.gz"]
    for url, body in files.items():
        name = url.rsplit("/", 1)[1]
        assert (tmp_path / "raw" / "dir" / name).read_bytes() == body
        assert state.get("dir", name) == {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def test_server_ignoring_range_cannot_retry_forever(tmp_path):
    """A server that always restarts from byte 0 and drops at 3000, 5000, 3000, 5000, ... never gets past 5000."""

    class RestartingTransport:
        def __init__(self):
            self.calls = 0

        def get(self, url, start=0):
            self.calls += 1
            cut = 3 if self.calls % 2 else 5

            def chunks():
                for n, i in enumerate(range(0, len(CONTENT), 1000)):
                    if n == cut:
                        raise requests.exceptions.ChunkedEncodingError("IncompleteRead")
                    yield CONTENT[i : i + 1000]

            return RangeResponse(200, chunks())

    t = RestartingTransport()
    sleep = SleepRecorder()
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        download("u", tmp_path / "f.bin", t, expected_size=len(CONTENT), max_retries=3, sleep=sleep)
    assert t.calls == 5  # 3000 and 5000 are progress; then 3 failures that never beat 5000, then give up
