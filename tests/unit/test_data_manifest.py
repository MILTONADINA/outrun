"""specs/01-data.md DATA-10: manifest parsing and validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from outrun.data.manifest import (
    ACCESS,
    TARGETS,
    Manifest,
    ManifestError,
    Source,
    load_manifest,
    parse_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
HEX = "ab" * 32


def _src(**kw) -> dict:
    s = {
        "id": "one_file",
        "desc": "d",
        "url": "https://example.org/a.csv",
        "access": "http",
        "approx_bytes": 10,
        "license": "CC BY 4.0",
        "target": "hdd",
        "sha256": "TBD",
    }
    s.update(kw)
    return s


def _doc(*sources) -> dict:
    return {"version": 1, "sources": list(sources) or [_src()]}


@pytest.mark.req("DATA-10")
def test_parse_valid_manifest():
    m = parse_manifest(
        _doc(
            _src(),
            _src(id="dir_src", url="s3://b/data/{2018,2019}/", access="s3_unsigned", sha256=None),
            _src(id="hrrr", url="s3://noaa-hrrr-bdp-pds/", access="herbie_byte_range", target="stream"),
            _src(
                id="hrrr2",
                url="s3://noaa-hrrr-bdp-pds/",
                access="herbie_byte_range",
                target="stream",
                sha256="n/a",
            ),
            _src(id="hashed", sha256=HEX.upper(), target="ssd"),
        )
    )
    assert isinstance(m, Manifest) and m.version == 1
    assert [s.id for s in m.sources] == ["one_file", "dir_src", "hrrr", "hrrr2", "hashed"]
    assert m.get("one_file").sha256 is None
    assert m.get("dir_src").sha256 is None
    assert m.get("hashed").sha256 == HEX  # lowercased
    assert m.get("one_file").is_single_file and not m.get("dir_src").is_single_file
    assert [s.id for s in m.by_target("stream")] == ["hrrr", "hrrr2"]
    assert m.get("hrrr2").sha256 is None
    with pytest.raises(KeyError):
        m.get("missing")


@pytest.mark.req("DATA-10")
def test_extra_keys_ignored():
    m = parse_manifest(_doc(_src(notes="whatever")))
    assert isinstance(m.sources[0], Source)


@pytest.mark.req("DATA-10")
@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(version=2),
        lambda d: d.pop("sources"),
        lambda d: d.update(sources=[]),
        lambda d: d["sources"][0].pop("license"),
        lambda d: d["sources"][0].update(id="Bad-Id"),
        lambda d: d["sources"].append(copy.deepcopy(d["sources"][0])),
        lambda d: d["sources"][0].update(access="ftp"),
        lambda d: d["sources"][0].update(target="cloud"),
        lambda d: d["sources"][0].update(approx_bytes=-1),
        lambda d: d["sources"][0].update(approx_bytes=True),
        lambda d: d["sources"][0].update(approx_bytes="10"),
        lambda d: d["sources"][0].update(sha256="abc123"),
        lambda d: d["sources"][0].update(sha256="n/a"),  # n/a only allowed for stream sources
        lambda d: d["sources"][0].update(target="stream"),
        lambda d: d["sources"][0].update(access="herbie_byte_range"),
        lambda d: d["sources"][0].update(url="s3://b/dir/", sha256=HEX),
    ],
)
def test_invalid_manifests_raise(mutate):
    d = _doc()
    mutate(d)
    with pytest.raises(ManifestError):
        parse_manifest(d)


@pytest.mark.req("DATA-10")
def test_error_names_source_id():
    with pytest.raises(ManifestError, match="one_file"):
        parse_manifest(_doc(_src(access="ftp")))
    assert issubclass(ManifestError, ValueError)


@pytest.mark.req("DATA-10")
def test_load_manifest_from_yaml(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "version: 1\nsources:\n  - id: a\n    desc: x\n    url: https://e.org/a\n    access: http\n"
        "    approx_bytes: 5\n    license: MIT\n    target: hdd\n    sha256: TBD\n"
    )
    assert load_manifest(p).get("a").approx_bytes == 5


@pytest.mark.req("DATA-10")
def test_committed_manifest_schema():
    m = load_manifest(ROOT / "data" / "manifest.yaml")
    assert len(m.sources) >= 1
    for s in m.sources:
        assert s.access in ACCESS and s.target in TARGETS
        assert (s.target == "stream") == (s.access == "herbie_byte_range")
