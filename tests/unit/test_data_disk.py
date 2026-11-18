"""specs/01-data.md DATA-14: disk budget."""

from __future__ import annotations

import os
from collections import namedtuple

import pytest

from outrun.data.disk import (
    GB,
    DiskBudget,
    DiskSpaceError,
    check_budget,
    check_free_space,
    cut_plan,
    load_disk_budget,
    planned_bytes,
    used_bytes,
)
from outrun.data.manifest import Manifest, Source

Usage = namedtuple("Usage", "total used free")


def _s(i, target, nbytes, access="http"):
    if target == "stream":
        access = "herbie_byte_range"
    return Source(f"s{i}", "d", f"https://e/{i}", access, nbytes, "MIT", target, None)


@pytest.mark.req("DATA-14")
def test_default_budget_matches_design(tmp_path):
    assert DiskBudget() == DiskBudget(375 * GB, 225 * GB)
    assert GB == 10**9
    assert load_disk_budget(config_dir=tmp_path) == DiskBudget()  # no disk.yaml -> defaults


@pytest.mark.req("DATA-14")
def test_budget_from_disk_yaml(tmp_path):
    (tmp_path / "disk.yaml").write_text("disk:\n  hdd_cap_gb: 300\n  ssd_cap_gb: 200\n")
    assert load_disk_budget(config_dir=tmp_path) == DiskBudget(300 * GB, 200 * GB)


@pytest.mark.req("DATA-14")
def test_planned_bytes_per_target():
    m = Manifest(1, (_s(0, "hdd", 5), _s(1, "hdd", 7), _s(2, "stream", 100)))
    assert planned_bytes(m) == {"hdd": 12, "ssd": 0, "stream": 100}


@pytest.mark.req("DATA-14")
def test_check_budget():
    b = DiskBudget(100 * GB, 50 * GB)
    assert check_budget({"hdd": 100 * GB, "ssd": 50 * GB, "stream": 10**15}, b) == []
    v = check_budget({"hdd": 101 * GB, "ssd": 0, "stream": 0}, b)
    assert len(v) == 1 and "hdd" in v[0]
    v = check_budget({"hdd": 90 * GB, "ssd": 40 * GB, "stream": 0}, b, used={"hdd": 20 * GB, "ssd": 20 * GB})
    assert len(v) == 2 and any("hdd" in s for s in v) and any("ssd" in s for s in v)


@pytest.mark.req("DATA-14")
def test_used_bytes(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"y" * 5)
    os.symlink(tmp_path / "a", tmp_path / "link")
    assert used_bytes(tmp_path) == 15
    assert used_bytes(tmp_path / "missing") == 0


@pytest.mark.req("DATA-14")
def test_check_free_space(tmp_path):
    fake = lambda p: Usage(1000, 400, 600)  # noqa: E731
    assert check_free_space(tmp_path, 500, disk_usage=fake) == 600
    with pytest.raises(DiskSpaceError):
        check_free_space(tmp_path, 601, disk_usage=fake)
    with pytest.raises(DiskSpaceError):
        check_free_space(tmp_path, 500, disk_usage=fake, reserve_bytes=200)
    assert issubclass(DiskSpaceError, OSError) and not issubclass(DiskSpaceError, NotImplementedError)


@pytest.mark.req("DATA-14")
def test_cut_plan_order():
    cuts = [("ir107_2018_h1", 20), ("wsts", 48), ("tornet_2016", 17)]
    assert cut_plan(300, 375, cuts) == []
    assert cut_plan(390, 375, cuts) == ["ir107_2018_h1"]
    assert cut_plan(400, 375, cuts) == ["ir107_2018_h1", "wsts"]
    with pytest.raises(DiskSpaceError):
        cut_plan(500, 375, cuts)
