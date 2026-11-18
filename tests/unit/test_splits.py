"""specs/15-evaluation.md: splits, quarantine, case hold-out, hashes (SPLIT-01..SPLIT-11).

Synthetic sample lists only. Case dates and the TorNet official assignment are made up here: they are inputs,
never facts baked into the code.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime, timedelta

import pytest

from outrun.eval.casestudy import CaseStudy, case_quarantine
from outrun.eval.splits import (
    SPLIT_NAMES,
    Split,
    SplitHashMismatch,
    SplitItem,
    SplitLeakageError,
    assert_valid_split,
    case_holdout,
    flood_split,
    ice_season,
    ice_split,
    load_quarantine,
    quarantine_dates,
    read_split,
    sevir_split,
    smoke_split,
    split_hash,
    tornet_split,
    validate_split,
    verify_hashes,
    wildfire_split,
    write_hashes,
    write_quarantine,
    write_split,
)


# ---------------------------------------------------------------- helpers (independent of validate_split)
def dt(y: int, m: int, d: int, h: int = 12) -> datetime:
    return datetime(y, m, d, h, tzinfo=UTC)


def days_between(start: date, end: date, step: int = 1):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step)


def ids(items) -> set[str]:
    return {i.id for i in items}


def days_of(items) -> set[date]:
    return {i.day for i in items if i.day is not None}


def assert_sorted(split: Split) -> None:
    for s in (split.train, split.val, split.test, *split.extra.values()):
        assert isinstance(s, tuple)
        assert [i.id for i in s] == sorted(i.id for i in s)


def assert_clean(split: Split) -> None:
    """Disjoint ids/groups and >= 2 days between any two of train/val/test (one gap day)."""
    sets = {"train": split.train, "val": split.val, "test": split.test}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not ids(sets[a]) & ids(sets[b]), (a, b)
            ga = {x.group for x in sets[a] if x.group is not None}
            gb = {x.group for x in sets[b] if x.group is not None}
            assert not ga & gb, (a, b)
            da, db = sorted(days_of(sets[a])), sorted(days_of(sets[b]))
            if da and db:
                gap = min(abs((x - y).days) for x in da for y in db)
                assert gap >= 2, f"{a}/{b} closest days only {gap} apart"
    assert_sorted(split)


# ---------------------------------------------------------------- synthetic inputs per builder
def sevir_events():
    ev = [
        (f"S{d:%Y%m%d}", dt(d.year, d.month, d.day))
        for d in days_between(date(2017, 12, 20), date(2019, 9, 30), 3)
    ]
    ev += [("E_gap1", dt(2019, 3, 31)), ("E_gap2", dt(2019, 5, 31)), ("E_cut", dt(2019, 6, 1, 0))]
    return ev


def tornet_samples():
    out = []
    for d in days_between(date(2019, 1, 1), date(2019, 12, 31)):
        jd = d.timetuple().tm_yday
        official = "test" if jd % 6 == 0 else "train"
        out.append((f"T{d:%m%d}a", dt(d.year, d.month, d.day, 1), official))
        out.append((f"T{d:%m%d}b", dt(d.year, d.month, d.day, 22), official))
    return out


def flood_samples():
    s = [
        (f"F{d:%Y%m%d}", dt(d.year, d.month, d.day))
        for d in days_between(date(2020, 6, 1), date(2025, 12, 31), 5)
    ]
    s += [
        ("F_dec31_23", dt(2023, 12, 31, 23)),
        ("F_dec31_24", dt(2024, 12, 31, 6)),
        ("F_jan1_25", dt(2025, 1, 1, 0)),
    ]
    return s


def wildfire_inputs():
    ndws = {
        "train": [f"n{i}" for i in range(20)],
        "eval": [f"e{i}" for i in range(5)],
        "test": [f"t{i}" for i in range(5)],
    }
    fires = [("w19a", date(2019, 7, 1)), ("w20a", date(2020, 8, 2)), ("w21a", date(2021, 7, 10)),
             ("w21b", date(2021, 8, 20)), ("w21c", date(2021, 9, 5))]  # fmt: skip
    return ndws, fires


def ice_bags():
    bags = []
    for season in range(2019, 2026):
        for d in days_between(date(season, 11, 15), date(season + 1, 3, 15), 4):
            for h in (0, 6):
                bags.append((f"OKZ0{season % 10}", dt(d.year, d.month, d.day, h)))
    bags += [("TXZ001", dt(2023, 6, 30, 0)), ("TXZ001", dt(2024, 6, 30, 0)), ("TXZ001", dt(2023, 7, 1, 0))]
    return bags


def smoke_inputs():
    # 10 blocks of 200 km with 1..4 monitors each (uneven sizes), 25 monitors in total
    sizes = [1, 2, 3, 4, 1, 2, 3, 4, 3, 2]
    monitors, k = {}, 0
    for b, n in enumerate(sizes):
        for j in range(n):
            monitors[f"M{k:02d}"] = (b * 200_000.0 + 1_000.0 + j * 10_000.0, 1_300_000.0 + 5_000.0 * j)
            k += 1
    samples = []
    for i, m in enumerate(sorted(monitors)):
        for d in days_between(date(2022, 1, 1) + timedelta(days=i % 7), date(2024, 12, 31), 7):
            samples.append((m, dt(d.year, d.month, d.day, 6)))
        samples += [(m, dt(2023, 6, 30, 0)), (m, dt(2023, 12, 31, 18))]
    return samples, monitors


BUILDERS = {
    "sevir": lambda q: sevir_split(sevir_events(), quarantine=q),
    "tornet": lambda q: tornet_split(tornet_samples(), quarantine=q, seed=0),
    "flood": lambda q: flood_split(flood_samples(), quarantine=q),
    "wildfire": lambda q: wildfire_split(*wildfire_inputs(), quarantine=q),
    "ice": lambda q: ice_split(ice_bags(), quarantine=q),
    "smoke": lambda q: smoke_split(*smoke_inputs(), quarantine=q),
}


# ---------------------------------------------------------------- SPLIT-01
@pytest.mark.req("SPLIT-01")
def test_sevir_split_official_cut_and_gap_days():
    s = sevir_split(sevir_events())
    assert_clean(s)
    assert all(i.day >= date(2019, 6, 1) for i in s.test)
    assert all(date(2019, 4, 1) <= i.day <= date(2019, 5, 30) for i in s.val)
    assert all(date(2018, 1, 1) <= i.day <= date(2019, 3, 30) for i in s.train)
    assert "E_cut" in ids(s.test)
    all_ids = ids(s.train) | ids(s.val) | ids(s.test)
    assert "E_gap1" not in all_ids and "E_gap2" not in all_ids
    assert not any(i.day.year == 2017 for i in s.train + s.val + s.test)
    expected_test = {e for e, t in sevir_events() if t.date() >= date(2019, 6, 1)}
    assert ids(s.test) == expected_test
    assert all(i.group == i.id for i in s.train + s.val + s.test)
    assert validate_split(s) == []


@pytest.mark.req("SPLIT-01")
def test_sevir_split_rejects_naive_and_duplicate():
    with pytest.raises(ValueError):
        sevir_split([("a", datetime(2018, 5, 1, 12))])
    with pytest.raises(ValueError):
        sevir_split([("a", dt(2018, 5, 1)), ("a", dt(2018, 6, 1))])


# ---------------------------------------------------------------- SPLIT-02
@pytest.mark.req("SPLIT-02")
def test_tornet_split_keeps_official_test_and_groups_days():
    samples = tornet_samples()
    s = tornet_split(samples, val_frac=0.1, seed=0)
    assert_clean(s)
    assert ids(s.test) == {c for c, _, o in samples if o == "test"}
    test_days = {t.date() for _, t, o in samples if o == "test"}
    pool_days = {t.date() for _, t, o in samples if o == "train"} - {
        d + timedelta(days=k) for d in test_days for k in (-1, 0, 1)
    }
    val_days = days_of(s.val)
    assert len(val_days) == max(1, round(0.1 * len(pool_days)))
    assert val_days <= pool_days
    # both chips of a val day are in val (grouped by day)
    assert all(sum(1 for i in s.val if i.day == d) == 2 for d in val_days)
    assert s == tornet_split(samples, val_frac=0.1, seed=0)
    assert validate_split(s) == []


@pytest.mark.req("SPLIT-02")
def test_tornet_split_rejects_unknown_official_label():
    with pytest.raises(ValueError):
        tornet_split([("x", dt(2019, 5, 1), "validation")])


# ---------------------------------------------------------------- SPLIT-03
@pytest.mark.req("SPLIT-03")
def test_flood_split_years_and_gap_days():
    s = flood_split(flood_samples(), test_year=2025)
    assert_clean(s)
    assert {i.day.year for i in s.train} <= {2021, 2022, 2023}
    assert {i.day.year for i in s.val} == {2024}
    assert {i.day.year for i in s.test} == {2025}
    assert "F_dec31_23" not in ids(s.train) and "F_dec31_24" not in ids(s.val)
    assert "F_jan1_25" in ids(s.test)
    assert not any(i.day.year == 2020 for i in s.train)
    assert validate_split(s) == []


@pytest.mark.req("SPLIT-03")
def test_flood_split_fallback_year_and_error():
    s = flood_split(flood_samples(), test_year=2024)
    assert {i.day.year for i in s.train} <= {2021, 2022}
    assert {i.day.year for i in s.val} == {2023} and {i.day.year for i in s.test} == {2024}
    assert "F_dec31_23" not in ids(s.val)
    with pytest.raises(ValueError):
        flood_split(flood_samples(), test_year=2022)


# ---------------------------------------------------------------- SPLIT-04
@pytest.mark.req("SPLIT-04")
def test_wildfire_split_ndws_files_and_wsts_2021():
    ndws, fires = wildfire_inputs()
    s = wildfire_split(ndws, fires)
    assert ids(s.train) == {f"ndws:{i}" for i in ndws["train"]}
    assert ids(s.val) == {f"ndws:{i}" for i in ndws["eval"]}
    assert ids(s.test) == {f"ndws:{i}" for i in ndws["test"]}
    assert all(i.day is None for i in s.train + s.val + s.test)
    assert ids(s.extra["test_wsts2021"]) == {"wsts:w21a", "wsts:w21b", "wsts:w21c"}
    everything = set().union(ids(s.train), ids(s.val), ids(s.test), *(ids(v) for v in s.extra.values()))
    assert "wsts:w19a" not in everything and "wsts:w20a" not in everything
    assert_clean(s)
    assert validate_split(s) == []


@pytest.mark.req("SPLIT-04")
def test_wildfire_split_rejects_leaky_or_malformed_ndws():
    ndws, fires = wildfire_inputs()
    leaky = dict(ndws, test=ndws["test"] + ["n3"])
    with pytest.raises(ValueError):
        wildfire_split(leaky, fires)
    with pytest.raises(ValueError):
        wildfire_split({"train": ndws["train"], "test": ndws["test"]}, fires)


# ---------------------------------------------------------------- SPLIT-05
@pytest.mark.req("SPLIT-05")
def test_ice_season_key():
    assert ice_season(date(2024, 1, 15)) == 2023
    assert ice_season(date(2023, 12, 1)) == 2023
    assert ice_season(date(2023, 6, 30)) == 2022
    assert ice_season(dt(2023, 7, 1, 0)) == 2023


@pytest.mark.req("SPLIT-05")
def test_ice_split_forward_winters_and_gap_days():
    s = ice_split(ice_bags())
    assert_clean(s)
    assert {ice_season(i.day) for i in s.train} == {2020, 2021, 2022}
    assert {ice_season(i.day) for i in s.val} == {2023}
    assert {ice_season(i.day) for i in s.test} == {2024}
    all_items = s.train + s.val + s.test
    assert all(i.group == i.id and "|" in i.id for i in all_items)
    assert "TXZ001|2023-06-30T00:00:00+00:00" not in ids(all_items)
    assert "TXZ001|2024-06-30T00:00:00+00:00" not in ids(all_items)
    assert "TXZ001|2023-07-01T00:00:00+00:00" in ids(s.val)
    s2 = ice_split(ice_bags(), test_seasons=(2024, 2025))
    assert {ice_season(i.day) for i in s2.test} == {2024, 2025}
    with pytest.raises(ValueError):
        ice_split(ice_bags(), val_season=2022)  # overlaps train seasons
    assert validate_split(s) == []


# ---------------------------------------------------------------- SPLIT-06
@pytest.mark.req("SPLIT-06")
def test_smoke_split_time_periods():
    samples, monitors = smoke_inputs()
    s = smoke_split(samples, monitors, seed=0)
    assert_clean(s)
    assert all(date(2022, 1, 1) <= i.day <= date(2023, 6, 29) for i in s.train)
    assert all(date(2023, 7, 1) <= i.day <= date(2023, 12, 30) for i in s.val)
    assert all(i.day.year == 2024 for i in s.test)
    assert {i.id.split("|")[0] for i in s.test} == set(monitors)  # test keeps every monitor
    assert validate_split(s) == []


@pytest.mark.req("SPLIT-06")
def test_smoke_split_spatial_holdout_blocks():
    samples, monitors = smoke_inputs()
    s = smoke_split(samples, monitors, seed=0)
    held = set(s.meta["holdout_monitors"])
    assert s.meta["holdout_monitors"] == sorted(held)
    block = {m: (math.floor(x / 200_000), math.floor(y / 200_000)) for m, (x, y) in monitors.items()}
    held_blocks = {block[m] for m in held}
    assert sorted(map(list, held_blocks)) == sorted(map(list, s.meta["holdout_blocks"]))
    assert all(m in held for m in monitors if block[m] in held_blocks)  # whole blocks only
    k = math.ceil(0.2 * len(monitors))
    assert len(held) >= k
    sizes = [sum(1 for m in monitors if block[m] == b) for b in held_blocks]
    assert len(held) - max(sizes) < k  # shortest prefix
    trainval_monitors = {i.id.split("|")[0] for i in s.train + s.val}
    assert not trainval_monitors & held
    assert ids(s.extra["test_spatial"]) == {i.id for i in s.test if i.id.split("|")[0] in held}
    assert s == smoke_split(samples, monitors, seed=0)


@pytest.mark.req("SPLIT-06")
@pytest.mark.parametrize("seed", [0, 3])
def test_spatial_holdout_monitors_reference(seed):
    import numpy as np

    from outrun.eval.splits import spatial_holdout_monitors

    _, monitors = smoke_inputs()
    block = {m: (math.floor(x / 200_000), math.floor(y / 200_000)) for m, (x, y) in monitors.items()}
    blocks = sorted(set(block.values()))
    order = [blocks[i] for i in np.random.default_rng(seed).permutation(len(blocks))]
    k = math.ceil(0.2 * len(monitors))
    held: set[str] = set()
    for b in order:
        if len(held) >= k:
            break
        held |= {m for m in monitors if block[m] == b}
    assert spatial_holdout_monitors(monitors, 0.2, 200_000, seed) == frozenset(held)
    s = smoke_split(*smoke_inputs(), seed=seed)
    sampled = {i.id.split("|")[0] for i in s.train + s.val + s.test} | set(s.meta["holdout_monitors"])
    assert set(s.meta["holdout_monitors"]) == set(
        spatial_holdout_monitors({m: monitors[m] for m in sampled}, 0.2, 200_000, seed)
    )
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            spatial_holdout_monitors(monitors, bad)


@pytest.mark.req("SPLIT-06")
def test_smoke_split_unknown_monitor():
    samples, monitors = smoke_inputs()
    with pytest.raises(ValueError):
        smoke_split(samples + [("ZZZ", dt(2022, 3, 1))], monitors)


# ---------------------------------------------------------------- SPLIT-07
@pytest.mark.req("SPLIT-07")
def test_quarantine_dates_radius():
    q = quarantine_dates([date(2019, 5, 26)])
    assert q == frozenset(date(2019, 5, d) for d in range(23, 30))
    assert len(quarantine_dates([date(2024, 1, 1), date(2024, 1, 3)])) == 9
    assert quarantine_dates([date(2024, 1, 1)], radius_days=0) == frozenset({date(2024, 1, 1)})


@pytest.mark.req("SPLIT-07")
def test_quarantine_file_roundtrip_and_tamper(tmp_path):
    p = write_quarantine(tmp_path / "splits" / "quarantine.json", {"case_a": date(2019, 5, 26)})
    d = json.loads(p.read_text())
    assert d["radius_days"] == 3 and d["cases"] == {"case_a": "2019-05-26"}
    assert d["dates"] == sorted(d["dates"]) and len(d["dates"]) == 7
    assert load_quarantine(p) == quarantine_dates([date(2019, 5, 26)])
    d["dates"] = d["dates"][1:]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_quarantine(p)


@pytest.mark.req("SPLIT-07")
@pytest.mark.req("SPLIT-08")
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_builder_quarantines_train_val_but_not_test(name):
    build = BUILDERS[name]
    base = build(frozenset())
    assert base.extra.get("case_holdout", ()) == ()
    dated_tv = [i for i in base.train + base.val if i.day is not None]
    dated_test = [i for i in base.test + base.extra.get("test_wsts2021", ()) if i.day is not None]
    centres = []
    if dated_tv:
        centres.append(dated_tv[len(dated_tv) // 2].day)
    if dated_test:
        centres.append(dated_test[len(dated_test) // 2].day)
    q = frozenset(c + timedelta(days=k) for c in centres for k in range(-3, 4))
    s = build(q)
    assert s.test == base.test  # official test set untouched
    if "test_wsts2021" in base.extra:
        assert s.extra["test_wsts2021"] == base.extra["test_wsts2021"]
    assert not days_of(s.train + s.val) & q
    hold = s.extra["case_holdout"]
    assert hold and all(i.day in q for i in hold)
    expected_in_hold = {i.id for i in base.train + base.val + base.test + base.extra.get("test_wsts2021", ())
                        if i.day in q}  # fmt: skip
    assert expected_in_hold <= ids(hold)
    assert [i.id for i in hold] == sorted(ids(hold))
    assert validate_split(s) == []


# ---------------------------------------------------------------- SPLIT-08
@pytest.mark.req("SPLIT-08")
def test_case_holdout_selects_quarantined_days():
    items = [SplitItem("b", date(2019, 5, 25)), SplitItem("a", date(2019, 5, 26)),
             SplitItem("c", date(2019, 6, 10)), SplitItem("d", None)]  # fmt: skip
    q = frozenset({date(2019, 5, 25), date(2019, 5, 26)})
    assert case_holdout(items, q) == (SplitItem("a", date(2019, 5, 26)), SplitItem("b", date(2019, 5, 25)))
    assert case_holdout(items, frozenset()) == ()


# ---------------------------------------------------------------- SPLIT-09
def _clean_split() -> Split:
    return Split(
        "toy",
        train=(SplitItem("a", date(2020, 1, 1), "g1"), SplitItem("b", date(2020, 1, 2), "g2")),
        val=(SplitItem("c", date(2020, 1, 5), "g3"),),
        test=(SplitItem("d", date(2020, 1, 8), "g4"), SplitItem("e", date(2020, 1, 9))),
        extra={"test_spatial": (SplitItem("e", date(2020, 1, 9)),)},
    )


@pytest.mark.req("SPLIT-09")
def test_validate_split_clean_and_extra_may_overlap_test():
    assert validate_split(_clean_split()) == []
    assert_valid_split(_clean_split())


@pytest.mark.req("SPLIT-09")
@pytest.mark.parametrize("kind", ["dup_id", "dup_group", "same_day", "gap_day", "extra_leak", "extra_day"])
def test_validate_split_reports_each_violation(kind):
    s = _clean_split()
    if kind == "dup_id":
        s = Split("toy", s.train, s.val + (SplitItem("a", date(2020, 1, 5)),), s.test)
    elif kind == "dup_group":
        s = Split("toy", s.train, (SplitItem("c", date(2020, 1, 5), "g1"),), s.test)
    elif kind == "same_day":
        s = Split("toy", s.train, (SplitItem("c", date(2020, 1, 2)),), s.test)
    elif kind == "gap_day":
        s = Split("toy", s.train, (SplitItem("c", date(2020, 1, 3)),), s.test)  # 1 day after train
    elif kind == "extra_leak":
        s = Split("toy", s.train, s.val, s.test, extra={"case_holdout": (SplitItem("b", date(2020, 1, 2)),)})
    elif kind == "extra_day":
        s = Split("toy", s.train, s.val, s.test, extra={"case_holdout": (SplitItem("z", date(2020, 1, 5)),)})
    problems = validate_split(s)
    assert problems and all(isinstance(p, str) for p in problems)
    with pytest.raises(SplitLeakageError):
        assert_valid_split(s)
    assert issubclass(SplitLeakageError, ValueError)


@pytest.mark.req("SPLIT-09")
def test_validate_split_gap_days_parameter():
    s = Split("toy", (SplitItem("a", date(2020, 1, 1)),), (SplitItem("b", date(2020, 1, 2)),), ())
    assert validate_split(s, gap_days=0) == []
    assert validate_split(s, gap_days=1)


# ---------------------------------------------------------------- SPLIT-10
@pytest.mark.req("SPLIT-10")
def test_write_read_split_roundtrip_and_hash(tmp_path):
    s = Split("toy", _clean_split().train, _clean_split().val, _clean_split().test,
              extra={"case_holdout": ()}, meta={"note": "x", "blocks": [[1, 2]]})  # fmt: skip
    p = write_split(s, tmp_path / "toy.json")
    raw = p.read_bytes()
    assert raw.endswith(b"\n")
    obj = json.loads(raw)
    assert raw.decode() == json.dumps(obj, sort_keys=True, indent=2) + "\n"
    assert read_split(p) == s
    assert split_hash(s) == hashlib.sha256(raw).hexdigest()
    s_other = Split("toy", s.train[:1], s.val, s.test)
    assert split_hash(s_other) != split_hash(s)


@pytest.mark.req("SPLIT-10")
def test_hashes_detect_changes(tmp_path):
    d = tmp_path / "splits"
    d.mkdir()
    write_split(_clean_split(), d / "sevir.json")
    write_split(Split("flood", (), (), (SplitItem("t1", date(2025, 3, 1)),)), d / "flood.json")
    write_quarantine(d / "quarantine.json", {"c": date(2019, 5, 26)})
    h = write_hashes(d)
    assert set(h) == {"sevir", "flood", "quarantine"}
    assert json.loads((d / "hashes.json").read_text()) == h
    assert h["flood"]["test_ids_sha256"] == hashlib.sha256(b"t1").hexdigest()
    assert h["sevir"]["test_ids_sha256"] == hashlib.sha256(b"d\ne").hexdigest()
    assert h["sevir"]["file_sha256"] == hashlib.sha256((d / "sevir.json").read_bytes()).hexdigest()
    assert verify_hashes(d) == []
    (d / "flood.json").write_text((d / "flood.json").read_text().replace("t1", "t2"))
    assert verify_hashes(d) == ["flood"]
    with pytest.raises(SplitHashMismatch):
        verify_hashes(d, strict=True)
    write_split(_clean_split(), d / "smoke.json")  # unlisted file
    assert verify_hashes(d) == ["flood", "smoke"]
    (d / "sevir.json").unlink()
    assert verify_hashes(d) == ["flood", "sevir", "smoke"]
    (d / "hashes.json").unlink()
    assert verify_hashes(d) == ["flood", "quarantine", "smoke"]


@pytest.mark.req("SPLIT-10")
def test_split_names_fixed_by_design():
    assert set(SPLIT_NAMES) == {"sevir", "tornet", "flood", "ice", "smoke", "ndws", "wsts"}


# ---------------------------------------------------------------- SPLIT-11
@pytest.mark.req("SPLIT-11")
def test_case_quarantine_only_quarantine_cases():
    cases = [
        CaseStudy("primary", ("storm", "tornado"), date(2019, 5, 26), "El Reno", "quarantine", primary=True),
        CaseStudy("ok_test_storm", ("storm",), date(2019, 7, 1), "OKC", "official_test"),
        CaseStudy("flood_case", ("flood",), date(2025, 5, 1), "OKC", "quarantine"),
    ]
    assert case_quarantine(cases) == {"primary": date(2019, 5, 26), "flood_case": date(2025, 5, 1)}
    with pytest.raises(ValueError):
        case_quarantine(cases + [CaseStudy("primary", ("ice",), date(2025, 1, 5), "OKC", "quarantine")])
    with pytest.raises(ValueError):
        case_quarantine(
            cases + [CaseStudy("p2", ("ice",), date(2025, 1, 5), "OKC", "quarantine", primary=True)]
        )


@pytest.mark.req("SPLIT-07")
def test_split_mask_selects_only_the_named_part():
    import numpy as np

    from outrun.eval.splits import Split, SplitItem, split_mask

    split = Split(
        "toy",
        train=(SplitItem("a"), SplitItem("b")),
        val=(SplitItem("c"),),
        test=(SplitItem("d"),),
        extra={"case_holdout": (SplitItem("q", date(2024, 5, 6)),)},
    )
    ids = ["d", "a", "q", "c", "b", "zz"]
    np.testing.assert_array_equal(split_mask(ids, split, "train"), [False, True, False, False, True, False])
    np.testing.assert_array_equal(split_mask(ids, split, "val"), [False, False, False, True, False, False])
    np.testing.assert_array_equal(split_mask(ids, split, "test"), [True, False, False, False, False, False])
    np.testing.assert_array_equal(
        split_mask(ids, split, "case_holdout"), [False, False, True, False, False, False]
    )
    assert split_mask([], split, "train").shape == (0,)
    with pytest.raises(ValueError):
        split_mask(ids, split, "holdout")
