"""Frozen train/val/test splits, quarantine of case-study days and split hashes (specs/15-evaluation.md, SPLIT-*).

Every hazard dataset builder (``hazards/<h>/dataset.py``) takes its sample IDs from the ``Split`` built here and
frozen in ``splits/<name>.json`` (names fixed by DESIGN 2.1: sevir, tornet, flood, ice, smoke, ndws, wsts,
quarantine, hashes). Official test splits are never modified: gap days and quarantine only remove items from
train and val.

Builders (one per hazard, SPLIT-01..SPLIT-06) all return a ``Split`` whose sets are sorted by ``SplitItem.id``,
that passes ``validate_split(split, gap_days=1)``, and whose ``extra["case_holdout"]`` holds the input items whose
day is quarantined (SPLIT-08).
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SPLIT_NAMES",
    "Split",
    "SplitHashMismatch",
    "SplitItem",
    "SplitLeakageError",
    "assert_valid_split",
    "case_holdout",
    "flood_split",
    "ice_season",
    "ice_split",
    "load_quarantine",
    "quarantine_dates",
    "read_split",
    "sevir_split",
    "smoke_split",
    "split_mask",
    "spatial_holdout_monitors",
    "split_hash",
    "tornet_split",
    "validate_split",
    "verify_hashes",
    "wildfire_split",
    "write_hashes",
    "write_quarantine",
    "write_split",
]

SPLIT_NAMES: tuple[str, ...] = ("sevir", "tornet", "flood", "ice", "smoke", "ndws", "wsts")


class SplitLeakageError(ValueError):
    """A split violates disjointness or the gap-day rule (SPLIT-09)."""


class SplitHashMismatch(RuntimeError):
    """A frozen split file no longer matches ``splits/hashes.json`` (SPLIT-10)."""


@dataclass(frozen=True, order=True)
class SplitItem:
    """One sample, event, chip, bag or fire.

    ``day`` is its UTC calendar day (``None`` when the dataset has no dates, e.g. NDWS). Items that share a
    non-None ``group`` (e.g. a SEVIR event id) must all be in the same set.
    """

    id: str
    day: date | None = None
    group: str | None = None


@dataclass(frozen=True)
class Split:
    """A frozen split. ``extra`` holds named additional evaluation sets (``case_holdout``, ``test_wsts2021``,
    ``test_spatial``); ``meta`` holds JSON-serialisable builder facts (lists, not tuples)."""

    dataset: str
    train: tuple[SplitItem, ...]
    val: tuple[SplitItem, ...]
    test: tuple[SplitItem, ...]
    extra: Mapping[str, tuple[SplitItem, ...]] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ small helpers
def _utc_day(t: datetime) -> date:
    """UTC calendar day of a tz-aware datetime (naive -> ValueError)."""
    if not isinstance(t, datetime) or t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"expected a timezone-aware datetime, got {t!r}")
    return t.astimezone(UTC).date()


def _sorted(items: Iterable[SplitItem]) -> tuple[SplitItem, ...]:
    return tuple(sorted(items, key=lambda i: i.id))


def _check_unique(ids: Iterable[str]) -> None:
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate input id {i!r}")
        seen.add(i)


def _finish(
    dataset: str,
    train: Iterable[SplitItem],
    val: Iterable[SplitItem],
    test: Iterable[SplitItem],
    in_period: Iterable[SplitItem],
    quarantine: Collection[date],
    extra: Mapping[str, tuple[SplitItem, ...]] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Split:
    """Remove quarantined days from train/val, attach ``case_holdout`` and sort every set."""
    q = frozenset(quarantine)
    keep = [i for i in train if i.day not in q]
    keep_val = [i for i in val if i.day not in q]
    ex = {"case_holdout": case_holdout(in_period, q)}
    ex.update(extra or {})
    return Split(dataset, _sorted(keep), _sorted(keep_val), _sorted(test), ex, dict(meta or {}))


# ------------------------------------------------------------------ quarantine (SPLIT-07)
def quarantine_dates(case_dates: Iterable[date], radius_days: int = 3) -> frozenset[date]:
    """SPLIT-07. Every case date +- ``radius_days`` (inclusive)."""
    return frozenset(d + timedelta(days=k) for d in case_dates for k in range(-radius_days, radius_days + 1))


def write_quarantine(path: str | Path, cases: Mapping[str, date], radius_days: int = 3) -> Path:
    """SPLIT-07. Write ``splits/quarantine.json``: ``{"radius_days", "cases": {id: "YYYY-MM-DD"}, "dates": [...]}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "radius_days": radius_days,
        "cases": {k: v.isoformat() for k, v in cases.items()},
        "dates": sorted(d.isoformat() for d in quarantine_dates(cases.values(), radius_days)),
    }
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    return path


def load_quarantine(path: str | Path) -> frozenset[date]:
    """SPLIT-07. Read ``quarantine.json``; ``ValueError`` if ``dates`` disagrees with ``cases`` +- ``radius_days``."""
    obj = json.loads(Path(path).read_text())
    dates = frozenset(date.fromisoformat(d) for d in obj["dates"])
    cases = [date.fromisoformat(d) for d in obj["cases"].values()]
    if len(dates) != len(obj["dates"]) or dates != quarantine_dates(cases, int(obj["radius_days"])):
        raise ValueError(f"{path}: 'dates' is not the +-{obj['radius_days']} day expansion of 'cases'")
    return dates


def split_mask(ids: Sequence[str], split: Split, part: str) -> np.ndarray:
    """SPLIT-07. Bool array, True where ``ids[i]`` is an item id of ``split.<part>``.

    ``part`` is "train", "val", "test" or a key of ``split.extra``; anything else -> ValueError. Every hazard dataset
    split factory selects its rows with this helper, so quarantined and test ids never reach train or val.
    """
    if part in ("train", "val", "test"):
        items = getattr(split, part)
    elif part in split.extra:
        items = split.extra[part]
    else:
        raise ValueError(f"unknown split part {part!r}")
    wanted = {i.id for i in items}
    return np.fromiter((i in wanted for i in ids), dtype=bool, count=len(ids))


# ------------------------------------------------------------------ case hold-out (SPLIT-08)
def case_holdout(items: Iterable[SplitItem], quarantine: Collection[date]) -> tuple[SplitItem, ...]:
    """SPLIT-08. Items whose day is quarantined, sorted by id (the separately reported case hold-out set)."""
    q = frozenset(quarantine)
    return _sorted(i for i in items if i.day is not None and i.day in q)


# ------------------------------------------------------------------ validation (SPLIT-09)
def _min_day_gap(a: list[date], b: list[date]) -> int | None:
    """Smallest |x - y| in days between two sorted day lists (None if either is empty)."""
    if not a or not b:
        return None
    best = None
    for x in a:
        k = bisect_left(b, x)
        for j in (k - 1, k):
            if 0 <= j < len(b):
                g = abs((x - b[j]).days)
                best = g if best is None else min(best, g)
    return best


def validate_split(split: Split, gap_days: int = 1) -> list[str]:
    """SPLIT-09. Human-readable violations; an empty list means the split is clean."""
    problems: list[str] = []
    sets = {"train": split.train, "val": split.val, "test": split.test}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = {x.id for x in sets[a]} & {x.id for x in sets[b]}
            if both:
                problems.append(f"{len(both)} id(s) in both {a} and {b}, e.g. {sorted(both)[0]!r}")
            ga = {x.group for x in sets[a] if x.group is not None}
            gb = {x.group for x in sets[b] if x.group is not None}
            if ga & gb:
                problems.append(f"{len(ga & gb)} group(s) in both {a} and {b}, e.g. {sorted(ga & gb)[0]!r}")
            da = sorted({x.day for x in sets[a] if x.day is not None})
            db = sorted({x.day for x in sets[b] if x.day is not None})
            gap = _min_day_gap(da, db)
            if gap is not None and gap <= gap_days:
                problems.append(f"{a} and {b} have days only {gap} day(s) apart (gap_days={gap_days})")
    tv = split.train + split.val
    tv_ids = {x.id for x in tv}
    tv_groups = {x.group for x in tv if x.group is not None}
    tv_days = {x.day for x in tv if x.day is not None}
    for name, items in split.extra.items():
        for x in items:
            if x.id in tv_ids:
                problems.append(f"extra {name!r} item {x.id!r} is also in train/val")
            elif x.group is not None and x.group in tv_groups:
                problems.append(f"extra {name!r} item {x.id!r} shares group {x.group!r} with train/val")
            elif x.day is not None and x.day in tv_days:
                problems.append(f"extra {name!r} item {x.id!r} is on a train/val day {x.day}")
    return problems


def assert_valid_split(split: Split, gap_days: int = 1) -> None:
    """SPLIT-09. Raise ``SplitLeakageError`` listing the violations, if any."""
    problems = validate_split(split, gap_days)
    if problems:
        raise SplitLeakageError(f"split {split.dataset!r}: " + "; ".join(problems))


# ------------------------------------------------------------------ per-hazard builders (SPLIT-01..06)
def sevir_split(events: Iterable[tuple[str, datetime]], quarantine: Collection[date] = ()) -> Split:
    """SPLIT-01. SEVIR storm events (event_id, start time UTC) to the official-cut split."""
    events = list(events)
    _check_unique(e for e, _ in events)
    items = [SplitItem(str(e), _utc_day(t), str(e)) for e, t in events]
    in_period = [i for i in items if i.day >= date(2018, 1, 1)]
    test = [i for i in in_period if i.day >= date(2019, 6, 1)]
    val = [i for i in in_period if date(2019, 4, 1) <= i.day <= date(2019, 5, 30)]
    train = [i for i in in_period if i.day <= date(2019, 3, 30)]
    return _finish("sevir", train, val, test, in_period, quarantine)


def tornet_split(
    samples: Iterable[tuple[str, datetime, str]],
    quarantine: Collection[date] = (),
    val_frac: float = 0.1,
    seed: int = 0,
) -> Split:
    """SPLIT-02. TorNet chips (chip_id, time UTC, official split "train"|"test") to train/val/test."""
    samples = list(samples)
    _check_unique(c for c, _, _ in samples)
    items: list[tuple[SplitItem, str]] = []
    for c, t, o in samples:
        if o not in ("train", "test"):
            raise ValueError(f"official split of {c!r} must be 'train' or 'test', got {o!r}")
        items.append((SplitItem(str(c), _utc_day(t)), o))
    q = frozenset(quarantine)
    test = [i for i, o in items if o == "test"]
    test_days = {i.day for i in test}
    near_test = {d + timedelta(days=k) for d in test_days for k in (-1, 0, 1)}
    pool = [i for i, o in items if o == "train" and i.day not in q and i.day not in near_test]
    pool_days = sorted({i.day for i in pool})
    if len(pool_days) < 2:
        val_days: set[date] = set()
    else:
        n_val = max(1, round(val_frac * len(pool_days)))
        pick = np.random.default_rng(seed).choice(len(pool_days), size=n_val, replace=False)
        val_days = {pool_days[k] for k in pick}
    near_val = {d + timedelta(days=k) for d in val_days for k in (-1, 0, 1)}
    val = [i for i in pool if i.day in val_days]
    train = [i for i in pool if i.day not in near_val]
    meta = {"val_days": sorted(d.isoformat() for d in val_days), "val_frac": val_frac, "seed": seed}
    return _finish("tornet", train, val, test, [i for i, _ in items], q, meta=meta)


def flood_split(
    samples: Iterable[tuple[str, datetime]],
    quarantine: Collection[date] = (),
    test_year: int = 2025,
    first_year: int = 2021,
) -> Split:
    """SPLIT-03. Flood samples (sample_id, valid time UTC) to a forward-in-time year split."""
    if test_year - 2 < first_year:
        raise ValueError(f"test_year {test_year} leaves no train year from first_year {first_year}")
    samples = list(samples)
    _check_unique(s for s, _ in samples)
    items = [SplitItem(str(s), _utc_day(t)) for s, t in samples]
    in_period = [i for i in items if first_year <= i.day.year <= test_year]
    test = [i for i in in_period if i.day.year == test_year]
    val = [i for i in in_period if i.day.year == test_year - 1 and (i.day.month, i.day.day) != (12, 31)]
    train = [i for i in in_period if i.day.year <= test_year - 2 and i.day != date(test_year - 2, 12, 31)]
    meta = {"first_year": first_year, "test_year": test_year}
    return _finish("flood", train, val, test, in_period, quarantine, meta=meta)


def wildfire_split(
    ndws: Mapping[str, Iterable[str]],
    wsts_fires: Iterable[tuple[str, date]],
    quarantine: Collection[date] = (),
) -> Split:
    """SPLIT-04. NDWS official files (train/eval/test) plus WSTS 2021 as the out-of-time set."""
    if set(ndws) != {"train", "eval", "test"}:
        raise ValueError(f"ndws must map exactly train/eval/test, got {sorted(ndws)}")
    parts = {k: [f"ndws:{i}" for i in v] for k, v in ndws.items()}
    seen: dict[str, str] = {}
    for k, v in parts.items():
        for i in v:
            if i in seen:
                raise ValueError(f"NDWS id {i!r} appears in both {seen[i]!r} and {k!r}")
            seen[i] = k
    fires = list(wsts_fires)
    _check_unique(f for f, _ in fires)
    w21 = [SplitItem(f"wsts:{f}", d, str(f)) for f, d in fires if d.year == 2021]
    return _finish(
        "wildfire",
        [SplitItem(i) for i in parts["train"]],
        [SplitItem(i) for i in parts["eval"]],
        [SplitItem(i) for i in parts["test"]],
        w21,
        quarantine,
        extra={"test_wsts2021": _sorted(w21)},
    )


def ice_season(t: datetime | date) -> int:
    """SPLIT-05. Winter season key: the year the season starts (Jul 1 .. Jun 30); 2024-01-15 -> 2023."""
    return t.year if t.month >= 7 else t.year - 1


def ice_split(
    bags: Iterable[tuple[str, datetime]],
    quarantine: Collection[date] = (),
    train_seasons: tuple[int, ...] = (2020, 2021, 2022),
    val_season: int = 2023,
    test_seasons: tuple[int, ...] = (2024,),
) -> Split:
    """SPLIT-05. Ice bags (zone_id, 6-h window start UTC) to a forward-in-time winter split."""
    tr, te = set(train_seasons), set(test_seasons)
    if tr & te or val_season in tr or val_season in te:
        raise ValueError("train, val and test seasons must not overlap")
    bags = list(bags)
    items = []
    for z, t in bags:
        day = _utc_day(t)
        key = f"{z}|{t.isoformat()}"
        items.append((SplitItem(key, day, key), ice_season(day)))
    _check_unique(i.id for i, _ in items)
    in_period = [i for i, s in items if s in tr or s in te or s == val_season]
    last_train = max(tr) if tr else None

    def season_end(s: int) -> date:
        return date(s + 1, 6, 30)

    train = [i for i, s in items if s in tr and not (s == last_train and i.day == season_end(s))]
    val = [i for i, s in items if s == val_season and i.day != season_end(s)]
    test = [i for i, s in items if s in te]
    meta = {"train_seasons": sorted(tr), "val_season": val_season, "test_seasons": sorted(te)}
    return _finish("ice", train, val, test, in_period, quarantine, meta=meta)


def spatial_holdout_monitors(
    monitors: Mapping[str, tuple[float, float]],
    frac: float = 0.2,
    block_m: float = 200_000.0,
    seed: int = 0,
) -> frozenset[str]:
    """SPLIT-06 / SMOKE-06. The one block-holdout algorithm: sorted ``(floor(x/block_m), floor(y/block_m))`` blocks,
    reordered by ``default_rng(seed).permutation(n_blocks)``; the shortest prefix holding at least
    ``ceil(frac * len(monitors))`` monitors is held out. ``frac`` outside (0, 1) -> ValueError."""
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    block = {m: (math.floor(x / block_m), math.floor(y / block_m)) for m, (x, y) in monitors.items()}
    blocks = sorted(set(block.values()))
    order = [blocks[i] for i in np.random.default_rng(seed).permutation(len(blocks))]
    k = math.ceil(frac * len(monitors))
    held: set[str] = set()
    for b in order:
        if len(held) >= k:
            break
        held |= {m for m, mb in block.items() if mb == b}
    return frozenset(held)


def smoke_split(
    samples: Iterable[tuple[str, datetime]],
    monitors: Mapping[str, tuple[float, float]],
    quarantine: Collection[date] = (),
    holdout_frac: float = 0.2,
    block_m: float = 200_000.0,
    seed: int = 0,
) -> Split:
    """SPLIT-06. Smoke samples (monitor_id, issue time UTC) to a time split plus spatially held-out monitors."""
    samples = list(samples)
    sampled = {m for m, _ in samples}
    missing = sampled - set(monitors)
    if missing:
        raise ValueError(f"sampled monitors without coordinates: {sorted(missing)[:5]}")
    items = [(m, SplitItem(f"{m}|{t.isoformat()}", _utc_day(t))) for m, t in samples]
    _check_unique(i.id for _, i in items)
    held = spatial_holdout_monitors({m: monitors[m] for m in sampled}, holdout_frac, block_m, seed)
    in_period = [i for _, i in items if date(2022, 1, 1) <= i.day <= date(2024, 12, 31)]
    train = [i for m, i in items if m not in held and date(2022, 1, 1) <= i.day <= date(2023, 6, 29)]
    val = [i for m, i in items if m not in held and date(2023, 7, 1) <= i.day <= date(2023, 12, 30)]
    test = [(m, i) for m, i in items if i.day.year == 2024]
    blocks = sorted(
        {(math.floor(monitors[m][0] / block_m), math.floor(monitors[m][1] / block_m)) for m in held}
    )
    meta = {"holdout_monitors": sorted(held), "holdout_blocks": [list(b) for b in blocks]}
    return _finish(
        "smoke",
        train,
        val,
        [i for _, i in test],
        in_period,
        quarantine,
        extra={"test_spatial": _sorted(i for m, i in test if m in held)},
        meta=meta,
    )


# ------------------------------------------------------------------ freeze + hashes (SPLIT-10)
def _item_obj(i: SplitItem) -> dict[str, Any]:
    return {"id": i.id, "day": i.day.isoformat() if i.day is not None else None, "group": i.group}


def _obj_item(o: Mapping[str, Any]) -> SplitItem:
    return SplitItem(o["id"], date.fromisoformat(o["day"]) if o["day"] is not None else None, o["group"])


def _canonical(split: Split) -> bytes:
    def items(s: Iterable[SplitItem]) -> list[dict[str, Any]]:
        return [_item_obj(i) for i in sorted(s, key=lambda i: i.id)]

    obj = {
        "dataset": split.dataset,
        "train": items(split.train),
        "val": items(split.val),
        "test": items(split.test),
        "extra": {k: items(v) for k, v in split.extra.items()},
        "meta": dict(split.meta),
    }
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode()


def write_split(split: Split, path: str | Path) -> Path:
    """SPLIT-10. Canonical JSON (sorted keys, items sorted by id, 2-space indent, trailing newline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(split))
    return path


def read_split(path: str | Path) -> Split:
    """SPLIT-10. Inverse of ``write_split``: ``read_split(write_split(s, p)) == s``."""
    obj = json.loads(Path(path).read_text())
    return Split(
        obj["dataset"],
        tuple(_obj_item(o) for o in obj["train"]),
        tuple(_obj_item(o) for o in obj["val"]),
        tuple(_obj_item(o) for o in obj["test"]),
        {k: tuple(_obj_item(o) for o in v) for k, v in obj.get("extra", {}).items()},
        obj.get("meta", {}),
    )


def split_hash(split: Split) -> str:
    """SPLIT-10. sha256 hex digest of the canonical JSON bytes ``write_split`` would write."""
    return hashlib.sha256(_canonical(split)).hexdigest()


def _hash_entry(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    entry = {"file_sha256": hashlib.sha256(raw).hexdigest()}
    obj = json.loads(raw)
    if isinstance(obj, dict) and {"train", "val", "test"} <= set(obj):
        test_ids = "\n".join(sorted(o["id"] for o in obj["test"]))
        entry["test_ids_sha256"] = hashlib.sha256(test_ids.encode()).hexdigest()
    return entry


def _json_files(splits_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(splits_dir.glob("*.json")) if p.name != "hashes.json"}


def write_hashes(splits_dir: str | Path) -> dict[str, dict[str, str]]:
    """SPLIT-10. Write ``hashes.json`` for every other ``*.json`` in ``splits_dir`` and return its content."""
    d = Path(splits_dir)
    hashes = {stem: _hash_entry(p) for stem, p in _json_files(d).items()}
    (d / "hashes.json").write_text(json.dumps(hashes, sort_keys=True, indent=2) + "\n")
    return hashes


def verify_hashes(splits_dir: str | Path, strict: bool = False) -> list[str]:
    """SPLIT-10. Names (file stems) whose file is changed, missing or unlisted; ``strict`` raises instead."""
    d = Path(splits_dir)
    files = _json_files(d)
    hp = d / "hashes.json"
    listed = json.loads(hp.read_text()) if hp.exists() else {}
    bad = set()
    for stem in set(files) | set(listed):
        if (
            stem not in files
            or stem not in listed
            or _hash_entry(files[stem])["file_sha256"] != listed[stem].get("file_sha256")
        ):
            bad.add(stem)
    result = sorted(bad)
    if strict and result:
        raise SplitHashMismatch(f"split files changed, missing or unlisted: {result}")
    return result
