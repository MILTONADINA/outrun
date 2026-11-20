"""T7: build and freeze ``splits/*.json`` from the real catalogs (SPLIT-01..10).

Run once, before any training: ``.venv/bin/python -m outrun.eval.freeze_splits [--raw DIR] [--out splits] [--cases FILE]``.
It writes ``quarantine.json``, the split files whose sample universe is known from RAW today (sevir, tornet, ndws,
wsts) and ``hashes.json``. flood, ice and smoke are written later, by the tasks that build their sample tables
(their builders are deterministic functions of dates + the frozen quarantine); ``hashes.json`` is then regenerated.

ID conventions (the dataset builders must use the same ones):
- SEVIR: catalog ``id`` (STORMEVENTS ``S...`` ids with both vil and ir107 in 2018-19); day = UTC day of the
  sequence start (``time_utc`` + first VIL minute offset).
- TorNet: catalog ``filename``; day = UTC day of ``start_time``; official split = ``type`` column (TORN-02).
- NDWS: ``ndws:<file>_<shard>:<record>``, e.g. ``ndws:train_03:00017`` (record index in its TFRecord file).
- WSTS: ``wsts:<fire folder>``, e.g. ``wsts:fire_25459876``; day = earliest daily GeoTIFF date of the fire.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from outrun.config import get_paths
from outrun.eval.casestudy import CaseStudy, case_quarantine
from outrun.eval.splits import (
    Split,
    assert_valid_split,
    load_quarantine,
    sevir_split,
    tornet_split,
    wildfire_split,
    write_hashes,
    write_quarantine,
    write_split,
)


def load_cases(path: Path) -> list[CaseStudy]:
    """Read the T7 case-study registry (``splits/cases.json``) as ``CaseStudy`` records (SPLIT-11)."""
    rows = json.loads(path.read_text())["cases"]
    return [
        CaseStudy(
            id=r["id"],
            hazards=tuple(r["hazards"]),
            day=date.fromisoformat(r["day"]),
            city=r["city"],
            held_out=r["held_out"],
            primary=bool(r.get("primary", False)),
        )
        for r in rows
    ]


def sevir_events(raw: Path) -> list[tuple[str, datetime]]:
    cat = pd.read_csv(raw / "sevir" / "CATALOG.csv", low_memory=False)
    cat = cat[cat["id"].astype(str).str.startswith("S") & cat["img_type"].isin(["vil", "ir107"])]
    t = pd.to_datetime(cat["time_utc"], utc=True)
    cat = cat[t.dt.year.isin([2018, 2019])]
    types = cat.groupby("id")["img_type"].agg(set)
    keep = set(types[types.apply(lambda s: {"vil", "ir107"} <= s)].index)
    vil = cat[(cat["img_type"] == "vil") & cat["id"].isin(keep)].drop_duplicates("id")
    out = []
    for eid, tu, offs in zip(vil["id"], vil["time_utc"], vil["minute_offsets"], strict=True):
        first = int(str(offs).split(":")[0])
        start = pd.Timestamp(tu, tz="UTC").to_pydatetime() + timedelta(minutes=first)
        out.append((str(eid), start))
    return out


def tornet_samples(raw: Path, years: tuple[int, ...] = (2015, 2016, 2019)) -> list[tuple[str, datetime, str]]:
    cat = pd.read_csv(raw / "tornet_catalog" / "catalog.csv")
    t = pd.to_datetime(cat["start_time"], utc=True)
    cat, t = cat[t.dt.year.isin(years)], t[t.dt.year.isin(years)]
    official = cat["type"].astype(str).str.strip().str.lower()
    return [(str(f), ts.to_pydatetime(), o) for f, ts, o in zip(cat["filename"], t, official, strict=True)]


def _tfrecord_count(stream) -> int:
    n = 0
    while True:
        head = stream.read(12)
        if not head:
            return n
        (length,) = struct.unpack("<Q", head[:8])
        remaining = length + 4
        while remaining:
            chunk = stream.read(min(remaining, 1 << 24))
            if not chunk:
                raise ValueError("truncated TFRecord")
            remaining -= len(chunk)
        n += 1


def ndws_ids(raw: Path) -> dict[str, list[str]]:
    pat = re.compile(r"next_day_wildfire_spread_(train|eval|test)_(\d+)\.tfrecord$")
    out: dict[str, list[str]] = {"train": [], "eval": [], "test": []}
    with zipfile.ZipFile(raw / "ndws" / "next-day-wildfire-spread.zip") as z:
        for name in sorted(z.namelist()):
            m = pat.search(name)
            if not m:
                continue
            with z.open(name) as f:
                n = _tfrecord_count(f)
            out[m.group(1)] += [f"{m.group(1)}_{m.group(2)}:{k:05d}" for k in range(n)]
    return out


def wsts_fires(raw: Path) -> list[tuple[str, date]]:
    pat = re.compile(r"(?:^|/)(\d{4})/(fire_[^/]+)/(\d{4}-\d{2}-\d{2})\.tif$")
    first: dict[str, date] = {}
    with zipfile.ZipFile(raw / "wsts" / "WildfireSpreadTS.zip") as z:
        for name in z.namelist():
            m = pat.search(name)
            if m:
                d = date.fromisoformat(m.group(3))
                first[m.group(2)] = min(first.get(m.group(2), d), d)
    return sorted(first.items())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=None, help="RAW root (default: OUTRUN_RAW or /Volumes/OutrunRaw)")
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--cases", type=Path, default=None, help="case registry (default: <out>/cases.json)")
    a = ap.parse_args(argv)
    if a.raw is None:
        a.raw = get_paths().raw
    cases = load_cases(a.cases or a.out / "cases.json")
    q = load_quarantine(write_quarantine(a.out / "quarantine.json", case_quarantine(cases)))

    splits: dict[str, Split] = {}
    splits["sevir"] = sevir_split(sevir_events(a.raw), quarantine=q)
    splits["tornet"] = tornet_split(tornet_samples(a.raw), quarantine=q, val_frac=0.1, seed=0)
    wf = wildfire_split(ndws_ids(a.raw), wsts_fires(a.raw), quarantine=q)
    splits["ndws"] = Split(
        "ndws", wf.train, wf.val, wf.test, dict(wf.extra), {"source": "NDWS official files"}
    )
    w21 = wf.extra["test_wsts2021"]
    splits["wsts"] = Split(
        "wsts", (), (), w21, {"case_holdout": tuple(i for i in w21 if i.day in q)}, {"out_of_time_year": 2021}
    )
    for name, s in splits.items():
        assert_valid_split(s)
        write_split(s, a.out / f"{name}.json")
        ex = {k: len(v) for k, v in s.extra.items()}
        print(f"{name}: train={len(s.train)} val={len(s.val)} test={len(s.test)} extra={ex}")
    write_hashes(a.out)
    print(
        "quarantine:", sorted(d.isoformat() for d in q)[0], "..", sorted(d.isoformat() for d in q)[-1], len(q)
    )
    print("frozen at", datetime.now(UTC).isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
