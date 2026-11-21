"""T15: freeze ``splits/ice.json`` (SPLIT-05) from the bag table, before any baseline or model is fit.

``.venv/bin/python -m outrun.hazards.ice.freeze [--bags DIR] [--splits splits]``. Refuses to overwrite an existing
``ice.json``. Universe: every positive bag plus a deterministic 1-in-``NEG_KEEP`` sample of the negative bags
(``sha1(id) % NEG_KEEP == 0``), because the full table (about 2.07 M bags) would make a multi-hundred-MB JSON.
The sample depends only on the bag id, never on labels beyond positive/negative, and is the same for every split.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from outrun.config import get_paths
from outrun.eval.splits import assert_valid_split, ice_split, load_quarantine, write_hashes, write_split

NEG_KEEP = 20


def bag_id(zone_id: str, window_start: pd.Timestamp) -> str:
    return f"{zone_id}|{pd.Timestamp(window_start).isoformat()}"


def keep_negative(bid: str) -> bool:
    return int(hashlib.sha1(bid.encode()).hexdigest(), 16) % NEG_KEEP == 0


def universe(bags: pd.DataFrame) -> pd.DataFrame:
    ids = [bag_id(z, t) for z, t in zip(bags["zone_id"], bags["window_start"], strict=True)]
    keep = (bags["label"].to_numpy() == 1) | pd.Series([keep_negative(i) for i in ids]).to_numpy()
    return bags[keep].reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bags", type=Path, default=get_paths().proc / "hrrr" / "ice" / "bags")
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    a = ap.parse_args(argv)
    out = a.splits / "ice.json"
    if out.exists():
        raise SystemExit(f"{out} already frozen; refusing to overwrite")
    bags = universe(pd.read_parquet(a.bags / "bags.parquet"))
    q = load_quarantine(a.splits / "quarantine.json")
    pairs = [(z, t.to_pydatetime()) for z, t in zip(bags["zone_id"], bags["window_start"], strict=True)]
    split = ice_split(pairs, quarantine=q)
    split = type(split)(
        split.dataset,
        split.train,
        split.val,
        split.test,
        dict(split.extra),
        {**split.meta, "neg_keep_1_in": NEG_KEEP, "universe": len(pairs)},
    )
    assert_valid_split(split)
    write_split(split, out)
    write_hashes(a.splits)
    print(
        "ice split:",
        {k: len(getattr(split, k)) for k in ("train", "val", "test")},
        "case_holdout",
        len(split.extra["case_holdout"]),
    )


if __name__ == "__main__":
    main()
