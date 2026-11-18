"""EPA AQS hourly PM2.5 (88101) reader (specs/01-data.md DATA-07). Output feeds SMOKE-01."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PM25_PARAMETER = 88101


def read_aqs_hourly(path: Path | str) -> pd.DataFrame:
    """DATA-07. zip or CSV -> site_id, time (UTC), pm25 (>= 0, POCs averaged), lat, lon; sorted."""
    cols = ["State Code", "County Code", "Site Num", "Parameter Code", "Latitude", "Longitude", "Date GMT",
            "Time GMT", "Sample Measurement"]  # fmt: skip
    # Site codes stay strings (zero padding; State Code can be "CC" for Canada). pandas reads the CSV inside a zip.
    raw = pd.read_csv(path, usecols=cols, dtype={"State Code": str, "County Code": str, "Site Num": str})
    raw = raw[pd.to_numeric(raw["Parameter Code"], errors="coerce") == PM25_PARAMETER]

    df = pd.DataFrame(
        {
            "site_id": raw["State Code"].str.zfill(2) + "-" + raw["County Code"].str.zfill(3) + "-"
            + raw["Site Num"].str.zfill(4),
            "time": pd.to_datetime(raw["Date GMT"] + " " + raw["Time GMT"], utc=True),
            "pm25": raw["Sample Measurement"].astype(float).clip(lower=0.0),
            "lat": raw["Latitude"].astype(float),
            "lon": raw["Longitude"].astype(float),
        }
    )  # fmt: skip
    # Several instruments (POCs) at one site and hour -> one averaged row.
    out = df.groupby(["site_id", "time"], as_index=False, sort=True).agg(
        pm25=("pm25", "mean"), lat=("lat", "first"), lon=("lon", "first")
    )
    return out.reset_index(drop=True)
