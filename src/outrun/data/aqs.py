"""EPA AQS hourly PM2.5 (88101) reader (specs/01-data.md DATA-07). Output feeds SMOKE-01."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PM25_PARAMETER = 88101


def read_aqs_hourly(path: Path | str) -> pd.DataFrame:
    """DATA-07. zip or CSV -> site_id, time (UTC), pm25 (>= 0, POCs averaged), lat, lon; sorted."""
    raise NotImplementedError
