"""Smoke data-side helpers added in T18 (HRRR wind rotation, vectorised features). Synthetic, offline."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest


@pytest.mark.req("SMOKE-04")
def test_rotate_winds_matches_hrrr_lambert():
    from pyproj import CRS, Transformer

    from outrun.hazards.smoke.winds import rotate_winds

    crs = CRS.from_proj4(
        "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +a=6371229 +b=6371229 +units=m"
    )
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    for lon, lat in [(-120.0, 40.0), (-97.5, 38.0), (-75.0, 42.0)]:
        x0, y0 = tr.transform(lon, lat)
        xe, ye = tr.transform(lon + 1e-4, lat)
        xn, yn = tr.transform(lon, lat + 1e-4)
        east = np.array([xe - x0, ye - y0]) / np.hypot(xe - x0, ye - y0)
        north = np.array([xn - x0, yn - y0]) / np.hypot(xn - x0, yn - y0)
        rng = np.random.default_rng(0)
        u, v = rng.normal(size=2)
        w = u * np.array([1.0, 0.0]) + v * np.array([0.0, 1.0])  # grid-relative wind vector
        ue, vn = rotate_winds(np.array(u), np.array(v), np.array(lon))
        assert float(ue) == pytest.approx(w @ east, abs=1e-5) and float(vn) == pytest.approx(
            w @ north, abs=1e-5
        )
        assert float(np.hypot(ue, vn)) == pytest.approx(float(np.hypot(u, v)))
    ue, vn = rotate_winds(np.array(1.0), np.array(0.0), np.array(-97.5))
    assert (float(ue), float(vn)) == pytest.approx((1.0, 0.0))
    _, vn_west = rotate_winds(np.array(1.0), np.array(0.0), np.array(-120.0))
    assert float(vn_west) > 0  # far west the grid +x axis points north of east


@pytest.mark.req("SMOKE-02")
def test_tabular_features_batch_matches_scalar():
    from outrun.hazards.smoke.dataset import SmokeFeatureConfig, tabular_features, tabular_features_batch

    rng = np.random.default_rng(1)
    hist = rng.random((5, 6)) * 40
    hist[0, 2] = np.nan
    hist[1] = np.nan
    hist[2, 0] = -3.0
    hms = np.array([0, 1, 2, 3, 0], float)
    times = [datetime(2023, 7, 4, h, tzinfo=UTC) for h in (0, 6, 12, 18, 6)]
    for use_hms in (True, False):
        cfg = SmokeFeatureConfig(use_hms=use_hms)
        batch = tabular_features_batch(hist, hms, pd.DatetimeIndex(times), cfg)
        for i in range(5):
            np.testing.assert_allclose(batch[i], tabular_features(hist[i], hms[i], times[i], cfg), atol=1e-6)
