# NOTES: running log

Log anything you'd want in the report while it's fresh: problems, fixes, decisions and why, numbers.
At least one dated row every day from Nov 17 to Dec 4.

| Date | Task | What happened / decided | Why / numbers |
|---|---|---|---|
| 2026-11-17 | T0 | Scaffold: pyproject, Makefile, CI, CLAUDE.md, TASKS.md, configs, manifest skeleton, specs 00-16 with req IDs, stubs, red tests, traceability + proposal-matrix tests | Python 3.12 venv; `pip install -e .[dev]` clean, no brew needed (cfgrib uses the bundled eccodes 2.50 wheel). Got torch 2.15.0 (MPS available), numpy 2.5, pandas 3.1, xarray 2026.9, **zarr 3.4** (v3 API), osmnx 2.1, ortools 9.15, geopandas 1.2, rasterio 1.5, herbie 2026.10. Several are a major version newer than DESIGN assumed (zarr 3, pandas 3, osmnx 2): watch for API changes. pySTEPS and OpenCV (DESIGN 2.3 storm baseline) are not yet dependencies; add in T8 if used. Traceability: 7 checks pass; `test_all_spec_files_exist` and `test_proposal_matrix` stay red until specs 01-16 exist (expected) |
