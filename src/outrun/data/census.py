"""2020 census block-group population (specs/01-data.md DATA-19)."""

from __future__ import annotations

import pandas as pd


def block_group_population(table: pd.DataFrame, pop_col: str = "P1_001N") -> pd.DataFrame:
    """DATA-19. -> geoid (12-char str), population (int64); bad/duplicate rows -> ValueError."""
    raise NotImplementedError
