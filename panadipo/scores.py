"""Study-level PanAdipo scores from clip scores.

Study score (used throughout): the mean of all clip scores of a study (view-agnostic). In POCUS, only clips classified
by the PanEcho view classifier as on-axis PLAX, mid-papillary PSAX, or A4C are averaged. View-specific scores average
the standard on-axis clips of each view.
"""

from __future__ import annotations

import pandas as pd


def study_scores(clips: pd.DataFrame, study: str = "study_id", views: list[str] | None = None,
                 view_col: str = "view") -> pd.Series:
    """Mean clip score per study, optionally restricted to the listed view-classifier classes."""
    c = clips if views is None else clips[clips[view_col].isin(views)]
    return c.dropna(subset=["score"]).groupby(study)["score"].mean()


def view_scores(clips: pd.DataFrame, view_groups: dict[str, list[str]], study: str = "study_id",
                view_col: str = "view") -> pd.DataFrame:
    """One column per view group: the mean score of that view's clips within each study."""
    return pd.DataFrame({g: study_scores(clips, study, members, view_col) for g, members in view_groups.items()})
