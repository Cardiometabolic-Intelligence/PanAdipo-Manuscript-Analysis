"""Report-derived labels.

Training label (Methods, "Development cohort and label definition"): a study is positive for a prominent epicardial
fat depot if the Findings or Conclusions field of the final echocardiography report contains the phrase
"prominent epicardial" (case-insensitive; this covers "prominent epicardial adipose tissue noted" and
"prominent epicardial fat [depot]"). There are no negation rules; studies without the phrase, or without report text,
are negative. In the test set the same rule is applied to every available report text (positive if any source
matches).

Pericardial effusion (Supplemental Table 6) is parsed from the same fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FAT_PATTERN = "prominent epicardial adipose tissue noted|prominent epicardial"
EFFUSION_NONE = "no significant pericardial effusion|no evidence of pericardial effusion|trivial pericardial effusion"
EFFUSION_PRESENT = ("small pericardial effusion|circumferential pericardial effusion|moderate pericardial effusion|"
                    "large pericardial effusion")


def _any_match(texts: list[pd.Series], pattern: str) -> pd.Series:
    hit = pd.Series(False, index=texts[0].index)
    for t in texts:
        hit |= t.astype("string").str.contains(pattern, case=False, regex=True, na=False)
    return hit


def prominent_epicardial_fat(*texts: pd.Series) -> pd.Series:
    """1 if any of the given report-text fields contains the label phrase, else 0."""
    return _any_match(list(texts), FAT_PATTERN).astype(int)


def pericardial_effusion(*texts: pd.Series) -> pd.Series:
    """'none_trace', 'present', or missing (not mentioned). 'none_trace' takes precedence if both match."""
    texts = list(texts)
    none, present = _any_match(texts, EFFUSION_NONE), _any_match(texts, EFFUSION_PRESENT)
    return pd.Series(np.where(none, "none_trace", np.where(present, "present", None)), index=texts[0].index)
