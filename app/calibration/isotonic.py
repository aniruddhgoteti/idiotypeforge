"""IsotonicRegression confidence wrapper.

Adapted from /Users/aniruddhgoteti/workspace/stasis-1/legacy/training/calibration.py.

Trains a monotonic mapping from raw ipLDDT scores to a calibrated probability
of "experimentally validated binder", using the labelled subset of RFdiffusion
/ BindCraft published designs (Bennett 2024, Cao 2024).

Pickled fit lives at `app/calibration/calibration_fit.pkl` (≈ 1 KB).

Usage:
    >>> from app.calibration.isotonic import calibrate
    >>> calibrated = calibrate(iplddt=0.86)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from sklearn.isotonic import IsotonicRegression

CALIB_PATH = Path(__file__).parent / "calibration_fit.pkl"


def fit(
    iplddts: Iterable[float],
    labels: Iterable[int],
    out_path: Path | None = None,
) -> IsotonicRegression:
    """Fit the calibrator on (raw_iplddt, validated_binary) pairs."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(list(iplddts), list(labels))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            pickle.dump(iso, fh)
    return iso


def load(path: Path = CALIB_PATH) -> IsotonicRegression | None:
    """Load the trained calibrator. Returns None if file missing."""
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def calibrate(iplddt: float, fit_obj: IsotonicRegression | None = None) -> float:
    """Map raw ipLDDT to calibrated P(validated binder).

    If no calibrator fit exists, falls back to identity (returns the input
    clamped to [0, 1]).
    """
    if fit_obj is None:
        fit_obj = load()
    if fit_obj is None:
        return max(0.0, min(1.0, iplddt))
    return float(fit_obj.predict([iplddt])[0])
