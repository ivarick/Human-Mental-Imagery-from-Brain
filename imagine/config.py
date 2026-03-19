"""Central configuration for the IMAGINE research pipeline."""

from __future__ import annotations

from pathlib import Path

CLASS_NAMES = [
    "apple",
    "bicycle",
    "brush",
    "cake",
    "clown",
    "cup",
    "desk",
    "foot",
    "mountain",
    "zebra",
]
N_CLASSES = len(CLASS_NAMES)

# Branch C-LDA parameters retained from the earlier validated configuration.
BANDS = [("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30)]
LOC_WINS = [(0.15, 0.8), (0.2, 0.8), (0.3, 0.8)]
IMG_WINS = [(1.0, 3.5), (1.0, 4.5), (1.5, 4.0), (0.5, 4.0), (1.0, 4.0)]
CH_TYPES = ["grad", "mag"]

# Focused windows derived from prior exploratory diagnostics and kept fixed here.
PEAK_TIERS = [
    (0.100, 0.250, 1.4, 2.8, 1, "N170"),
    (0.300, 0.420, 1.4, 2.8, 1, "P300"),
    (0.100, 0.420, 0.6, 1.5, 2, "early"),
]

DEFAULT_OUTPUT_NAME = "predictions.csv"
DEFAULT_LABELS_NAME = "labels_imagine-train.csv"
DEFAULT_SAMPLE_SUBMISSION_NAME = "sample_submission.csv"


def default_data_root() -> Path:
    """Use the current working directory so the local folder layout still works."""
    return Path.cwd()
