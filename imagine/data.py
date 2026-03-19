"""Dataset and prediction-file helpers for the IMAGINE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import (
    CLASS_NAMES,
    DEFAULT_LABELS_NAME,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_SAMPLE_SUBMISSION_NAME,
)


@dataclass(frozen=True)
class DatasetPaths:
    """Filesystem contract for the local IMAGINE dataset."""

    data_root: Path
    train_dir: Path
    test_dir: Path
    labels_file: Path
    sample_submission_file: Path
    output_file: Path


def resolve_paths(data_root: Path | None = None, output: Path | None = None) -> DatasetPaths:
    root = Path.cwd() if data_root is None else Path(data_root).resolve()
    output_file = Path(output).resolve() if output is not None else root / DEFAULT_OUTPUT_NAME
    return DatasetPaths(
        data_root=root,
        train_dir=root / "train",
        test_dir=root / "test",
        labels_file=root / DEFAULT_LABELS_NAME,
        sample_submission_file=root / DEFAULT_SAMPLE_SUBMISSION_NAME,
        output_file=output_file,
    )


def build_label_encoder() -> LabelEncoder:
    """Freeze the class order used by both training diagnostics and inference."""
    return LabelEncoder().fit(CLASS_NAMES)


def load_labels(labels_file: Path) -> pd.DataFrame:
    return pd.read_csv(labels_file)


def list_subjects(subject_dir: Path) -> list[str]:
    return sorted(path.name for path in subject_dir.iterdir() if path.is_dir())


def build_prediction_rows(results) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result in results:
        for trial_idx, label in zip(result.trial_ordinals, result.predictions):
            rows.append({"ID": f"{result.subject_id}_{trial_idx}", "label": label})
    return rows


def write_predictions(results, output_file: Path, sample_submission_file: Path | None = None) -> pd.DataFrame:
    """Write subject-level predictions in the expected CSV format."""
    sub_df = pd.DataFrame(build_prediction_rows(results))
    if sample_submission_file is not None and sample_submission_file.exists():
        sample = pd.read_csv(sample_submission_file)
        sub_df = sample[["ID"]].merge(sub_df, on="ID", how="left")
        if sub_df["label"].isna().any():
            missing = sub_df[sub_df["label"].isna()]["ID"].tolist()[:5]
            raise ValueError(f"Missing: {missing}")
    sub_df.to_csv(output_file, index=False)
    return sub_df
