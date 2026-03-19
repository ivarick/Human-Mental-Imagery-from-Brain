from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from imagine.branches.c_lda import _lda_perdomain, _lda_shared
from imagine.branches.peak import _prototype_scan
from imagine.config import CLASS_NAMES, N_CLASSES
from imagine.data import DatasetPaths, build_label_encoder, resolve_paths, write_predictions
from imagine.pipeline import SubjectResult, process_subject


FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent


def dataset_available() -> bool:
    return (REPO_ROOT / "train").exists() and (REPO_ROOT / "test").exists() and (
        REPO_ROOT / "labels_imagine-train.csv"
    ).exists()


class UnitTests(unittest.TestCase):
    def test_label_encoder_order(self):
        encoder = build_label_encoder()
        self.assertEqual(encoder.classes_.tolist(), CLASS_NAMES)

    def test_resolve_paths_uses_expected_layout(self):
        root = Path(r"C:\tmp\imagine-data")
        paths = resolve_paths(root, root / "custom_predictions.csv")
        self.assertEqual(
            paths,
            DatasetPaths(
                data_root=root.resolve(),
                train_dir=(root / "train").resolve(),
                test_dir=(root / "test").resolve(),
                labels_file=(root / "labels_imagine-train.csv").resolve(),
                sample_submission_file=(root / "sample_submission.csv").resolve(),
                output_file=(root / "custom_predictions.csv").resolve(),
            ),
        )

    def test_prediction_writer_without_sample_submission(self):
        result = SubjectResult(
            subject_id="sub-99",
            predictions=["apple", "zebra"],
            trial_ordinals=[1, 2],
            elapsed_seconds=0.1,
            best_pair=(0.0, 0.0, 0.0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "predictions.csv"
            df = write_predictions([result], output_file, sample_submission_file=Path(tmpdir) / "missing.csv")
            self.assertEqual(df.to_dict("records"), [{"ID": "sub-99_1", "label": "apple"}, {"ID": "sub-99_2", "label": "zebra"}])
            self.assertTrue(output_file.exists())

    def test_lda_probability_shapes_and_normalization(self):
        rng = np.random.default_rng(42)
        X_loc = rng.normal(size=(20, 8))
        X_img = rng.normal(size=(6, 8))
        y_loc = np.repeat(np.arange(N_CLASSES), 2)

        for fn in (_lda_perdomain, _lda_shared):
            proba = fn(X_loc, y_loc, X_img, N_CLASSES)
            self.assertEqual(proba.shape, (6, N_CLASSES))
            np.testing.assert_allclose(proba.sum(axis=1), np.ones(6), atol=1e-6)

    def test_peak_scan_probability_shapes_and_vote_count(self):
        rng = np.random.default_rng(123)
        loc_x = rng.normal(size=(10, 4, 5))
        img_x = rng.normal(size=(3, 4, 5))
        y_loc = np.arange(N_CLASSES)
        loc_times = np.array([0.10, 0.15, 0.20, 0.35, 0.40])
        img_times = np.array([0.70, 1.40, 1.50, 2.00, 2.80])

        sum_P, n_votes, _best_lt, _best_it, _best_acc = _prototype_scan(
            loc_x, img_x, y_loc, loc_times, img_times
        )
        self.assertEqual(sum_P.shape, (3, N_CLASSES))
        self.assertGreater(n_votes, 0)
        np.testing.assert_allclose(sum_P.sum(axis=1), np.full(3, n_votes), atol=1e-6)


@unittest.skipUnless(dataset_available(), "Local IMAGINE dataset not available")
class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd

        cls.labels = pd.read_csv(REPO_ROOT / "labels_imagine-train.csv")
        cls.encoder = build_label_encoder()

    def test_train_subject_matches_recorded_reference_output(self):
        expected = json.loads((FIXTURES_DIR / "train_sub02_expected.json").read_text())
        result = process_subject("sub-02", REPO_ROOT / "train", self.labels, self.encoder, is_train=True)
        self.assertEqual(result.subject_id, expected["subject_id"])
        self.assertEqual(result.predictions, expected["predictions"])
        self.assertAlmostEqual(result.accuracy, expected["accuracy"], places=12)
        self.assertEqual(result.n_total, expected["n_total"])
        for actual, recorded in zip(result.best_pair, expected["best_pair"]):
            self.assertAlmostEqual(float(actual), recorded, places=12)

    def test_test_subject_rows_match_recorded_reference_output(self):
        expected = json.loads((FIXTURES_DIR / "test_sub01_expected.json").read_text())
        result = process_subject("sub-01", REPO_ROOT / "test", self.labels, self.encoder, is_train=False)
        rows = [
            {"ID": f"{result.subject_id}_{trial_idx}", "label": label}
            for trial_idx, label in zip(result.trial_ordinals, result.predictions)
        ]
        self.assertEqual(result.subject_id, expected["subject_id"])
        self.assertEqual(rows, expected["rows"])


class SmokeTests(unittest.TestCase):
    def test_cli_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "imagine.cli", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("evaluate-train", proc.stdout)


if __name__ == "__main__":
    unittest.main()
