"""Command-line interface for the IMAGINE research pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .data import build_label_encoder, load_labels, resolve_paths, write_predictions
from .pipeline import evaluate_train, predict_test, summarize_train_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research-oriented IMAGINE MEG decoding pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("evaluate-train", "predict-test", "run-all"):
        cmd = subparsers.add_parser(command_name)
        cmd.add_argument(
            "--data-root",
            type=Path,
            default=None,
            help="Path containing train/, test/, and label files. Defaults to the current directory.",
        )
        cmd.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Prediction CSV output path. Defaults to <data-root>/predictions.csv.",
        )

    return parser


def print_banner():
    print("=" * 70)
    print("IMAGINE - Research Pipeline")
    print("  1. C-LDA:  v23-derived cross-domain ensemble")
    print("  2. Peak:   averaged prototype, focused temporal tiers, grad+mag")
    print("  Tiers: N170[100-250ms], P300[300-420ms], early[100-420ms]")
    print("  Motivation: sensory-to-imagery transfer with sparse temporal evidence")
    print("  Ensemble: 60% LDA + 40% Peak, plain argmax")
    print("=" * 70)


def run_evaluate_train(data_root: Path | None = None, output: Path | None = None):
    del output
    np.random.seed(42)
    paths = resolve_paths(data_root=data_root)
    labels_df = load_labels(paths.labels_file)
    le = build_label_encoder()
    results = evaluate_train(paths.train_dir, labels_df, le)
    summarize_train_results(results)
    return 0


def run_predict_test(data_root: Path | None = None, output: Path | None = None):
    np.random.seed(42)
    paths = resolve_paths(data_root=data_root, output=output)
    le = build_label_encoder()
    results = predict_test(paths.test_dir, le)
    print("\n>>> Phase 3: Writing predictions")
    sub_df = write_predictions(results, paths.output_file, paths.sample_submission_file)
    print(f"  Saved {len(sub_df)} rows to {paths.output_file}")
    return 0


def run_all(data_root: Path | None = None, output: Path | None = None):
    t_global = time.time()
    np.random.seed(42)
    paths = resolve_paths(data_root=data_root, output=output)
    labels_df = load_labels(paths.labels_file)
    le = build_label_encoder()

    train_results = evaluate_train(paths.train_dir, labels_df, le)
    avg = summarize_train_results(train_results)
    test_results = predict_test(paths.test_dir, le, labels_df=labels_df)

    print("\n>>> Phase 3: Writing predictions")
    sub_df = write_predictions(test_results, paths.output_file, paths.sample_submission_file)
    print(f"  Saved {len(sub_df)} rows to {paths.output_file}")
    dt = time.time() - t_global

    total_correct = sum(result.n_correct or 0 for result in train_results)
    total_trials = sum(result.n_total or 0 for result in train_results)
    print(f"\n>>> Total: {dt / 60:.1f} min | Train: {avg:.4f} ({total_correct}/{total_trials})")
    return 0


def main(argv: list[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    print_banner()

    if args.command == "evaluate-train":
        return run_evaluate_train(args.data_root, args.output)
    if args.command == "predict-test":
        return run_predict_test(args.data_root, args.output)
    if args.command == "run-all":
        return run_all(args.data_root, args.output)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
