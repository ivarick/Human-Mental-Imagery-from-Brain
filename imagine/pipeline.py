"""Subject-level orchestration for the IMAGINE ensemble."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass

import mne
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from .branches.c_lda import branch_c_lda
from .branches.peak import branch_peak_narrowed
from .config import N_CLASSES
from .preprocessing import preprocess_epochs


@dataclass
class SubjectResult:
    subject_id: str
    predictions: list[str]
    trial_ordinals: list[int]
    elapsed_seconds: float
    best_pair: tuple[float, float, float]
    accuracy: float | None = None
    n_correct: int | None = None
    n_total: int | None = None


def process_subject(sub, data_dir, labels_df, le: LabelEncoder, is_train: bool = True) -> SubjectResult:
    """Run the full two-branch ensemble for a single subject."""

    t0 = time.time()

    ep_loc = mne.read_epochs(
        str(data_dir / sub / f"{sub}_localizer-epo.fif"),
        preload=True,
        verbose=False,
    )
    ep_img = mne.read_epochs(
        str(data_dir / sub / f"{sub}_imagine-epo.fif"),
        preload=True,
        verbose=False,
    )

    ep_loc_c = preprocess_epochs(ep_loc, is_imagery=False)
    ep_img_c = preprocess_epochs(ep_img, is_imagery=True)

    class_to_name = {value: key for key, value in ep_loc.event_id.items()}
    y_loc = le.transform([class_to_name[event_code] for event_code in ep_loc.events[:, 2]])
    n_img = len(ep_img)

    y_img = None
    if is_train:
        sub_df = labels_df[labels_df["subject"] == sub].sort_values("trial_idx")
        y_img = le.transform(sub_df["label"].values)

    proba_lda = None
    try:
        proba_lda = branch_c_lda(ep_loc_c, ep_img_c, y_loc)
    except Exception as exc:
        print(f"    [LDA] {sub} failed: {exc}")

    proba_peak = None
    best_pair = (0, 0, 0)
    try:
        proba_peak, best_pair = branch_peak_narrowed(ep_loc_c, ep_img_c, y_loc, y_img=y_img)
    except Exception as exc:
        print(f"    [Peak] {sub} failed: {exc}")

    if proba_lda is not None and proba_peak is not None:
        final = 0.60 * proba_lda + 0.40 * proba_peak
    elif proba_lda is not None:
        final = proba_lda
    elif proba_peak is not None:
        final = proba_peak
    else:
        final = np.full((n_img, N_CLASSES), 1.0 / N_CLASSES)

    preds = final.argmax(axis=1)
    preds_labels = le.inverse_transform(preds).tolist()

    result = SubjectResult(
        subject_id=sub,
        predictions=preds_labels,
        trial_ordinals=list(range(1, n_img + 1)),
        elapsed_seconds=time.time() - t0,
        best_pair=best_pair,
    )

    if is_train:
        acc = accuracy_score(y_img, preds)
        result.accuracy = acc
        result.n_correct = int((y_img == preds).sum())
        result.n_total = n_img

        parts = []
        if proba_lda is not None:
            a_lda = accuracy_score(y_img, proba_lda.argmax(1))
            parts.append(f"LDA={a_lda:.3f}")
        if proba_peak is not None:
            a_peak = accuracy_score(y_img, proba_peak.argmax(1))
            parts.append(f"Peak={a_peak:.3f}")
        parts.append(f"ens={acc:.3f}")
        print(f"  {sub}: {' | '.join(parts)} [{result.elapsed_seconds:.0f}s]")

    return result


def evaluate_train(train_dir, labels_df, le: LabelEncoder) -> list[SubjectResult]:
    results = []
    train_subs = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    print(f"\n>>> Phase 1: Train ({len(train_subs)} subjects)")
    for sub in train_subs:
        try:
            results.append(process_subject(sub, train_dir, labels_df, le, is_train=True))
        except Exception as exc:
            print(f"  {sub} FAILED: {exc}")
            traceback.print_exc()
    return results


def summarize_train_results(train_results: list[SubjectResult]) -> float:
    total_correct = sum(result.n_correct or 0 for result in train_results)
    total_trials = sum(result.n_total or 0 for result in train_results)
    avg = total_correct / total_trials if total_trials > 0 else 0.0
    print(f"\n>>> TRAIN SUMMARY: {avg:.4f} ({total_correct}/{total_trials})  [chance=0.1000]")

    best_pairs = [result.best_pair for result in train_results if result.best_pair[2] > 0]
    if best_pairs:
        mean_lt = np.mean([pair[0] for pair in best_pairs])
        mean_it = np.mean([pair[1] for pair in best_pairs])
        mean_acc = np.mean([pair[2] for pair in best_pairs])
        med_lt = np.median([pair[0] for pair in best_pairs])
        med_it = np.median([pair[1] for pair in best_pairs])
        print("\n>>> TEMPORAL SIGNAL:")
        print(f"    Mean  best loc_t={mean_lt:.3f}s, img_t={mean_it:.3f}s, acc={mean_acc:.3f}")
        print(f"    Median best loc_t={med_lt:.3f}s, img_t={med_it:.3f}s")
        print("    Per-subject best pairs:")
        for result in train_results:
            best_pair = result.best_pair
            if best_pair[2] > 0:
                print(
                    f"      {result.subject_id}: loc_t={best_pair[0]:.3f}s, "
                    f"img_t={best_pair[1]:.3f}s, acc={best_pair[2]:.3f}"
                )
    return avg


def predict_test(test_dir, le: LabelEncoder, labels_df=None) -> list[SubjectResult]:
    results = []
    test_subs = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    print(f"\n>>> Phase 2: Test ({len(test_subs)} subjects)")
    for sub in test_subs:
        t0 = time.time()
        try:
            results.append(process_subject(sub, test_dir, labels_df, le, is_train=False))
            print(f"  {sub}: done [{time.time() - t0:.0f}s]")
        except Exception as exc:
            print(f"  {sub} FAILED: {exc}")
            traceback.print_exc()
    return results
