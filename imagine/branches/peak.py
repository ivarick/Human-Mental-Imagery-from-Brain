"""Time-localized prototype branch for the IMAGINE ensemble."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler

from ..config import N_CLASSES, PEAK_TIERS

warnings.filterwarnings("ignore")


def _prototype_scan(loc_x, img_x, y_loc, loc_times, img_times, y_img=None):
    """Scan focused sensory and imagery windows with class prototypes.

    Localizer trials are averaged into class templates at each timepoint.
    Templates are then compared with imagery snapshots using cosine distance.
    The narrow windows reflect the empirical expectation that imagery evidence is
    sparse in time rather than uniformly distributed across the trial.
    """

    n_img = img_x.shape[0]
    n_loc_t = loc_x.shape[2]
    n_img_t = img_x.shape[2]

    prototypes = np.array([loc_x[y_loc == c].mean(axis=0) for c in range(N_CLASSES)])

    sum_P = np.zeros((n_img, N_CLASSES))
    n_votes = 0
    best_acc = 0.0
    best_lt = 0.0
    best_it = 0.0

    for loc_lo, loc_hi, img_lo, img_hi, img_step, _label in PEAK_TIERS:
        loc_mask = (loc_times >= loc_lo) & (loc_times <= loc_hi)
        loc_idx = np.where(loc_mask)[0]
        img_mask = (img_times >= img_lo) & (img_times <= img_hi)
        img_idx = np.where(img_mask)[0]

        for lt in loc_idx:
            lt_s = max(0, lt - 1)
            lt_e = min(n_loc_t, lt + 2)
            proto_t = prototypes[:, :, lt_s:lt_e].mean(axis=-1)

            loc_snap = loc_x[:, :, lt_s:lt_e].mean(axis=-1)
            sc_loc = StandardScaler().fit(loc_snap)
            proto_scaled = sc_loc.transform(proto_t)

            for it in img_idx[::img_step]:
                it_s = max(0, it - 1)
                it_e = min(n_img_t, it + 2)
                img_snap = img_x[:, :, it_s:it_e].mean(axis=-1)

                sc_img = StandardScaler().fit(img_snap)
                img_t_scaled = sc_img.transform(img_snap)

                D = cdist(img_t_scaled, proto_scaled, metric="cosine")
                P = softmax(-D / (D.mean() + 1e-10), axis=1)
                sum_P += P
                n_votes += 1

                if y_img is not None:
                    preds = P.argmax(axis=1)
                    acc = (preds == y_img).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_lt = loc_times[lt]
                        best_it = img_times[it]

    return sum_P, n_votes, best_lt, best_it, best_acc


def branch_peak_narrowed(ep_loc_clean, ep_img_clean, y_loc, y_img=None):
    """Average prototype evidence over gradiometers and magnetometers.

    Gradiometers emphasize field gradients, while magnetometers preserve the
    absolute magnetic field. The original ensemble treated them as
    complementary views of the same representational dynamics.
    """

    n_img = len(ep_img_clean)
    sum_P_total = np.zeros((n_img, N_CLASSES))
    n_votes_total = 0
    best_acc_global = 0.0
    best_lt_global = 0.0
    best_it_global = 0.0

    for meg_type in ["grad", "mag"]:
        ep_l = ep_loc_clean.copy().pick_types(meg=meg_type)
        ep_i = ep_img_clean.copy().pick_types(meg=meg_type)

        loc_x = ep_l.get_data()
        img_x = ep_i.get_data()
        loc_times = ep_l.times
        img_times = ep_i.times

        sum_P, n_votes, b_lt, b_it, b_acc = _prototype_scan(
            loc_x, img_x, y_loc, loc_times, img_times, y_img=y_img
        )

        sum_P_total += sum_P
        n_votes_total += n_votes

        if b_acc > best_acc_global:
            best_acc_global = b_acc
            best_lt_global = b_lt
            best_it_global = b_it

    if y_img is not None and best_acc_global > 0:
        print(
            f"      [Peak] best pair: loc_t={best_lt_global:.3f}s, "
            f"img_t={best_it_global:.3f}s, acc={best_acc_global:.3f}"
        )

    if n_votes_total == 0:
        return None, (0, 0, 0)
    return sum_P_total / n_votes_total, (best_lt_global, best_it_global, best_acc_global)
