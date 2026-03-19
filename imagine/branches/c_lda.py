"""Cross-domain LDA branch retained from the earlier ensemble formulation."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from ..config import BANDS, CH_TYPES, IMG_WINS, LOC_WINS, N_CLASSES

warnings.filterwarnings("ignore")


def _filter_copy(epochs, fmin, fmax):
    filtered = epochs.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered.filter(fmin, fmax, verbose=False)
    return filtered


def _bandvar_cached(ep_filtered, tmin, tmax):
    ep_c = ep_filtered.copy().crop(
        tmin=max(tmin, ep_filtered.tmin),
        tmax=min(tmax, ep_filtered.tmax),
    )
    return ep_c.get_data().var(axis=-1)


def _psd_features(data, sfreq=100.0):
    n_trials, n_channels, n_times = data.shape
    feats = np.zeros((n_trials, n_channels * 3))
    nperseg = min(n_times, 128)
    for idx in range(n_trials):
        freqs, psd = welch(data[idx], fs=sfreq, nperseg=nperseg, axis=-1)
        parts = []
        for _band_name, lo, hi in BANDS:
            band_mask = (freqs >= lo) & (freqs <= hi)
            parts.append(psd[:, band_mask].mean(axis=-1))
        feats[idx] = np.concatenate(parts)
    return feats


def _lda_perdomain(X_loc, y_loc, X_img, n_classes):
    """Scale localizer and imagery domains independently before LDA transfer."""
    sc_loc = StandardScaler().fit(X_loc)
    sc_img = StandardScaler().fit(X_img)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(sc_loc.transform(X_loc), y_loc)
    dec = clf.decision_function(sc_img.transform(X_img))
    e = np.exp(dec - dec.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    full = np.zeros((len(p), n_classes))
    for i, cls in enumerate(clf.classes_):
        full[:, cls] = p[:, i]
    return full


def _lda_shared(X_loc, y_loc, X_img, n_classes):
    """Use localizer statistics to normalize both domains when transfer is stable."""
    sc = StandardScaler().fit(X_loc)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(sc.transform(X_loc), y_loc)
    dec = clf.decision_function(sc.transform(X_img))
    e = np.exp(dec - dec.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    full = np.zeros((len(p), n_classes))
    for i, cls in enumerate(clf.classes_):
        full[:, cls] = p[:, i]
    return full


def branch_c_lda(ep_loc_clean, ep_img_clean, y_loc):
    """Aggregate multiple LDA variants over channel types, bands, and windows."""

    sum_proba = None
    n_votes = 0

    cache_loc = {}
    cache_img = {}
    for ch in CH_TYPES:
        ep_l_ch = ep_loc_clean.copy().pick_types(meg=ch)
        ep_i_ch = ep_img_clean.copy().pick_types(meg=ch)
        for band_name, fmin, fmax in BANDS:
            key = (ch, band_name)
            cache_loc[key] = _filter_copy(ep_l_ch, fmin, fmax)
            cache_img[key] = _filter_copy(ep_i_ch, fmin, fmax)

    for ch in CH_TYPES:
        for band_name, _fmin, _fmax in BANDS:
            key = (ch, band_name)
            ep_lf = cache_loc[key]
            ep_if = cache_img[key]
            for lt0, lt1 in LOC_WINS:
                fl = _bandvar_cached(ep_lf, lt0, lt1)
                for w0, w1 in IMG_WINS:
                    at0 = max(w0, ep_if.tmin)
                    at1 = min(w1, ep_if.tmax)
                    if at1 - at0 < 0.3:
                        continue
                    try:
                        fi = _bandvar_cached(ep_if, at0, at1)
                        p = _lda_perdomain(fl, y_loc, fi, N_CLASSES)
                        sum_proba = p if sum_proba is None else sum_proba + p
                        n_votes += 1
                    except Exception:
                        continue

    for ch in CH_TYPES:
        for band_name, _fmin, _fmax in BANDS:
            key = (ch, band_name)
            ep_lf = cache_loc[key]
            ep_if = cache_img[key]
            fl = _bandvar_cached(ep_lf, LOC_WINS[0][0], LOC_WINS[0][1])
            for w0, w1 in IMG_WINS[:2]:
                at0 = max(w0, ep_if.tmin)
                at1 = min(w1, ep_if.tmax)
                if at1 - at0 < 0.3:
                    continue
                try:
                    fi = _bandvar_cached(ep_if, at0, at1)
                    p = _lda_shared(fl, y_loc, fi, N_CLASSES)
                    sum_proba = p if sum_proba is None else sum_proba + p
                    n_votes += 1
                except Exception:
                    continue

    ep_l_psd = _filter_copy(ep_loc_clean.copy().pick_types(meg="grad"), 0.5, 45)
    ep_i_psd = _filter_copy(ep_img_clean.copy().pick_types(meg="grad"), 0.5, 45)
    sfreq = ep_l_psd.info["sfreq"]
    for lt0, lt1 in LOC_WINS[:2]:
        tl0 = max(lt0, ep_l_psd.tmin)
        tl1 = min(lt1, ep_l_psd.tmax)
        if tl1 <= tl0:
            continue
        X_l = ep_l_psd.copy().crop(tl0, tl1).get_data()
        fl_psd = _psd_features(X_l, sfreq=sfreq)
        for w0, w1 in IMG_WINS[:3]:
            at0 = max(w0, ep_i_psd.tmin)
            at1 = min(w1, ep_i_psd.tmax)
            if at1 - at0 < 0.3:
                continue
            try:
                X_i = ep_i_psd.copy().crop(at0, at1).get_data()
                fi_psd = _psd_features(X_i, sfreq=sfreq)
                p1 = _lda_perdomain(fl_psd, y_loc, fi_psd, N_CLASSES)
                p2 = _lda_shared(fl_psd, y_loc, fi_psd, N_CLASSES)
                sum_proba = (sum_proba + p1 + p2) if sum_proba is not None else (p1 + p2)
                n_votes += 2
            except Exception:
                continue

    if sum_proba is None or n_votes == 0:
        return None
    return sum_proba / n_votes
