"""Signal preprocessing shared across the IMAGINE branches."""

from __future__ import annotations

import warnings

import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


def _safe_filter_inplace(ep: mne.BaseEpochs, l_freq, h_freq) -> mne.BaseEpochs:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ep.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    return ep


def preprocess_epochs(ep: mne.BaseEpochs, is_imagery: bool = False) -> mne.BaseEpochs:
    """Apply the minimal preprocessing used in the original monolithic script.

    The localizer condition receives baseline correction because it contains a
    pre-stimulus interval that can stabilize class templates. Imagery epochs do
    not receive that baseline here because the original pipeline treated them as
    self-contained post-cue activity.
    """

    ep.pick_types(meg=True, eeg=False, eog=False, ecg=False)
    _safe_filter_inplace(ep, l_freq=None, h_freq=45.0)
    if not is_imagery and ep.tmin < 0.0:
        ep.apply_baseline((max(-0.2, ep.tmin), 0.0))
    return ep
