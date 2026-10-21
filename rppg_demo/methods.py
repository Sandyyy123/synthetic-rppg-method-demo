"""rPPG pulse-extraction methods: GREEN baseline, CHROM and POS.

CHROM: de Haan & Jeanne 2013, IEEE TBME 60(10):2878-2886, doi:10.1109/TBME.2013.2266196
POS:   Wang, den Brinker, Stuijk & de Haan 2017, IEEE TBME 64(7):1479-1491,
       doi:10.1109/TBME.2016.2609282
Both are implemented with sliding windows and overlap-add, as in the papers.
"""
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass(x, fs, lo=0.7, hi=3.0, order=3):
    """Zero-phase Butterworth band-pass over the cardiac band (42-180 bpm)."""
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def green(rgb, fs):
    g = rgb[:, 1] / np.mean(rgb[:, 1]) - 1.0
    return bandpass(g, fs)


def chrom(rgb, fs, win_s=1.6):
    n = rgb.shape[0]
    L = int(round(win_s * fs))
    L += L % 2
    out = np.zeros(n)
    hann = np.hanning(L)
    for start in range(0, n - L + 1, L // 2):
        seg = rgb[start:start + L]
        cn = seg / np.mean(seg, axis=0)                  # temporal normalisation
        xs = 3 * cn[:, 0] - 2 * cn[:, 1]
        ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
        xf = bandpass(xs, fs) if L > 21 else xs - xs.mean()
        yf = bandpass(ys, fs) if L > 21 else ys - ys.mean()
        alpha = np.std(xf) / (np.std(yf) + 1e-12)
        s = xf - alpha * yf
        out[start:start + L] += hann * (s - s.mean())
    return bandpass(out, fs)


def pos(rgb, fs, win_s=1.6):
    n = rgb.shape[0]
    L = int(round(win_s * fs))
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])  # projection plane
    h = np.zeros(n)
    for start in range(0, n - L + 1):
        seg = rgb[start:start + L]
        cn = seg / np.mean(seg, axis=0)
        s = cn @ P.T                                      # (L, 2)
        s1, s2 = s[:, 0], s[:, 1]
        hh = s1 + (np.std(s1) / (np.std(s2) + 1e-12)) * s2
        h[start:start + L] += hh - hh.mean()              # overlap-add
    return bandpass(h, fs)


METHODS = {"GREEN": green, "CHROM": chrom, "POS": pos}
