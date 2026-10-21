"""Heart-rate estimation, signal-quality index and agreement statistics."""
import numpy as np


def spectral_hr(x, fs, lo=0.7, hi=3.0, pad=8):
    """HR (bpm) from the dominant FFT peak in [lo, hi] Hz, plus an SNR-style SQI.

    SQI is adapted from the de Haan & Jeanne 2013 SNR: there the template sits at the
    reference HR; here it sits at the estimated peak (blind SQI). Power within +/-0.1 Hz of
    the peak and its first harmonic, divided by the remaining in-band power (dB).
    """
    x = (x - np.mean(x)) * np.hanning(len(x))
    nfft = int(2 ** np.ceil(np.log2(len(x) * pad)))
    spec = np.abs(np.fft.rfft(x, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = (f >= lo) & (f <= hi)
    fpk = f[band][np.argmax(spec[band])]
    sig = (np.abs(f - fpk) <= 0.1) | (np.abs(f - 2 * fpk) <= 0.2)
    wide = (f >= lo) & (f <= 2 * hi)
    p_sig = spec[sig & wide].sum()
    p_noise = spec[wide & ~sig].sum() + 1e-20
    return 60.0 * fpk, 10 * np.log10(p_sig / p_noise)


def windowed_hr(x, true_hr, fs, win_s=10.0, step_s=2.0):
    """Sliding-window estimates vs window-mean ground-truth HR."""
    L, S = int(win_s * fs), int(step_s * fs)
    est, ref, sqi = [], [], []
    for a in range(0, len(x) - L + 1, S):
        h, q = spectral_hr(x[a:a + L], fs)
        est.append(h); ref.append(float(np.mean(true_hr[a:a + L]))); sqi.append(q)
    return np.array(est), np.array(ref), np.array(sqi)


def agreement(est, ref):
    """MAE, RMSE, Pearson r and Bland-Altman bias with 95% limits of agreement.

    Bland & Altman 1986, Lancet 327(8476):307-310, doi:10.1016/S0140-6736(86)90837-8.
    Note: windows from one recording are not independent; for real data the
    limits should use a repeated-measures variant (Bland & Altman 2007).
    """
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    d = est - ref
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    r = float(np.corrcoef(est, ref)[0, 1]) if np.std(est) > 0 and np.std(ref) > 0 else float("nan")
    return {"n": int(len(d)), "mae": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))), "pearson_r": r,
            "ba_bias": bias, "ba_loa_low": bias - 1.96 * sd, "ba_loa_high": bias + 1.96 * sd,
            "within_5bpm_pct": float(100 * np.mean(np.abs(d) <= 5.0))}
