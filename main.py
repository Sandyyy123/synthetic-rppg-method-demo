"""Run the synthetic rPPG benchmark and write results.json.

Usage: python main.py [--subjects 20] [--out results.json]
Everything here is SYNTHETIC SIGNALS, NOT FACIAL VIDEO RESULTS.
"""
import argparse
import json
import platform
from collections import defaultdict

import numpy as np
import scipy

from rppg_demo.hr import agreement, windowed_hr
from rppg_demo.methods import METHODS
from rppg_demo.synth import generate, scenario_grid


def calibrate_gate(n_subjects=10, pct=25):
    """Fixed SQI threshold per method: the pct-th percentile of SQI on an INDEPENDENT
    calibration set (different seeds), pooled over all conditions."""
    sq = defaultdict(list)
    for sc in scenario_grid(n_subjects, base_seed=99, seed_offset=500000):
        _, rgb, hr = generate(sc)
        for m, fn in METHODS.items():
            sq[m] += windowed_hr(fn(rgb, sc.fs), hr, sc.fs)[2].tolist()
    return {m: float(np.percentile(v, pct)) for m, v in sq.items()}


def run(n_subjects=20):
    thr = calibrate_gate()
    pooled = defaultdict(lambda: {"est": [], "ref": [], "sqi": []})
    for sc in scenario_grid(n_subjects):
        _, rgb, hr = generate(sc)
        for m, fn in METHODS.items():
            est, ref, sqi = windowed_hr(fn(rgb, sc.fs), hr, sc.fs)
            for key in (m + "|" + sc.name, m + "|ALL"):
                pooled[key]["est"] += est.tolist()
                pooled[key]["ref"] += ref.tolist()
                pooled[key]["sqi"] += sqi.tolist()
    results = {}
    for key, v in sorted(pooled.items()):
        m, cond = key.split("|")
        a = agreement(v["est"], v["ref"])
        sqi = np.array(v["sqi"]); err = np.abs(np.array(v["est"]) - np.array(v["ref"]))
        a["sqi_median_db"] = float(np.median(sqi))
        # SQI gating with a FIXED threshold set on the calibration set (not on these windows).
        keep = sqi >= thr[m]
        a["sqi_gate_threshold_db"] = thr[m]
        a["mae_after_sqi_gate"] = float(err[keep].mean()) if keep.any() else float("nan")
        a["retained_pct_after_gate"] = float(100 * keep.mean())
        results.setdefault(m, {})[cond] = a
    return {
        "label": "SYNTHETIC SIGNALS, NOT FACIAL VIDEO RESULTS",
        "design": {"subjects_per_condition": n_subjects,
                   "conditions": ["still_lighter", "still_darker", "motion_lighter", "motion_darker", "hard_darker"],
                   "fs_hz": 30.0, "duration_s": 60.0, "hr_window_s": 10.0, "hr_step_s": 2.0,
                   "band_hz": [0.7, 3.0], "true_hr_range_bpm": [55, 110],
                   "sqi_gate": "fixed per-method threshold = 25th percentile of SQI on an independent calibration set (10 subjects x 5 conditions, different seeds)"},
        "env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "results": results,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    res = run(args.subjects)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    for m in ("GREEN", "CHROM", "POS"):
        for c in ("still_lighter", "still_darker", "motion_lighter", "motion_darker", "hard_darker", "ALL"):
            a = res["results"][m][c]
            print(f"{m:5s} {c:15s} n={a['n']:4d} MAE={a['mae']:6.2f} RMSE={a['rmse']:6.2f} "
                  f"bias={a['ba_bias']:+6.2f} LoA=[{a['ba_loa_low']:+7.2f},{a['ba_loa_high']:+7.2f}]")
    print("Wrote", args.out, "-", res["label"])
