import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run  # noqa: E402
from rppg_demo.hr import agreement, spectral_hr, windowed_hr  # noqa: E402
from rppg_demo.methods import bandpass, chrom, green, pos  # noqa: E402
from rppg_demo.synth import Scenario, generate  # noqa: E402

FS = 30.0


def _mae(method, sc):
    _, rgb, hr = generate(sc)
    est, ref, _ = windowed_hr(method(rgb, sc.fs), hr, sc.fs)
    return agreement(est, ref)["mae"]


def test_generator_is_deterministic_and_labelled_shape():
    sc = Scenario(name="x", seed=7)
    t1, rgb1, hr1 = generate(sc)
    t2, rgb2, hr2 = generate(sc)
    assert rgb1.shape == (int(FS * 60), 3)
    np.testing.assert_array_equal(rgb1, rgb2)
    np.testing.assert_array_equal(hr1, hr2)


@pytest.mark.parametrize("method", [pos, chrom, green])
def test_methods_recover_known_hr_on_still_signal(method):
    sc = Scenario(name="still", hr_bpm=84.0, seed=3)
    assert _mae(method, sc) < 1.5


@pytest.mark.parametrize("method", [pos, chrom])
def test_pos_chrom_cancel_white_intensity_motion_where_green_fails(method):
    sc = Scenario(name="motion", hr_bpm=66.0, motion=0.03, seed=11)
    assert _mae(green, sc) > 5.0          # single-channel baseline breaks
    assert _mae(method, sc) < 2.0         # colour-projection methods survive


def test_bland_altman_matches_hand_calculation():
    rng = np.random.default_rng(0)
    ref = rng.uniform(60, 100, 200)
    est = ref + 2.0 + rng.normal(0, 1.0, 200)
    a = agreement(est, ref)
    d = est - ref
    assert a["ba_bias"] == pytest.approx(d.mean())
    assert a["ba_loa_high"] - a["ba_loa_low"] == pytest.approx(2 * 1.96 * d.std(ddof=1))
    assert a["mae"] == pytest.approx(np.abs(d).mean())
    assert a["rmse"] == pytest.approx(np.sqrt((d ** 2).mean()))


def test_spectral_hr_and_sqi():
    t = np.arange(int(10 * FS)) / FS
    clean = np.sin(2 * np.pi * 1.25 * t)                       # 75 bpm
    noisy = clean + np.random.default_rng(1).normal(0, 2.0, t.size)
    hr_c, q_c = spectral_hr(clean, FS)
    _, q_n = spectral_hr(noisy, FS)
    assert hr_c == pytest.approx(75.0, abs=1.0)
    assert q_c > q_n


def test_bandpass_removes_dc_and_out_of_band():
    t = np.arange(int(20 * FS)) / FS
    x = 5.0 + np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 1.5 * t)
    y = bandpass(x, FS)
    assert abs(y.mean()) < 0.05
    hr, _ = spectral_hr(y, FS)
    assert hr == pytest.approx(90.0, abs=1.5)


def test_benchmark_output_is_labelled_synthetic():
    res = run(n_subjects=2)
    assert res["label"] == "SYNTHETIC SIGNALS, NOT FACIAL VIDEO RESULTS"
    assert set(res["results"]) == {"GREEN", "CHROM", "POS"}
    assert "ALL" in res["results"]["POS"]


def test_sqi_gate_calibration_set_is_independent_of_evaluation_set():
    from rppg_demo.synth import scenario_grid
    ev = {sc.seed for sc in scenario_grid(20)}
    cal = {sc.seed for sc in scenario_grid(10, base_seed=99, seed_offset=500000)}
    assert ev.isdisjoint(cal)
