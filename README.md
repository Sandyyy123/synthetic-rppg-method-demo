> **Proprietary - All Rights Reserved. Licensed to Sandeep Grover. No reuse without written permission. See LICENSE.**
> Proposal methodology demo on synthetic data.

# Synthetic rPPG method demo (POS, CHROM, GREEN)

**Everything in this folder runs on SYNTHETIC SIGNALS. These are NOT facial video results** and say nothing
about accuracy on real faces, cameras or subjects.

## What it does
1. `rppg_demo/synth.py` generates per-frame mean R, G, B traces that a skin ROI would yield, from a simple
   skin-reflection model with a KNOWN heart rate (55-110 bpm, slowly drifting), illumination drift,
   white-light intensity motion with a specular term, a coloured (non-white) motion term, sensor noise,
   and two illustrative skin-tone settings (darker = lower DC level and lower pulsatile amplitude).
   All coefficients are illustrative assumptions.
2. `rppg_demo/methods.py` implements GREEN (single channel baseline), CHROM (de Haan & Jeanne 2013) and
   POS (Wang et al. 2017) with sliding windows and overlap-add, then a 0.7-3.0 Hz Butterworth band-pass.
3. `rppg_demo/hr.py` estimates HR from the FFT peak in 10 s windows (2 s step), computes a blind SNR-style signal
   quality index (SQI, adapted from de Haan & Jeanne 2013), and agreement statistics: MAE, RMSE, Pearson r and Bland-Altman bias with 95%
   limits of agreement.
4. `main.py` runs 20 synthetic "subjects" x 5 conditions x 3 methods and writes `results.json`. The SQI gate uses
   one fixed threshold per method, set on a separate calibration set (different seeds), and reports the share of
   windows kept per condition, because a gate can silently exclude a subgroup.

## Run
```
pip install -r requirements.txt
python main.py --out results.json
python -m pytest -v tests
```
Runtime depends on the CPU. Run the tests locally to verify your environment.

## Limits (read before interpreting any number)
- The synthetic white-light motion is exactly what POS and CHROM are built to cancel (for POS it is removed
  exactly), so their low error here shows the implementations behave as intended. It does not predict real-world error.
- Windows from the same recording are not independent; real-data Bland-Altman limits should use a
  repeated-measures method (Bland & Altman 2007) and subject-level resampling.
- No face detection, landmark tracking or skin segmentation is exercised here; the input is already an ROI trace.
  On real video those steps add error sources that this demo does not model.

## References
- de Haan G, Jeanne V. Robust pulse rate from chrominance-based rPPG. IEEE TBME 2013;60(10):2878-2886. https://doi.org/10.1109/TBME.2013.2266196
- Wang W, den Brinker AC, Stuijk S, de Haan G. Algorithmic principles of remote PPG. IEEE TBME 2017;64(7):1479-1491. https://doi.org/10.1109/TBME.2016.2609282
- Bland JM, Altman DG. Agreement between methods of measurement with multiple observations per individual. J Biopharm Stat 2007;17(4):571-582. https://doi.org/10.1080/10543400701329422
- Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet 1986;327(8476):307-310. https://doi.org/10.1016/S0140-6736(86)90837-8

Sandeep Grover, PhD
