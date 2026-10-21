"""SYNTHETIC RGB skin-trace generator. Not facial video, not real subjects.

Produces the per-frame mean R, G, B values that a face skin ROI would yield,
with a KNOWN heart rate, so that rPPG methods can be scored exactly.

Model (simplified, after the skin-reflection model in Wang et al. 2017,
doi:10.1109/TBME.2016.2609282):
    C(t) = I(t) * (u_skin + p(t) * u_pulse + m(t) * u_spec) + noise
where I(t) is illumination intensity (slow drift + motion-driven change),
u_skin the DC skin colour, p(t) the blood-volume pulse, u_pulse the pulsatile
colour direction (green strongest), and m(t) a specular/motion term.
All coefficients are illustrative assumptions, not measured values.
"""
from dataclasses import dataclass

import numpy as np

# Illustrative relative pulsatile strengths per channel (G > B > R).
PULSE_DIR = np.array([0.33, 0.77, 0.53])
# Colour direction of non-white motion artefact (illustrative assumption).
CHROMA_DIR = np.array([1.0, 0.55, 0.25])
# Illustrative DC skin colours (normalised 0-1) for two skin-tone settings.
SKIN_DC = {"lighter": np.array([0.78, 0.58, 0.48]),
           "darker": np.array([0.42, 0.28, 0.22])}
# Lower absolute pulsatile amplitude for the darker setting (assumption). Because the DC level is
# also lower, the normalised pulse (AC/DC) is similar; what changes is the relative size of the
# fixed sensor noise.
PULSE_AMP = {"lighter": 0.010, "darker": 0.005}


@dataclass
class Scenario:
    name: str
    hr_bpm: float = 72.0
    hr_drift_bpm: float = 4.0       # slow sinusoidal HR variation amplitude
    motion: float = 0.0             # amplitude of motion-driven intensity change
    illum_drift: float = 0.02       # slow illumination drift amplitude
    noise_sd: float = 0.0015        # sensor noise SD
    chroma_motion: float = 0.0      # coloured (non-white) motion term, e.g. ROI drift onto shadow/hair
    skin: str = "lighter"
    fs: float = 30.0
    duration_s: float = 60.0
    seed: int = 0


def generate(sc: Scenario):
    """Return (t, rgb[N,3], true_hr_bpm[N]) for one synthetic recording."""
    rng = np.random.default_rng(sc.seed)
    n = int(sc.fs * sc.duration_s)
    t = np.arange(n) / sc.fs
    hr = sc.hr_bpm + sc.hr_drift_bpm * np.sin(2 * np.pi * t / 40.0 + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(hr / 60.0) / sc.fs
    pulse = np.sin(phase) + 0.25 * np.sin(2 * phase + 0.6)  # with a harmonic
    pulse = pulse / np.std(pulse)

    illum = 1.0 + sc.illum_drift * np.sin(2 * np.pi * t / 25.0 + rng.uniform(0, 2 * np.pi))
    # Motion: band-limited random intensity change (0.2-2 Hz), overlaps cardiac band.
    mot = np.zeros(n)
    if sc.motion > 0:
        for f in rng.uniform(0.2, 2.0, size=6):
            mot += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        mot = sc.motion * mot / np.std(mot)
    intensity = illum * (1.0 + mot)
    spec = 0.5 * np.abs(mot)  # specular term rises with motion (white light)

    dc = SKIN_DC[sc.skin]
    amp = PULSE_AMP[sc.skin]
    rgb = intensity[:, None] * (dc[None, :] + amp * pulse[:, None] * PULSE_DIR[None, :]
                                + spec[:, None] * np.ones(3)[None, :] * 0.1)
    if sc.chroma_motion > 0:
        # Breaks the white-light assumption POS/CHROM rely on: motion with a colour tint.
        rgb = rgb + sc.chroma_motion * (mot / (np.std(mot) + 1e-12))[:, None] * CHROMA_DIR[None, :]
    rgb = rgb + rng.normal(0, sc.noise_sd, size=rgb.shape)
    return t, rgb, hr


def scenario_grid(n_subjects: int = 20, base_seed: int = 2026, seed_offset: int = 0):
    """Synthetic 'subjects' across five conditions with varied true HR.

    Use a different base_seed/seed_offset for an independent calibration set.
    """
    rng = np.random.default_rng(base_seed)
    conds = {
        "still_lighter": dict(motion=0.0, skin="lighter"),
        "still_darker": dict(motion=0.0, skin="darker"),
        "motion_lighter": dict(motion=0.03, skin="lighter"),
        "motion_darker": dict(motion=0.03, skin="darker"),
        # Hardest case: coloured motion + low light (3x sensor noise) + darker skin.
        "hard_darker": dict(motion=0.03, chroma_motion=0.004, noise_sd=0.0045, skin="darker"),
    }
    out = []
    for cname, kw in conds.items():
        for i in range(n_subjects):
            out.append(Scenario(name=cname, hr_bpm=float(rng.uniform(55, 110)),
                                seed=seed_offset + 1000 * len(out) + i, **kw))
    return out
