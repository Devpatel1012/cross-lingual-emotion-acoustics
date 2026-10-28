"""Extract prosodic / voice-quality / spectral features from every clip.

Feature set is eGeMAPS-inspired and computed with librosa + Praat (parselmouth):
  pitch      f0 statistics in semitones (gender- and language-robust)
  energy     RMS loudness in dB plus its dynamics and attack rate
  quality    jitter, shimmer, HNR
  spectral   centroid/spread/rolloff/flatness/flux, alpha ratio, Hammarberg
  temporal   voiced ratio, pause ratio, syllable-peak rate, segment durations
  cepstral   MFCC 1-13 mean and std

Output: features/features.csv  (one row per clip)
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

from config import FEATURES, ROOT, TARGET_SR
from feature_defs import ALL_F, PITCH_F

warnings.filterwarnings("ignore")

F0_FLOOR, F0_CEIL = 60.0, 600.0
FRAME, HOP = 1024, 256          # 64 ms / 16 ms at 16 kHz


def st(hz):
    """Hz -> semitones relative to 100 Hz."""
    return 12.0 * np.log2(np.maximum(hz, 1e-6) / 100.0)


def seg_durations(mask, hop_s):
    out, run = [], 0
    for v in mask:
        if v:
            run += 1
        elif run:
            out.append(run * hop_s)
            run = 0
    if run:
        out.append(run * hop_s)
    return float(np.mean(out)) if out else np.nan


def extract(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    dur = y.size / sr
    f = {}

    # ---- pitch (Praat autocorrelation) -----------------------------------
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=F0_FLOOR,
                         pitch_ceiling=F0_CEIL)
    f0 = pitch.selected_array["frequency"]
    voiced = f0 > 0
    f["voiced_ratio"] = float(voiced.mean()) if f0.size else np.nan
    if voiced.sum() >= 5:
        v = st(f0[voiced])
        f["f0_mean_st"] = float(v.mean())
        f["f0_median_st"] = float(np.median(v))
        f["f0_std_st"] = float(v.std())
        p5, p95 = np.percentile(v, [5, 95])
        f["f0_p5_st"], f["f0_p95_st"] = float(p5), float(p95)
        f["f0_range_st"] = float(p95 - p5)
        d = np.diff(v)
        f["f0_abs_slope_st_s"] = float(np.abs(d).mean() / 0.01)
        f["f0_rise_ratio"] = float((d > 0).mean())
    else:
        for k in PITCH_F:
            f[k] = np.nan

    # ---- energy / loudness ----------------------------------------------
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    active = rms_db > (rms_db.max() - 35.0)
    f["rms_mean_db"] = float(rms_db[active].mean())
    f["rms_std_db"] = float(rms_db[active].std())
    p5, p95 = np.percentile(rms_db[active], [5, 95])
    f["rms_range_db"] = float(p95 - p5)
    f["rms_p95_db"] = float(p95)
    peaks = librosa.util.peak_pick(rms_db, pre_max=3, post_max=3, pre_avg=5,
                                   post_avg=5, delta=2.0, wait=5)
    f["loud_peak_rate"] = float(len(peaks) / dur)
    drdt = np.diff(rms_db) / (HOP / sr)
    f["attack_slope_db_s"] = float(np.percentile(drdt, 95))

    # ---- voice quality ---------------------------------------------------
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", F0_FLOOR, F0_CEIL)
        f["jitter_local"] = call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3)
        f["shimmer_local"] = call([snd, pp], "Get shimmer (local)", 0, 0,
                                  1e-4, 0.02, 1.3, 1.6)
    except Exception:
        f["jitter_local"] = f["shimmer_local"] = np.nan
    try:
        h = snd.to_harmonicity_cc(minimum_pitch=F0_FLOOR)
        hv = h.values[h.values > -200]
        f["hnr_db"] = float(hv.mean()) if hv.size else np.nan
    except Exception:
        f["hnr_db"] = np.nan

    # ---- spectral --------------------------------------------------------
    S = np.abs(librosa.stft(y, n_fft=FRAME, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME)
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    f["centroid_hz"] = float(cent.mean())
    f["centroid_std_hz"] = float(cent.std())
    f["spread_hz"] = float(librosa.feature.spectral_bandwidth(S=S, sr=sr)[0].mean())
    f["rolloff85_hz"] = float(librosa.feature.spectral_rolloff(
        S=S, sr=sr, roll_percent=0.85)[0].mean())
    f["flatness"] = float(librosa.feature.spectral_flatness(S=S)[0].mean())
    f["flux"] = float(np.mean(np.sqrt(((np.diff(S, axis=1) /
                                        (S.max() + 1e-10)) ** 2).sum(axis=0))))
    P = S ** 2
    band = lambda lo, hi: P[(freqs >= lo) & (freqs < hi)].sum() + 1e-12
    f["alpha_ratio_db"] = float(10 * np.log10(band(1000, 5000) / band(50, 1000)))
    f["hf_ratio_db"] = float(10 * np.log10(band(4000, 8000) / band(0, 4000)))
    mag_db = 20 * np.log10(np.maximum(S.mean(axis=1), 1e-10))
    pk = lambda lo, hi: mag_db[(freqs >= lo) & (freqs < hi)].max()
    f["hammarberg_db"] = float(pk(0, 2000) - pk(2000, 5000))
    for lo, hi, key in ((0, 500, "slope_0_500_db_khz"),
                        (500, 1500, "slope_500_1500_db_khz")):
        m = (freqs >= lo) & (freqs < hi)
        f[key] = float(np.polyfit(freqs[m] / 1000.0, mag_db[m], 1)[0])

    # ---- temporal --------------------------------------------------------
    f["pause_ratio"] = float(1.0 - active.mean())
    env = librosa.onset.onset_strength(S=librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME,
                                       hop_length=HOP)), sr=sr, hop_length=HOP)
    sp = librosa.util.peak_pick(env, pre_max=4, post_max=4, pre_avg=6,
                                post_avg=6, delta=0.15, wait=4)
    f["syl_rate_hz"] = float(len(sp) / dur)
    hop_s = 0.01
    f["voiced_seg_mean_s"] = seg_durations(voiced, hop_s)
    f["unvoiced_seg_mean_s"] = seg_durations(~voiced, hop_s)

    # ---- cepstral --------------------------------------------------------
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=FRAME, hop_length=HOP)
    for i in range(13):
        f[f"mfcc{i + 1}_mean"] = float(mf[i].mean())
        f[f"mfcc{i + 1}_std"] = float(mf[i].std())

    f["dur_s"] = float(dur)
    return f


def _job(args):
    clip_id, path = args
    try:
        return clip_id, extract(path), None
    except Exception as e:
        return clip_id, None, f"{type(e).__name__}: {e}"


def main():
    man = pd.read_csv(FEATURES / "manifest.csv")
    jobs = [(r.clip_id, str(ROOT / r.path)) for r in man.itertuples()]
    rows, errs = {}, []
    with ProcessPoolExecutor() as ex:
        for i, (cid, feats, err) in enumerate(ex.map(_job, jobs, chunksize=4), 1):
            if err:
                errs.append((cid, err))
            else:
                rows[cid] = feats
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    feat = pd.DataFrame.from_dict(rows, orient="index")
    feat.index.name = "clip_id"
    out = man.merge(feat.reset_index(), on="clip_id", how="inner")
    out.to_csv(FEATURES / "features.csv", index=False)
    print(f"features: {FEATURES / 'features.csv'}  {out.shape}")
    print(f"failed: {len(errs)}")
    for c, e in errs[:10]:
        print("   ", c, e)
    nan = out[ALL_F].isna().mean().sort_values(ascending=False)
    print("top NaN rates:\n", nan.head(8).to_string())


if __name__ == "__main__":
    main()
