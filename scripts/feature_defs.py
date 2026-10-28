"""Names and groupings of the acoustic features (shared by extraction/analysis)."""

PITCH_F = ["f0_mean_st", "f0_median_st", "f0_std_st", "f0_range_st",
           "f0_p5_st", "f0_p95_st", "f0_abs_slope_st_s", "f0_rise_ratio"]
ENERGY_F = ["rms_mean_db", "rms_std_db", "rms_range_db", "rms_p95_db",
            "loud_peak_rate", "attack_slope_db_s"]
QUALITY_F = ["jitter_local", "shimmer_local", "hnr_db"]
SPECTRAL_F = ["centroid_hz", "centroid_std_hz", "spread_hz", "rolloff85_hz",
              "flatness", "flux", "alpha_ratio_db", "hammarberg_db",
              "slope_0_500_db_khz", "slope_500_1500_db_khz", "hf_ratio_db"]
TEMPORAL_F = ["voiced_ratio", "pause_ratio", "syl_rate_hz",
              "voiced_seg_mean_s", "unvoiced_seg_mean_s"]
CEPSTRAL_F = [f"mfcc{i}_{s}" for i in range(1, 14) for s in ("mean", "std")]

GROUPS = {"pitch": PITCH_F, "energy": ENERGY_F, "quality": QUALITY_F,
          "spectral": SPECTRAL_F, "temporal": TEMPORAL_F,
          "cepstral": CEPSTRAL_F}
INTERPRETABLE = PITCH_F + ENERGY_F + QUALITY_F + SPECTRAL_F + TEMPORAL_F
ALL_F = INTERPRETABLE + CEPSTRAL_F

# Features that move if a corpus was recorded or mastered at a different gain.
# Everything else is a ratio, a dB difference, a shape or a time measure, and
# is invariant to a constant per-corpus level offset.
GAIN_SENSITIVE = ["rms_mean_db", "rms_p95_db", "mfcc1_mean"]

# Human-readable labels for figures and the report.
PRETTY = {
    "f0_mean_st": "mean F0 (semitones)", "f0_median_st": "median F0 (st)",
    "f0_std_st": "F0 variability (st)", "f0_range_st": "F0 range p5-p95 (st)",
    "f0_p5_st": "F0 5th pct (st)", "f0_p95_st": "F0 95th pct (st)",
    "f0_abs_slope_st_s": "F0 movement rate (st/s)",
    "f0_rise_ratio": "fraction of rising F0 frames",
    "rms_mean_db": "mean loudness (dB)", "rms_std_db": "loudness variability (dB)",
    "rms_range_db": "loudness range (dB)", "rms_p95_db": "loudness 95th pct (dB)",
    "loud_peak_rate": "loudness peaks per s",
    "attack_slope_db_s": "attack steepness (dB/s)",
    "jitter_local": "jitter (local)", "shimmer_local": "shimmer (local)",
    "hnr_db": "harmonics-to-noise ratio (dB)",
    "centroid_hz": "spectral centroid (Hz)",
    "centroid_std_hz": "centroid variability (Hz)",
    "spread_hz": "spectral spread (Hz)", "rolloff85_hz": "85% roll-off (Hz)",
    "flatness": "spectral flatness", "flux": "spectral flux",
    "alpha_ratio_db": "alpha ratio 1-5k/50-1k (dB)",
    "hammarberg_db": "Hammarberg index (dB)",
    "slope_0_500_db_khz": "spectral slope 0-500 Hz",
    "slope_500_1500_db_khz": "spectral slope 500-1500 Hz",
    "hf_ratio_db": "high-frequency ratio (dB)",
    "voiced_ratio": "voiced frame ratio", "pause_ratio": "pause ratio",
    "syl_rate_hz": "syllable-peak rate (Hz)",
    "voiced_seg_mean_s": "mean voiced segment (s)",
    "unvoiced_seg_mean_s": "mean unvoiced segment (s)",
}
