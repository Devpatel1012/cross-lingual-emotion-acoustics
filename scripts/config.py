"""Registry of corpora, emotion label maps and sampling parameters.

Self-contained: nothing here imports from the diarization pipeline, and every
path stays inside cross_lingual_emotion_experiment/.

All sources are fetched file-by-file (HF rows API / HF file resolve / GitHub
raw) rather than as archives, because the link available here runs at roughly
60 kB/s and the full archives are 0.2-1.6 GB each.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLIPS = ROOT / "data" / "clips"
FEATURES = ROOT / "features"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

# Target emotions. "sad" is the anchor emotion of the experiment (crying /
# grief); "angry" and "happy" are the contrast emotions (3 per language max).
EMOTIONS = ["sad", "angry", "happy"]

TARGET_SR = 16000
CLIPS_PER_CELL = 18          # clips per (language, emotion) cell
POOL_PER_CELL = 30           # candidates downloaded per cell
MIN_DUR = 1.0                # seconds, after silence trimming
MAX_DUR = 12.0
TRIM_TOP_DB = 35

# Canonical label synonyms seen across the corpora.
LABEL_SYNONYMS = {
    "sad": {"sad", "sadness", "sadnes", "tristeza", "tristesse", "trauer",
            "sorrow", "grief", "s"},
    "angry": {"angry", "anger", "enojo", "colere", "colère", "wut", "mad",
              "a", "w"},
    "happy": {"happy", "happiness", "joy", "felicidad", "joie", "freude",
              "h", "f"},
}


def canon(label):
    """Map a corpus-specific emotion label onto sad / angry / happy."""
    if label is None:
        return None
    k = str(label).strip().lower()
    for emo, syn in LABEL_SYNONYMS.items():
        if k in syn:
            return emo
    return None


# ---------------------------------------------------------------------------
# Corpora. 11 languages / 7 language families.
#   kind="hf_rows"   paged through datasets-server /rows, per-row audio asset
#   kind="hf_files"  individual wav files resolved from a dataset repo
#   kind="github"    individual wav files from a GitHub repo
# speaker_from: how to recover a speaker id ("col:<name>", "ravdess",
#                "underscore:<i>", "dash:-1", "paren", "none")
# ---------------------------------------------------------------------------
SOURCES = {
    "en": dict(
        language="English", family="Germanic", kind="hf_rows",
        corpus="RAVDESS",
        repo="DynamicSuperb/EmotionalSpeechAudioClassification_RAVDESS-EmotionalSound",
        config="default", split="test", audio_col="audio", label_col="label",
        name_col="file", speaker_from="ravdess",
        licence="CC BY-NC-SA 4.0", reference="Livingstone & Russo 2018",
    ),
    "de": dict(
        language="German", family="Germanic", kind="hf_rows",
        corpus="EmoDB", repo="confit/emodb-parquet",
        config="default", split="train", audio_col="audio",
        label_col="emotion", name_col=None, speaker_from="none",
        licence="free for research", reference="Burkhardt et al. 2005",
    ),
    "el": dict(
        language="Greek", family="Hellenic", kind="hf_rows",
        corpus="AESDD", repo="mteb/AESDD",
        config="default", split="train", audio_col="audio",
        label_col="label", name_col=None, speaker_from="none",
        licence="research use", reference="Vryzas et al. 2018",
    ),
    "pl": dict(
        language="Polish", family="Slavic", kind="hf_rows",
        corpus="nEMO", repo="amu-cai/nEMO",
        config="default", split="train", audio_col="audio",
        label_col="emotion", name_col="file_id", speaker_from="col:speaker_id",
        licence="CC BY-NC-SA 4.0", reference="Christop 2024",
    ),
    "ru": dict(
        language="Russian", family="Slavic", kind="hf_rows",
        corpus="RESD", repo="Aniemore/resd",
        config="default", split="train", audio_col="speech",
        label_col="emotion", name_col="name", speaker_from="underscore:0",
        licence="MIT", reference="Aniemore RESD",
    ),
    "fa": dict(
        language="Persian", family="Iranian", kind="hf_rows",
        corpus="ShEMO", repo="minoosh/shEMO_speech",
        config="default", split="train", audio_col="audio",
        label_col="emotion", name_col=None, speaker_from="none",
        licence="research use", reference="Nezami et al. 2019",
    ),
    "bn": dict(
        language="Bangla", family="Indo-Aryan", kind="hf_rows",
        corpus="SUBESCO", repo="sustcsenlp/bn_emotion_speech_corpus",
        config="default", split="train", audio_col="audio",
        label_col="text", name_col="text", label_in_name=True,
        speaker_from="underscore:2",
        licence="CC BY 4.0", reference="Sultana et al. 2021 (SUBESCO)",
    ),
    "ja": dict(
        language="Japanese", family="Japonic", kind="hf_rows",
        corpus="JVNV", repo="asahi417/jvnv-emotional-speech-corpus",
        config="default", split="test", audio_col="audio",
        label_col="style", name_col=None, speaker_from="col:speaker_id",
        licence="CC BY-SA 4.0", reference="Xin et al. 2023 (JVNV)",
    ),
    "zh": dict(
        language="Mandarin", family="Sinitic", kind="hf_files",
        corpus="CASIA", repo="BillyLin/CASIA_speech_emotion_recognition",
        prefix="CASIA/", speaker_from="dash:-1",
        licence="research use", reference="CASIA Chinese emotional corpus",
    ),
    "ur": dict(
        language="Urdu", family="Indo-Aryan", kind="github",
        corpus="URDU-Dataset", repo="siddiquelatif/URDU-Dataset",
        branch="master", prefix="", speaker_from="underscore:0",
        licence="research use", reference="Latif et al. 2018",
    ),
    "tr": dict(
        language="Turkish", family="Turkic", kind="github",
        corpus="TurEV-DB", repo="Xeonen/TurEV-DB", branch="master",
        prefix="Sound Source/", speaker_from="none",
        # TurEV-DB is single-word utterances of ~1 s; the global 1.0 s floor
        # would reject the whole corpus after silence trimming
        min_dur=0.5,
        licence="research use", reference="Canpolat et al. 2020",
    ),
}
