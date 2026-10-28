"""Select a balanced clip set from the downloaded pool and standardise it.

For every (language, emotion) cell we keep CLIPS_PER_CELL clips, spread over as
many speakers as the corpus exposes, resampled to 16 kHz mono, silence-trimmed
and capped at MAX_DUR seconds.

Amplitude is deliberately NOT normalised: level differences are handled later
by per-language standardisation of dB-domain features, which removes a constant
per-corpus recording gain while keeping within-corpus loudness contrasts.

Output: data/clips/<lang>/<emotion>/*.wav + features/manifest.csv
"""
import argparse
from collections import defaultdict

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from config import (CLIPS, CLIPS_PER_CELL, EMOTIONS, FEATURES, MAX_DUR,
                    MIN_DUR, RAW, SOURCES, TARGET_SR, TRIM_TOP_DB)


def log(*a):
    print(*a, flush=True)


def standardise(y, min_dur=MIN_DUR):
    """Trim silence, cap length. Returns None if too short or silent."""
    if y is None or y.size == 0 or not np.isfinite(y).all():
        return None
    y = y - np.mean(y)
    if np.max(np.abs(y)) < 1e-5:
        return None
    yt, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if yt.size / TARGET_SR < min_dur:
        return None
    cap = int(MAX_DUR * TARGET_SR)
    if yt.size > cap:                          # centre crop
        s = (yt.size - cap) // 2
        yt = yt[s:s + cap]
    return yt


def round_robin(recs):
    """Order candidates so speakers alternate (no actor dominates a cell)."""
    by_spk = defaultdict(list)
    for r in recs:
        by_spk[r["speaker"]].append(r)
    for v in by_spk.values():
        v.sort(key=lambda r: r["file"])
    order, out = sorted(by_spk), []
    i = 0
    while any(by_spk[s] for s in order):
        s = order[i % len(order)]
        if by_spk[s]:
            out.append(by_spk[s].pop(0))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    langs = [l for l in (args.only.split(",") if args.only else SOURCES)
             if l in SOURCES and (RAW / l / "candidates.csv").exists()]

    rows = []
    for lang in langs:
        cfg = SOURCES[lang]
        cand = pd.read_csv(RAW / lang / "candidates.csv")
        per_emo = defaultdict(list)
        for r in cand.itertuples():
            per_emo[r.emotion].append(dict(speaker=str(r.speaker),
                                           utt=str(r.utt), file=str(r.file)))
        log(f"[{lang}] {cfg['language']} / {cfg['corpus']}: " +
            ", ".join(f"{e}={len(per_emo.get(e, []))}" for e in EMOTIONS))
        for emo in EMOTIONS:
            pool = round_robin(per_emo.get(emo, []))
            outdir = CLIPS / lang / emo
            outdir.mkdir(parents=True, exist_ok=True)
            kept, skipped, spks = 0, 0, set()
            for r in pool:
                if kept >= CLIPS_PER_CELL:
                    break
                try:
                    y, _ = librosa.load(RAW / r["file"], sr=TARGET_SR, mono=True)
                    y = standardise(y, cfg.get('min_dur', MIN_DUR))
                except Exception:
                    y = None
                if y is None:
                    skipped += 1
                    continue
                name = f"{lang}_{emo}_{r['speaker']}_{kept:02d}.wav"
                sf.write(outdir / name, y, TARGET_SR, subtype="PCM_16")
                spks.add(r["speaker"])
                rows.append(dict(
                    clip_id=name[:-4], language=cfg["language"], lang=lang,
                    family=cfg["family"], corpus=cfg["corpus"], emotion=emo,
                    speaker=r["speaker"], source_utt=r["utt"],
                    duration=round(y.size / TARGET_SR, 3),
                    path=str((outdir / name).relative_to(CLIPS.parent.parent)),
                ))
                kept += 1
            log(f"   {emo}: kept {kept} clips from {len(spks)} speaker ids "
                f"({skipped} unusable)")

    FEATURES.mkdir(parents=True, exist_ok=True)
    man = FEATURES / "manifest.csv"
    df = pd.DataFrame(rows)
    if man.exists() and args.only:
        old = pd.read_csv(man)
        df = pd.concat([old[~old.lang.isin(langs)], df], ignore_index=True)
    df = df.sort_values(["lang", "emotion", "clip_id"])
    df.to_csv(man, index=False)
    log(f"\nmanifest: {man}  ({len(df)} clips, {df.lang.nunique()} languages)")
    log(df.pivot_table(index="lang", columns="emotion", values="clip_id",
                       aggfunc="count").to_string())
    log(f"\ntotal audio: {df.duration.sum() / 60:.1f} min, "
        f"median clip {df.duration.median():.2f} s")


if __name__ == "__main__":
    main()
