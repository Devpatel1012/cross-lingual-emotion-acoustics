"""Download a balanced pool of emotional-speech clips, one file at a time.

For every (language, emotion) cell we collect POOL_PER_CELL candidates:
  hf_rows   page through datasets-server /rows, keep rows whose label maps to
            sad/angry/happy, download the per-row audio asset
  hf_files  resolve individual wav files from a dataset repo
  github    raw.githubusercontent per-file

Writes data/raw/<lang>/<emotion>/*.wav and data/raw/<lang>/candidates.csv.

Usage: python3 01_fetch.py [--only en,de] [--workers 16]
"""
import argparse
import csv
import json
import random
import re
import subprocess
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from config import EMOTIONS, POOL_PER_CELL, RAW, SOURCES, canon

DS = "https://datasets-server.huggingface.co"
UA = "Mozilla/5.0 (cross-lingual-emotion-experiment)"
PAGE = 50


def log(*a):
    print(*a, flush=True)


def api(url, timeout=120, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            if i == retries - 1:
                raise
            log(f"    api retry {i + 1}: {type(e).__name__}")
    return None


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    rc = subprocess.run(["curl", "-sSL", "--retry", "2", "--max-time", "180",
                         "-A", UA, "-o", str(tmp), url],
                        capture_output=True).returncode
    if rc == 0 and tmp.exists() and tmp.stat().st_size > 1000:
        tmp.rename(dest)
        return True
    tmp.unlink(missing_ok=True)
    return False


def speaker_id(cfg, row, name):
    mode = cfg.get("speaker_from", "none")
    if mode.startswith("col:"):
        return str(row.get(mode[4:], "u"))
    if mode == "ravdess":
        parts = str(name).split(".")[0].split("-")
        return f"a{parts[6]}" if len(parts) == 7 else "u"
    if mode.startswith("underscore:"):
        toks = str(name).split("_")
        i = int(mode.split(":")[1])
        return toks[i] if len(toks) > i else "u"
    if mode.startswith("dash:"):
        toks = str(name).split("-")
        return toks[int(mode.split(":")[1])] if len(toks) > 1 else "u"
    if mode == "paren":
        m = re.search(r"\((\d+)\)", str(name))
        return m.group(1) if m else "u"
    return "u"


def label_of(cfg, row, class_names):
    if cfg.get("label_in_name"):
        # only whole words: single letters in a filename are speaker/sentence
        # codes, not emotion labels
        for tok in re.split(r"[_\-\s.]+", str(row.get(cfg["name_col"], ""))):
            c = canon(tok) if len(tok) >= 3 else None
            if c:
                return c
        return None
    v = row.get(cfg["label_col"])
    if isinstance(v, int) and class_names:
        v = class_names[v] if v < len(class_names) else None
    return canon(v)


# --------------------------------------------------------------------------
def fetch_hf_rows(lang, cfg, workers):
    ds = urllib.parse.quote(cfg["repo"], safe="")
    url0 = (f"{DS}/first-rows?dataset={ds}&config={cfg['config']}"
            f"&split={cfg['split']}")
    meta = api(url0)
    class_names = None
    for f in meta.get("features", []):
        if f["name"] == cfg.get("label_col"):
            class_names = f["type"].get("names")
    total = meta.get("num_rows_total") or 0
    if not total:
        total = api(f"{DS}/size?dataset={ds}&config={cfg['config']}"
                    )["size"]["splits"][0]["num_rows"]
    log(f"  {total} rows, label classes={class_names}")

    # Visit pages spread over the whole split so one speaker/session cannot
    # dominate (most corpora are stored grouped by speaker).
    offsets = list(range(0, max(total - 1, 1), PAGE))
    random.Random(7).shuffle(offsets)
    pool = defaultdict(list)
    jobs = []
    need = POOL_PER_CELL
    for off in offsets:
        if all(len(pool[e]) >= need for e in EMOTIONS):
            break
        page = api(f"{DS}/rows?dataset={ds}&config={cfg['config']}"
                   f"&split={cfg['split']}&offset={off}&length={PAGE}")
        for item in page.get("rows", []):
            row = item["row"]
            emo = label_of(cfg, row, class_names)
            if emo not in EMOTIONS or len(pool[emo]) >= need:
                continue
            au = row.get(cfg["audio_col"])
            src = au[0]["src"] if isinstance(au, list) and au else None
            if not src:
                continue
            name = str(row.get(cfg["name_col"]) or f"r{item['row_idx']}")
            spk = speaker_id(cfg, row, name)
            stem = re.sub(r"[^A-Za-z0-9]+", "", name)[-24:] or str(item["row_idx"])
            ext = "." + src.split("?")[0].rsplit(".", 1)[-1][:4]
            dest = RAW / lang / emo / f"{spk}_{item['row_idx']}_{stem}{ext}"
            pool[emo].append(dict(emotion=emo, speaker=spk, utt=name,
                                  file=dest))
            jobs.append((src, dest))
        log(f"    offset {off}: " +
            " ".join(f"{e}={len(pool[e])}" for e in EMOTIONS))
    ok = run_downloads(jobs, workers)
    return [r for e in EMOTIONS for r in pool[e] if r["file"] in ok]


def fetch_hf_files(lang, cfg, workers):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
    by_emo = defaultdict(list)
    for f in files:
        if not f.lower().endswith(".wav"):
            continue
        rel = f[len(cfg["prefix"]):] if f.startswith(cfg["prefix"]) else f
        emo = canon(rel.split("/")[0])
        if emo in EMOTIONS:
            by_emo[emo].append(f)
    pool, jobs = [], []
    for emo, fl in by_emo.items():
        fl.sort()
        step = max(1, len(fl) // POOL_PER_CELL)
        for f in fl[::step][:POOL_PER_CELL]:
            name = f.rsplit("/", 1)[-1]
            spk = speaker_id(cfg, {}, name.rsplit(".", 1)[0])
            dest = RAW / lang / emo / f"{spk}_{name}"
            pool.append(dict(emotion=emo, speaker=spk, utt=name, file=dest))
            jobs.append((f"https://huggingface.co/datasets/{cfg['repo']}/"
                         f"resolve/main/{urllib.parse.quote(f)}", dest))
        log(f"    {emo}: {min(len(fl), POOL_PER_CELL)} of {len(fl)}")
    ok = run_downloads(jobs, workers)
    return [r for r in pool if r["file"] in ok]


def fetch_github(lang, cfg, workers):
    tree = api(f"https://api.github.com/repos/{cfg['repo']}/git/trees/"
               f"{cfg['branch']}?recursive=1")
    pre = cfg.get("prefix", "")
    by_emo = defaultdict(list)
    for node in tree["tree"]:
        p = node["path"]
        if not p.lower().endswith(".wav") or not p.startswith(pre):
            continue
        emo = canon(p[len(pre):].split("/")[0])
        if emo in EMOTIONS:
            by_emo[emo].append(p)
    pool, jobs = [], []
    for emo, fl in by_emo.items():
        fl.sort()
        step = max(1, len(fl) // POOL_PER_CELL)
        for p in fl[::step][:POOL_PER_CELL]:
            name = p.rsplit("/", 1)[-1]
            spk = speaker_id(cfg, {}, name.rsplit(".", 1)[0])
            dest = RAW / lang / emo / f"{spk}_{name}"
            pool.append(dict(emotion=emo, speaker=spk, utt=name, file=dest))
            jobs.append((f"https://raw.githubusercontent.com/{cfg['repo']}/"
                         f"{cfg['branch']}/{urllib.parse.quote(p)}", dest))
        log(f"    {emo}: {min(len(fl), POOL_PER_CELL)} of {len(fl)}")
    ok = run_downloads(jobs, workers)
    return [r for r in pool if r["file"] in ok]


def run_downloads(jobs, workers):
    log(f"    downloading {len(jobs)} files with {workers} workers")
    ok = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (src, dest), good in zip(jobs, ex.map(lambda j: download(*j), jobs)):
            if good:
                ok.add(dest)
    log(f"    {len(ok)}/{len(jobs)} downloaded")
    return ok


HANDLERS = {"hf_rows": fetch_hf_rows, "hf_files": fetch_hf_files,
            "github": fetch_github}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    langs = [l for l in (args.only.split(",") if args.only else SOURCES)
             if l in SOURCES]
    for lang in langs:
        cfg = SOURCES[lang]
        log(f"[{lang}] {cfg['language']} / {cfg['corpus']} ({cfg['kind']})")
        try:
            recs = HANDLERS[cfg["kind"]](lang, cfg, args.workers)
        except Exception as e:
            log(f"  !! FAILED {lang}: {type(e).__name__}: {e}")
            continue
        out = RAW / lang / "candidates.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["lang", "emotion", "speaker", "utt", "file"])
            for r in recs:
                w.writerow([lang, r["emotion"], r["speaker"], r["utt"],
                            str(r["file"].relative_to(RAW))])
        counts = defaultdict(int)
        for r in recs:
            counts[r["emotion"]] += 1
        log(f"  {lang}: " + " ".join(f"{e}={counts[e]}" for e in EMOTIONS))
    log("done")


if __name__ == "__main__":
    main()
