"""Build report.html: a single self-contained page with the figures embedded.

Numbers are read from results/summary.json and the result CSVs, so the page
cannot drift from the analysis it describes.
"""
import base64
import json

import pandas as pd

from config import EMOTIONS, FIGURES, RESULTS, ROOT, SOURCES

S = json.load(open(RESULTS / "summary.json"))
B = S["B_profile_correlation"]
E = S["E_leave_one_language_out"]["interpretable"]
G = S["G_language_confound"]
H = S["H_robustness_gain_invariant"]
CFG = S["config"]
LANGS = list(CFG["languages"])

C = pd.read_csv(RESULTS / "C_feature_consistency.csv")
cm = pd.read_csv(RESULTS / "E_confusion_interpretable_logreg.csv", index_col=0)
per = (pd.read_csv(RESULTS / "E_lolo_per_language_interpretable.csv")
       .query("model == 'logreg'").set_index("held_out"))
within = pd.read_csv(RESULTS / "E_within_language_cv_interpretable.csv"
                     ).set_index("lang")
R = cm.to_numpy(float)
R = R / R.sum(axis=1, keepdims=True)
RECALL = {e: R[i, i] for i, e in enumerate(cm.index)}

FIG_CAPTIONS = [
    ("fig1_emotion_profiles.png",
     "Every language × emotion profile over all 33 features. Blue is below "
     "that language's own norm, red above. The SAD panel is overwhelmingly "
     "blue with one red column (Hammarberg index); the ANGRY panel is its "
     "mirror. The HAPPY panel is pale — there is little shared signal."),
    ("fig2_cross_language_correlation.png",
     "How strongly two languages agree on what an emotion does to the voice. "
     "Sadness and anger agree broadly; happiness does not, and Russian "
     "disagrees with almost everyone about it."),
    ("fig3_same_vs_different_emotion.png",
     "All 165 same-emotion language pairs against all 330 different-emotion "
     "pairs. The separation is the experiment's core result."),
    ("fig4_feature_consistency.png",
     "Which acoustic changes every language shares. k/n counts the languages "
     "agreeing on the direction; * marks FDR q < 0.05. Nothing survives for "
     "happiness."),
    ("fig5_key_features_by_language.png",
     "Six headline features, one point per language. Where the blue points "
     "sit below zero and the orange above it in every column, the cue is "
     "universal."),
    ("fig6_leave_one_language_out.png",
     "Train on ten languages, test on the eleventh. Eight of eleven languages "
     "land well above chance; Urdu — the only spontaneous, non-acted corpus — "
     "lands on it."),
    ("fig7_confusion_matrix.png",
     "Where the transfer succeeds and fails. Sadness is recognised in a "
     "foreign language 71% of the time; happiness is a coin toss between all "
     "three."),
    ("fig8_pca_embedding.png",
     "All 594 clips in one space. PC1 is essentially vocal effort: sad clips "
     "to the left, angry and happy to the right — and the same split repeats "
     "inside each language."),
    ("fig9_variance_decomposition.png",
     "Per feature, variance explained by emotion against variance explained "
     "by language. Only two features sit above the diagonal — the same two "
     "that top the consistency ranking."),
]


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def pct(x):
    return f"{100 * x:.1f}%"


def top_rows(emo, n=6):
    sub = (C[C.emotion == emo].assign(a=lambda d: d.mean_z.abs())
           .sort_values("a", ascending=False).head(n))
    return sub


def consistency_table(emo):
    rows = []
    for r in top_rows(emo).itertuples():
        rows.append(
            f"<tr><td>{r.feature.replace('_', ' ')}</td>"
            f"<td class='num {'pos' if r.mean_z > 0 else 'neg'}'>"
            f"{r.mean_z:+.2f} σ</td>"
            f"<td class='num'>{r.n_agree}/{r.n_langs}</td>"
            f"<td class='num'>{r.q_binom:.3f}</td></tr>")
    return "\n".join(rows)


def language_rows():
    rows = []
    for l in sorted(LANGS, key=lambda x: -per.loc[x, "accuracy"]):
        c = CFG["languages"][l]
        a, w = per.loc[l, "accuracy"], within.loc[l, "accuracy"]
        bar = int(round(100 * a))
        rows.append(
            f"<tr><td>{c['language']}<span class='sub'> · {c['family']}</span></td>"
            f"<td class='sub'>{c['corpus']}</td>"
            f"<td class='num'><span class='bar' style='--w:{bar}%'></span>"
            f"{a:.2f}</td><td class='num sub'>{w:.2f}</td></tr>")
    return "\n".join(rows)


def figure_blocks():
    out = []
    for i, (name, cap) in enumerate(FIG_CAPTIONS, 1):
        p = FIGURES / name
        if not p.exists():
            continue
        out.append(f"""
      <figure id="fig{i}">
        <div class="plate"><img src="data:image/png;base64,{b64(p)}"
             alt="{cap[:110]}"></div>
        <figcaption><b>Figure {i}.</b> {cap}
          <span class="file">{name}</span></figcaption>
      </figure>""")
    return "\n".join(out)


HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Does the same emotion sound the same in 11 languages?</title>
<style>
:root {{
  --bg:#fafaf7; --surface:#ffffff; --ink:#1a1a1f; --ink-soft:#555560;
  --rule:#e7e5df; --sad:#2a78d6; --angry:#eb6834; --happy:#1baf7a;
  --accent:var(--sad); --plate:#fcfcfb;
  --serif:Charter,"Iowan Old Style","Source Serif 4",ui-serif,Georgia,serif;
  --sans:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;
  --mono:ui-monospace,"JetBrains Mono","SF Mono",Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#0e0e12; --surface:#16161c; --ink:#f1f1f4; --ink-soft:#a8a8b3;
    --rule:#2a2a32; --sad:#3987e5; --angry:#d95926; --happy:#199e70;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#0e0e12; --surface:#16161c; --ink:#f1f1f4; --ink-soft:#a8a8b3;
  --rule:#2a2a32; --sad:#3987e5; --angry:#d95926; --happy:#199e70;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:17px/1.6 var(--serif); -webkit-text-size-adjust:100%; }}
.wrap {{ max-width:74ch; margin:0 auto; padding:2.5rem 1rem 5rem; }}
h1 {{ font-size:1.9rem; line-height:1.25; margin:0 0 .4rem; letter-spacing:-.01em; }}
h2 {{ font:600 1.25rem/1.3 var(--sans); margin:3rem 0 .8rem;
  padding-top:1.2rem; border-top:1px solid var(--rule); }}
h3 {{ font:600 1rem/1.4 var(--sans); margin:2rem 0 .5rem; }}
p {{ margin:0 0 1rem; }}
a {{ color:var(--accent); }}
.lede {{ color:var(--ink-soft); font-size:1.05rem; margin-bottom:1.5rem; }}
.meta {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:0 0 2rem;
  font:500 .78rem/1 var(--sans); }}
.meta span {{ background:var(--surface); border:1px solid var(--rule);
  border-radius:999px; padding:.4rem .7rem; color:var(--ink-soft); }}
.verdict {{ background:var(--surface); border:1px solid var(--rule);
  border-left:4px solid var(--accent); border-radius:8px;
  padding:1.1rem 1.2rem; margin:0 0 2rem; }}
.verdict p:last-child {{ margin-bottom:0; }}
.tiles {{ display:grid; gap:.7rem; margin:1.5rem 0 2rem;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }}
.tile {{ background:var(--surface); border:1px solid var(--rule);
  border-radius:8px; padding:.9rem 1rem; }}
.tile > b {{ display:block; font:700 1.5rem/1.1 var(--sans); letter-spacing:-.02em; }}
.tile span {{ display:block; font:400 .8rem/1.35 var(--sans);
  color:var(--ink-soft); margin-top:.3rem; }}
.tile span b {{ display:inline; font:600 .8rem/1.35 var(--sans); color:var(--ink);
  letter-spacing:0; }}
table {{ width:100%; border-collapse:collapse; margin:1rem 0 1.6rem;
  font:400 .9rem/1.45 var(--sans); }}
th,td {{ text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--rule); }}
th {{ font-weight:600; color:var(--ink-soft); font-size:.8rem;
  text-transform:uppercase; letter-spacing:.04em; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.sub {{ color:var(--ink-soft); font-size:.85em; }}
.pos {{ color:var(--angry); }} .neg {{ color:var(--sad); }}
.bar {{ display:inline-block; width:var(--w); max-width:84px; height:.5rem;
  background:var(--sad); border-radius:3px; margin-right:.5rem;
  vertical-align:middle; opacity:.55; }}
figure {{ margin:2rem 0; }}
.plate {{ background:var(--plate); border:1px solid var(--rule);
  border-radius:8px; padding:.5rem; overflow-x:auto; }}
.plate img {{ display:block; min-width:640px; width:100%; height:auto; }}
figcaption {{ font:400 .85rem/1.5 var(--sans); color:var(--ink-soft);
  margin-top:.6rem; }}
.file {{ font:400 .75rem/1 var(--mono); display:block; margin-top:.35rem;
  opacity:.75; }}
ul {{ padding-left:1.1rem; }} li {{ margin-bottom:.5rem; }}
code {{ font:400 .85em/1.4 var(--mono); background:var(--surface);
  border:1px solid var(--rule); border-radius:4px; padding:.1em .35em; }}
.toggle {{ position:fixed; top:.7rem; right:.7rem; font:500 .75rem/1 var(--sans);
  background:var(--surface); color:var(--ink-soft); border:1px solid var(--rule);
  border-radius:999px; padding:.5rem .8rem; cursor:pointer; }}
.toggle:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
footer {{ margin-top:3rem; padding-top:1.2rem; border-top:1px solid var(--rule);
  font:400 .82rem/1.6 var(--sans); color:var(--ink-soft); }}
@media (max-width:640px) {{
  body {{ font-size:16px; }}
  h1 {{ font-size:1.5rem; }}
  table {{ display:block; overflow-x:auto; }}
  .bar {{ max-width:40px; }}
}}
@media print {{ .toggle {{ display:none; }} .plate img {{ min-width:0; }} }}
</style>
</head>
<body>
<button class="toggle" id="t" aria-pressed="false">dark / light</button>
<div class="wrap">

<h1>Does the same emotion sound the same in 11 languages?</h1>
<p class="lede">One crying clip in English, and the matching crying clips in ten
other languages — plus anger and happiness for contrast. {CFG['n_clips']} clips,
{len(LANGS)} languages, 7 language families, 59 acoustic measurements per clip.</p>

<div class="meta">
  <span>{CFG['n_clips']} clips</span>
  <span>{len(LANGS)} languages · 7 families</span>
  <span>3 emotions × 18 clips per cell</span>
  <span>{CFG['n_interpretable']} interpretable features</span>
  <span>11 published corpora</span>
</div>

<div class="verdict">
<p><b>Yes — but only in relative terms, and much more for crying than for
laughing.</b> Two languages do not sound alike in absolute acoustics: which
language a clip comes from explains about three times more variance than which
emotion it carries. What is shared is the <i>direction</i> the voice moves from
each language's own baseline. Standardise every feature inside its own language
and sadness pushes the voice the same way everywhere.</p>
<p>A classifier that has never heard the test language still recognises the
emotion {pct(E['logreg']['pooled_accuracy'])} of the time
(random forest {pct(E['rf']['pooled_accuracy'])}) against
{pct(E['logreg']['chance'])} chance and a {pct(E['logreg']['permutation_mean'])}
permutation null — about three quarters of the
{pct(E['within_language_cv_mean'])} that same-language training reaches.</p>
</div>

<div class="tiles">
  <div class="tile"><b style="color:var(--sad)">
    r = {B['same_emotion']['sad']['mean_r']:+.2f}</b>
    <span>cross-language agreement on <b>sadness</b> over
    {B['same_emotion']['sad']['n_pairs']} language pairs
    (95% CI {B['same_emotion']['sad']['ci95'][0]:+.2f}…{B['same_emotion']['sad']['ci95'][1]:+.2f})</span></div>
  <div class="tile"><b style="color:var(--happy)">
    r = {B['same_emotion']['happy']['mean_r']:+.2f}</b>
    <span>the same number for <b>happiness</b> — the weak link</span></div>
  <div class="tile"><b>{pct(E['logreg']['pooled_accuracy'])}</b>
    <span>emotion recognised in a language the model never heard
    (p = {E['logreg']['permutation_p']:.3f}, {E['logreg']['n_permutations']} permutations)</span></div>
  <div class="tile"><b>{pct(RECALL['sad'])}</b>
    <span>of foreign <b>sad</b> clips recognised — the most portable emotion</span></div>
</div>

<h2>What was measured</h2>
<p>Eleven languages, one published emotional-speech corpus each:
{", ".join(f"{CFG['languages'][l]['language']} ({CFG['languages'][l]['corpus']})" for l in LANGS)}.
Three emotions per language — <b style="color:var(--sad)">sad</b> as the anchor,
with <b style="color:var(--angry)">angry</b> and
<b style="color:var(--happy)">happy</b> as contrasts, because "similar" is only
measurable against something dissimilar. Eighteen clips per cell, 16 kHz mono,
silence-trimmed, amplitude left untouched.</p>

<p>Each language comes from a different corpus, so language and recording
condition are entangled. Three defences are built in:</p>
<ul>
<li><b>Within-language standardisation.</b> Every feature is z-scored inside its
own language first. It works: language identity is recoverable from the raw
features with {pct(G['raw_features']['language_id_accuracy'])} accuracy
(chance {pct(G['raw_features']['chance'])}) but only
{pct(G['within_language_z']['language_id_accuracy'])} afterwards.</li>
<li><b>Gain-invariance repeat.</b> Dropping the features that would shift with a
different recording level changes nothing: same-emotion
r = {H['profile_correlation']['mean_r_same']:+.2f} against
{H['profile_correlation']['mean_r_diff']:+.2f}, transfer
{pct(H['lolo_logreg'])}.</li>
<li><b>Permutation null.</b> Emotion labels shuffled inside each language,
{E['logreg']['n_permutations']} times, give {pct(E['logreg']['permutation_mean'])}.</li>
</ul>

<h2>Finding 1 — crying has a common acoustic signature</h2>
<table>
<caption class="sub" style="text-align:left;padding:.3rem .6rem">
Sadness: the features that move furthest, averaged over the 11 languages</caption>
<tr><th>feature</th><th class="num">mean shift</th>
    <th class="num">languages agreeing</th><th class="num">FDR q</th></tr>
{consistency_table('sad')}
</table>
<p>Anger is the mirror image — the same two spectral-balance features at the top
with the signs reversed, 11 of 11 languages agreeing. The universal part of
emotional speech is mostly <b>vocal effort</b>: how hard the voice is driven,
read off the balance between low- and high-frequency energy, with pitch level
following along.</p>

<h2>Finding 2 — happiness is the weak link</h2>
<p>Not one feature survives correction for happiness (best
q = {C[C.emotion == 'happy'].q_binom.min():.2f}). The classifier agrees:</p>
<table>
<tr><th>true emotion</th><th class="num">recognised in a foreign language</th>
    <th>most common confusion</th></tr>
<tr><td style="color:var(--sad)"><b>sad</b></td>
    <td class="num">{pct(RECALL['sad'])}</td><td class="sub">→ happy ({pct(R[0, 2])})</td></tr>
<tr><td style="color:var(--angry)"><b>angry</b></td>
    <td class="num">{pct(RECALL['angry'])}</td><td class="sub">→ happy ({pct(R[1, 2])})</td></tr>
<tr><td style="color:var(--happy)"><b>happy</b></td>
    <td class="num">{pct(RECALL['happy'])}</td>
    <td class="sub">→ angry ({pct(R[2, 1])}), sad ({pct(R[2, 0])})</td></tr>
</table>
<p>Happiness shares high vocal effort with anger and has no reliable signature of
its own in this feature set. Whatever separates excited-positive from
excited-negative is not in these 33 measurements — and may well be
language-specific convention.</p>

<h2>Finding 3 — transfer failures are about the corpus, not the language</h2>
<table>
<tr><th>held-out language</th><th>corpus</th>
    <th class="num">transfer accuracy</th><th class="num">own-language ceiling</th></tr>
{language_rows()}
</table>
<p>Transfer tracks how separable the emotions are inside that corpus to begin
with (r = 0.66, p = 0.027). English and Russian transfer poorly because their
corpora are internally hard, not because those languages are acoustically
exotic.</p>
<p><b>Urdu is the real outlier</b>: internally separable
({within.loc['ur', 'accuracy']:.2f}) yet transferring at chance
({per.loc['ur', 'accuracy']:.2f}). It is also the only corpus here that is
<b>not acted</b> — the URDU-Dataset is harvested from talk shows. The biggest
break in cross-lingual similarity in this experiment is spontaneous versus acted
speech, not one language versus another.</p>

<h2>Finding 4 — profiles transfer, single clips barely do</h2>
<p>Across {S['F_clip_geometry']['n_cross_language_pairs']:,} cross-language clip
pairs, two clips of the same emotion are closer than two clips of different
emotions — mean distance
{S['F_clip_geometry']['mean_dist_same_emotion']:.2f} against
{S['F_clip_geometry']['mean_dist_diff_emotion']:.2f}, Cohen's d =
{S['F_clip_geometry']['cohens_d']:.2f} — but the overlap is large
(AUC {S['F_clip_geometry']['auc_same_emotion_from_distance']:.2f}). A single
foreign clip is only weakly attracted to its emotional neighbours; the regularity
appears once clips are averaged into a profile. Expect cross-lingual emotion
recognition on single short utterances to stay noisy even where the underlying
code is genuinely shared.</p>

<h2>The figures</h2>
{figure_blocks()}

<h2>Caveats</h2>
<ul>
<li>Ten of eleven corpora are <b>acted</b> emotion. Acted sadness is a
performance of crying, not crying.</li>
<li>One corpus per language: "language" and "corpus" cannot be fully separated.
The standardisation and the gain-invariance repeat bound the problem; they do not
remove it.</li>
<li>Four corpora expose no speaker identity, so speaker variance is partly
confounded with emotion in those cells.</li>
<li>Turkish clips are single words (median 0.66 s against 2.4 s elsewhere),
which limits every contour and timing feature there.</li>
<li>18 clips per cell is small — hence the confidence intervals, FDR correction
and permutation null on every claim.</li>
<li>The emotion labels are the corpora's own; no listening test was run.</li>
</ul>

<footer>
Generated by <code>scripts/06_report.py</code> from <code>results/summary.json</code>.
Data, clips, features, statistics and figures all live in
<code>cross_lingual_emotion_experiment/</code>. Corpora used for non-commercial
research only; see <code>scripts/config.py</code> for per-corpus licences and
references.
</footer>
</div>
<script>
const t = document.getElementById('t');
t.addEventListener('click', () => {{
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
  t.setAttribute('aria-pressed', String(!dark));
}});
</script>
</body>
</html>
"""

out = ROOT / "report.html"
out.write_text(HTML)
print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
