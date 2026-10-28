# Does the same emotion sound the same in 11 languages?

**Experiment run:** 2026-10-27/28 · 594 clips · 11 languages · 7 language families ·
3 emotions · 59 acoustic features · all data, code and results in
`cross_lingual_emotion_experiment/`.

---

## The short answer

**Yes — but only in a specific sense, and much more for crying than for laughing.**

Two languages do **not** sound alike in absolute terms. Across the 33 interpretable
features, *which language* a clip comes from explains on average **three times more
variance (η² = 0.20)** than *which emotion* it carries (η² = 0.07). A sad Japanese
clip and a sad German clip are not near each other in raw acoustic space.

What **is** shared is the *direction of the change* relative to each language's own
neutral baseline. Once every feature is standardised inside its own language,
sadness pushes the voice the same way in all 11 languages:

| | mean cross-language profile correlation | language pairs agreeing |
|---|---|---|
| **sad** (the anchor emotion) | **r = +0.40** (95% CI +0.34…+0.45) | 53 of 55 |
| **angry** | r = +0.34 (+0.28…+0.40) | 52 of 55 |
| **happy** | r = +0.15 (+0.05…+0.23) | 39 of 55 |
| *different* emotions, different languages | r = −0.14 | — |

Same-emotion pairs correlate far more than different-emotion pairs
(Mann-Whitney p < 1e-30, Cliff's δ = 0.66). And a classifier that has **never heard
the test language** still recognises the emotion **57.6 %** of the time (random
forest 62.3 %) against a 33.3 % chance level and a 33.5 % permutation null
(p = 0.007). Within-language training reaches 74.4 %, so roughly **three quarters
of what is learnable about these emotions transfers across languages**.

---

## What was measured

| | |
|---|---|
| Languages | English, German, Greek, Polish, Russian, Persian, Urdu, Bangla, Turkish, Mandarin, Japanese |
| Families | Germanic, Hellenic, Slavic, Iranian, Indo-Aryan, Turkic, Sinitic, Japonic |
| Emotions | **sad** (anchor), **angry**, **happy** — 3 per language, as specified |
| Clips | 18 per language × emotion = 594 clips, 33.2 min of audio |
| Corpora | RAVDESS, EmoDB, AESDD, nEMO, RESD, ShEMO, URDU-Dataset, SUBESCO, TurEV-DB, CASIA, JVNV — one per language |
| Features | 33 interpretable (F0 in semitones, loudness in dB, jitter/shimmer/HNR, spectral balance, timing) + 26 MFCC statistics |

The **anchor emotion is sadness** — grief/crying-type vocal expression — exactly as
asked: one sad clip in English, and the matching sad clips in ten other languages.
Anger and happiness are the two contrast emotions that make "similar" measurable:
without them, every pair of sad clips would look similar simply because all speech
is similar.

Each language comes from a different corpus, so language and recording condition are
entangled. Three defences are built into the analysis:

1. **Within-language standardisation.** Every feature is z-scored inside its own
   language before anything is compared, removing each corpus' gain, microphone,
   speakers and phonology. The check works: language identity is recoverable from
   the raw features with **81.8 %** accuracy (chance 9.1 %) but only **1.5 %** after
   standardisation.
2. **Gain-invariance repeat.** Dropping the three features that would shift with a
   different recording level changes nothing: same-emotion r = +0.32 vs −0.15
   (p = 1.2e-33), transfer accuracy 57.1 %.
3. **Permutation null.** Emotion labels shuffled within each language, 150 times,
   give 33.5 % — the real 57.6 % is not an artefact of the pipeline.

---

## Finding 1 — crying has a common acoustic signature

Features where **all 11 languages** move the same direction for sadness
(FDR q = 0.005), with the mean shift in within-language standard deviations:

| feature | shift | languages agreeing | what it means |
|---|---|---|---|
| Hammarberg index | **+0.81 σ** | 11/11 | energy concentrated below 2 kHz |
| alpha ratio (1–5 k / 50–1 k) | **−0.78 σ** | 11/11 | little high-frequency energy |
| spectral flux | −0.50 σ | 11/11 | the spectrum moves slowly |
| median / mean F0 | −0.57 σ | 10/11 | lower pitch |
| 85 % spectral roll-off | −0.54 σ | 10/11 | darker timbre |
| voiced frame ratio | −0.48 σ | 10/11 | more breaks in voicing |

Anger is the mirror image, with the same two features at the top and the signs
reversed: alpha ratio **+0.66 σ** (11/11), Hammarberg **−0.60 σ** (11/11), spectral
centroid and roll-off both +0.42 σ (11/11), F0 +0.38 σ (9/11).

So the universal part of emotional speech is mostly **vocal effort**: how hard the
voice is driven, read off the balance between low and high frequency energy, with
pitch level following it. That single axis is visible in the PCA
(`fig8`): PC1 separates sad clips from angry and happy clips across all 11
languages at once.

## Finding 2 — happiness is the weak link

Not one feature survives multiple-comparison correction for happiness (best
q = 0.10). The best candidates — more voiced frames, more spectral flux, less
jitter and shimmer, slightly higher and more often rising F0 — point the same way in
8–10 of 11 languages but with shifts of only ~0.2 σ.

The classifier shows the same thing. Recall from the transfer test:

| true emotion | recall | most common confusion |
|---|---|---|
| **sad** | **71 %** | → happy (20 %) |
| angry | 58 % | → happy (30 %) |
| happy | 44 % | → angry (31 %), sad (25 %) |

Sadness is the most portable emotion across languages; happiness sits between anger
and sadness on the effort axis and has no reliable signature of its own in this
feature set. Anything beyond arousal — the distinction between excited-positive and
excited-negative — appears to be carried by something these 33 features do not
capture, and plausibly by language-specific convention.

## Finding 3 — transfer failures are about the corpus, not the language

Per-language transfer (train on the other 10, test on this one, logistic regression):

| | bn | de | el | zh | ja | fa | pl | tr | en | ru | ur |
|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | .74 | .70 | .67 | .69 | .61 | .61 | .59 | .54 | .43 | .41 | **.35** |

Transfer accuracy tracks how separable the emotions are *inside* that corpus to
begin with (r = 0.66, p = 0.027). English and Russian transfer poorly because their
corpora are internally hard (within-language ceilings of 0.56 and 0.50), not because
English or Russian is acoustically exotic.

**Urdu is the one real outlier**: it is internally separable (0.74) yet transfers at
chance (0.35). It is also the only corpus in the set that is **not acted** — the
URDU-Dataset is harvested from talk shows. The single biggest break in cross-lingual
similarity here is spontaneous vs. acted speech, not one language vs. another.

Turkish is a second caveat in the other direction: TurEV-DB is single-word
utterances (median 0.66 s against 2.4 s elsewhere), which limits every pitch-contour
and timing feature, and it still transfers at 0.54.

## Finding 4 — the effect is real but modest at the level of a single clip

Over 160,380 cross-language clip pairs, two clips of the same emotion are closer
than two clips of different emotions — mean distance 7.58 vs 8.04 (Cohen's d = 0.24,
p < 1e-30) — but the overlap is large (AUC 0.57, silhouette by emotion 0.01). A
single foreign clip is only weakly attracted to its emotional neighbours; the
regularity appears when clips are averaged into a profile.

Practical reading: **profiles transfer, individual clips do not.** Expect
cross-lingual emotion recognition on single short utterances to be noisy even when
the underlying code is genuinely shared.

---

## Answering the original question directly

> *Take one crying clip in English, find crying clips in nine other languages — are
> their energy, pitch and other qualities similar?*

- **Pitch:** yes, in relative terms. Crying lowers F0 by ~0.57 σ below that
  language's own norm in 10 of 11 languages. Absolute F0 is not comparable — the
  mean across languages ranges over roughly an octave because of speaker and corpus
  differences.
- **Energy:** the *distribution* of energy is the single most universal cue
  (Hammarberg index, alpha ratio: 11/11 languages, the two features where emotion
  beats language in explained variance). Absolute loudness is not comparable at all,
  because each corpus was recorded and mastered at its own level — which is why the
  analysis works in dB-relative terms and repeats itself without the level-sensitive
  features.
- **Other qualities:** voice quality (jitter, shimmer, HNR) is far less consistent
  than textbook descriptions of sadness suggest — it never reaches the top of the
  consistency ranking for any of the three emotions here.
- **Overall:** similar enough that a model trained on ten languages recognises the
  eleventh well above chance; different enough that language explains three times
  more acoustic variance than emotion does.

---

## Figures

| file | shows |
|---|---|
| `fig1_emotion_profiles.png` | every language × emotion profile, all 33 features |
| `fig2_cross_language_correlation.png` | language × language agreement, per emotion |
| `fig3_same_vs_different_emotion.png` | the same-vs-different correlation distributions |
| `fig4_feature_consistency.png` | which features are universal, with CIs and agreement counts |
| `fig5_key_features_by_language.png` | six headline features, one point per language |
| `fig6_leave_one_language_out.png` | transfer accuracy per held-out language |
| `fig7_confusion_matrix.png` | pooled transfer confusion (sad best, happy worst) |
| `fig8_pca_embedding.png` | all clips in one space, and one language at a time |
| `fig9_variance_decomposition.png` | emotion η² vs language η² per feature |

## Caveats

- Ten of eleven corpora are **acted** emotion. Acted sadness is a performance of
  crying, not crying.
- One corpus per language: "language" and "corpus" cannot be fully separated. The
  within-language standardisation and the gain-invariance repeat bound the problem;
  they do not remove it.
- Four corpora expose no speaker identity, so speaker variance is partly confounded
  with emotion in those cells.
- 18 clips per cell is small. Every claim above is reported with a confidence
  interval, an FDR-corrected p-value or a permutation null for that reason.
- "Sad / angry / happy" are corpus labels, not verified percepts; no listening test
  was run.

Raw numbers for everything quoted here: `results/summary.json` and the per-analysis
CSVs beside it.
