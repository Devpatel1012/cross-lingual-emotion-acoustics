"""Cross-lingual similarity analysis.

Core idea: every feature is standardised *within language* (z_lang), which
removes each corpus' recording gain, speaker set and phonological baseline.
What is left is how a speaker deviates from their own language's norm when
expressing an emotion. If emotions are acoustically similar across languages,
those deviation profiles must line up.

Analyses
  A  emotion profiles per (language, emotion)
  B  cross-language profile correlation, same vs different emotion
  C  per-feature direction consistency across languages (binomial test)
  D  variance decomposition: emotion vs language (eta^2)
  E  leave-one-language-out emotion classification (the generalisation test)
  F  clip-level neighbourhood structure (silhouette, same-emotion AUC)
  G  confound check: how much language identity survives normalisation

Outputs: results/*.csv and results/summary.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import EMOTIONS, FEATURES, RESULTS, SOURCES
from feature_defs import CEPSTRAL_F, GAIN_SENSITIVE, GROUPS, INTERPRETABLE

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20261027)
ANCHOR = "sad"


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
def load():
    df = pd.read_csv(FEATURES / "features.csv")
    feats = [f for f in INTERPRETABLE + CEPSTRAL_F if f in df.columns]
    nan_rate = df[feats].isna().mean()
    keep = [f for f in feats if nan_rate[f] <= 0.2]
    dropped = [f for f in feats if f not in keep]
    if dropped:
        log(f"dropped {len(dropped)} features with >20% NaN: {dropped}")
    # median-impute the remaining gaps inside each language
    df[keep] = df.groupby("lang")[keep].transform(lambda s: s.fillna(s.median()))
    df[keep] = df[keep].fillna(df[keep].median())
    return df, keep


def zscore_within(df, cols, by):
    out = df.copy()
    g = out.groupby(by)[cols]
    out[cols] = (out[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    out[cols] = out[cols].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
def analysis_A(z, cols):
    prof = z.groupby(["lang", "emotion"])[cols].mean().reset_index()
    prof.to_csv(RESULTS / "A_emotion_profiles_zlang.csv", index=False)
    return prof


def analysis_B(prof, cols, langs, tag=""):
    """Correlate emotion profiles between languages."""
    P = {(r.lang, r.emotion): prof.loc[i, cols].to_numpy(float)
         for i, r in zip(prof.index, prof.itertuples())}
    same, diff, rows = {e: [] for e in EMOTIONS}, [], []
    for i, la in enumerate(langs):
        for lb in langs[i + 1:]:
            for ea in EMOTIONS:
                for eb in EMOTIONS:
                    if (la, ea) not in P or (lb, eb) not in P:
                        continue
                    r = stats.pearsonr(P[(la, ea)], P[(lb, eb)])[0]
                    rows.append(dict(lang_a=la, lang_b=lb, emo_a=ea, emo_b=eb,
                                     r=r))
                    if ea == eb:
                        same[ea].append(r)
                    else:
                        diff.append(r)
    pd.DataFrame(rows).to_csv(
        RESULTS / f"B_profile_correlations{tag}.csv", index=False)

    out = {"same_emotion": {}, "different_emotion": {}}
    for e in EMOTIONS:
        v = np.array(same[e])
        ci = stats.bootstrap((v,), np.mean, n_resamples=4000,
                             random_state=1).confidence_interval
        out["same_emotion"][e] = dict(
            n_pairs=len(v), mean_r=float(v.mean()), median_r=float(np.median(v)),
            ci95=[float(ci.low), float(ci.high)],
            frac_positive=float((v > 0).mean()))
    d = np.array(diff)
    out["different_emotion"] = dict(n_pairs=len(d), mean_r=float(d.mean()),
                                    median_r=float(np.median(d)))
    allsame = np.concatenate([np.array(same[e]) for e in EMOTIONS])
    u, p = stats.mannwhitneyu(allsame, d, alternative="greater")
    out["same_vs_different"] = dict(
        mean_r_same=float(allsame.mean()), mean_r_diff=float(d.mean()),
        mannwhitney_U=float(u), p_value=float(p),
        cliffs_delta=float(2 * u / (len(allsame) * len(d)) - 1))
    # per-emotion matrices for plotting
    for e in EMOTIONS:
        M = pd.DataFrame(np.nan, index=langs, columns=langs, dtype=float)
        for la in langs:
            for lb in langs:
                if (la, e) in P and (lb, e) in P:
                    M.loc[la, lb] = stats.pearsonr(P[(la, e)], P[(lb, e)])[0]
        M.to_csv(RESULTS / f"B_corrmatrix_{e}{tag}.csv")
    return out


def analysis_C(prof, cols, langs):
    """Per-feature direction consistency across languages."""
    rows = []
    for e in EMOTIONS:
        sub = prof[prof.emotion == e]
        for f in cols:
            v = sub[f].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size < 4:
                continue
            maj = 1 if (v > 0).sum() >= (v < 0).sum() else -1
            k = int((np.sign(v) == maj).sum())
            p = stats.binomtest(k, v.size, 0.5, alternative="greater").pvalue
            t, pt = stats.ttest_1samp(v, 0.0)
            sem = v.std(ddof=1) / np.sqrt(v.size)
            rows.append(dict(emotion=e, feature=f, group=group_of(f),
                             mean_z=v.mean(), sd_z=v.std(ddof=1),
                             ci_lo=v.mean() - 1.96 * sem,
                             ci_hi=v.mean() + 1.96 * sem,
                             n_langs=v.size, n_agree=k,
                             agree_frac=k / v.size, direction=maj,
                             p_binom=p, t=t, p_ttest=pt))
    df = pd.DataFrame(rows)
    # Benjamini-Hochberg within each emotion
    df["q_binom"] = df.groupby("emotion").p_binom.transform(bh)
    df["q_ttest"] = df.groupby("emotion").p_ttest.transform(bh)
    df.sort_values(["emotion", "p_binom", "p_ttest"]).to_csv(
        RESULTS / "C_feature_consistency.csv", index=False)
    return df


def bh(p):
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[idx] * n / rank)
        q[idx] = prev
    return q


def group_of(f):
    for g, fs in GROUPS.items():
        if f in fs:
            return g
    return "other"


def analysis_D(df, cols):
    """Variance decomposition: how much of each feature is emotion vs language."""
    rows = []
    for f in cols:
        y = df[f].to_numpy(float)
        y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
        sst = np.nansum((y - np.nanmean(y)) ** 2)
        eta = {}
        for factor in ("emotion", "lang"):
            ss = 0.0
            for _, idx in df.groupby(factor).groups.items():
                g = y[df.index.get_indexer(idx)]
                ss += g.size * (np.nanmean(g) - np.nanmean(y)) ** 2
            eta[factor] = ss / sst if sst > 0 else np.nan
        Fe, pe = stats.f_oneway(*[y[(df.emotion == e).to_numpy()] for e in EMOTIONS])
        rows.append(dict(feature=f, group=group_of(f), eta2_emotion=eta["emotion"],
                         eta2_language=eta["lang"],
                         ratio=eta["emotion"] / (eta["lang"] + 1e-12),
                         F_emotion=Fe, p_emotion=pe))
    out = pd.DataFrame(rows).sort_values("eta2_emotion", ascending=False)
    out.to_csv(RESULTS / "D_variance_decomposition.csv", index=False)
    return out


def models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(
            C=0.3, max_iter=4000)),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=2.0, gamma="scale")),
        "rf": RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                     random_state=0, n_jobs=-1),
    }


def analysis_E(z, cols, label="interpretable", n_perm=150):
    """Leave-one-language-out emotion recognition."""
    y = z.emotion.to_numpy()
    X = z[cols].to_numpy(float)
    langs = sorted(z.lang.unique())
    res, per_lang = {}, []
    for name, mdl in models().items():
        preds = np.empty(len(y), dtype=object)
        for lg in langs:
            te = (z.lang == lg).to_numpy()
            m = models()[name]
            m.fit(X[~te], y[~te])
            preds[te] = m.predict(X[te])
            per_lang.append(dict(feature_set=label, model=name, held_out=lg,
                                 n=int(te.sum()),
                                 accuracy=float((preds[te] == y[te]).mean()),
                                 macro_f1=float(f1_score(y[te], preds[te],
                                                         average="macro"))))
        acc = float((preds == y).mean())
        # permutation test: shuffle emotion labels within each language.
        # Only the primary feature set / linear model is permuted: a null fit
        # carries no signal and converges slowly, so this is the expensive part.
        null = []
        perm_model = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.3, max_iter=600, tol=1e-3))
        reps = n_perm if (name == "logreg" and label == "interpretable") else 0
        for _ in range(reps):
            ys = y.copy()
            for lg in langs:
                m = (z.lang == lg).to_numpy()
                ys[m] = RNG.permutation(ys[m])
            p = np.empty(len(ys), dtype=object)
            for lg in langs:
                te = (z.lang == lg).to_numpy()
                mm = clone(perm_model)
                mm.fit(X[~te], ys[~te])
                p[te] = mm.predict(X[te])
            null.append((p == ys).mean())
        cm = confusion_matrix(y, preds, labels=EMOTIONS)
        res[name] = dict(
            pooled_accuracy=acc,
            pooled_macro_f1=float(f1_score(y, preds, average="macro")),
            chance=1 / len(EMOTIONS),
            confusion_matrix=cm.tolist(), labels=EMOTIONS,
            permutation_mean=float(np.mean(null)) if null else None,
            permutation_p=float((np.sum(np.array(null) >= acc) + 1) /
                                (len(null) + 1)) if null else None,
            n_permutations=len(null))
        pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_csv(
            RESULTS / f"E_confusion_{label}_{name}.csv")
        log(f"  LOLO[{label}/{name}] acc={acc:.3f} "
            f"f1={res[name]['pooled_macro_f1']:.3f}")
    # within-language ceiling for reference
    within = []
    for lg in langs:
        sub = z[z.lang == lg]
        if sub.emotion.nunique() < 2 or len(sub) < 15:
            continue
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        s = cross_val_score(models()["logreg"], sub[cols].to_numpy(float),
                            sub.emotion.to_numpy(), cv=cv)
        within.append(dict(lang=lg, accuracy=float(s.mean())))
    pd.DataFrame(per_lang).to_csv(RESULTS / f"E_lolo_per_language_{label}.csv",
                                  index=False)
    pd.DataFrame(within).to_csv(RESULTS / f"E_within_language_cv_{label}.csv",
                                index=False)
    res["within_language_cv_mean"] = float(np.mean([w["accuracy"] for w in within]))
    res["per_language"] = per_lang
    return res


def analysis_F(z, cols):
    """Clip-level geometry: is a sad clip closer to foreign sad clips?"""
    X = z[cols].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    emo = z.emotion.to_numpy()
    lang = z.lang.to_numpy()
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    cross = lang[:, None] != lang[None, :]
    same_e = emo[:, None] == emo[None, :]
    iu = np.triu_indices(len(X), 1)
    mask = cross[iu]
    dd, ss = d[iu][mask], same_e[iu][mask]
    auc = roc_auc_score(ss.astype(int), -dd)
    out = dict(
        n_cross_language_pairs=int(mask.sum()),
        mean_dist_same_emotion=float(dd[ss].mean()),
        mean_dist_diff_emotion=float(dd[~ss].mean()),
        cohens_d=float((dd[~ss].mean() - dd[ss].mean()) /
                       np.sqrt((dd[ss].var() + dd[~ss].var()) / 2)),
        auc_same_emotion_from_distance=float(auc),
        silhouette_by_emotion=float(silhouette_score(X, emo)),
        silhouette_by_language=float(silhouette_score(X, lang)),
    )
    u, p = stats.mannwhitneyu(dd[ss], dd[~ss], alternative="less")
    out["mannwhitney_p"] = float(p)
    return out


def analysis_G(df, z, cols):
    """Confound check: language identifiability before/after normalisation."""
    out = {}
    for name, frame in (("raw_features", df), ("within_language_z", z)):
        X = frame[cols].to_numpy(float)
        y = frame.lang.to_numpy()
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        s = cross_val_score(models()["logreg"], X, y, cv=cv)
        out[name] = dict(language_id_accuracy=float(s.mean()),
                         chance=1 / len(np.unique(y)))
    return out


# ---------------------------------------------------------------------------
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df, cols = load()
    cell = df.groupby(["lang", "emotion"]).size().unstack().reindex(
        columns=EMOTIONS)
    thin = cell[(cell.fillna(0) < 5).any(axis=1)].index.tolist()
    if thin:
        log(f"excluding languages with under-filled cells: {thin}")
        log(cell.loc[thin].to_string())
        df = df[~df.lang.isin(thin)].reset_index(drop=True)
    inter = [c for c in cols if c in INTERPRETABLE]
    langs = sorted(df.lang.unique())
    log(f"{len(df)} clips, {len(langs)} languages, {len(cols)} features "
        f"({len(inter)} interpretable)")

    z = zscore_within(df, cols, "lang")
    z.to_csv(FEATURES / "features_zlang.csv", index=False)

    log("A/B: emotion profiles and cross-language correlation")
    prof = analysis_A(z, cols)
    B = analysis_B(prof, inter, langs)
    log(f"  same-emotion mean r = "
        f"{B['same_vs_different']['mean_r_same']:.3f} vs different-emotion "
        f"{B['same_vs_different']['mean_r_diff']:.3f} "
        f"(p={B['same_vs_different']['p_value']:.2e})")

    log("C: per-feature consistency")
    C = analysis_C(prof, inter, langs)
    log("D: variance decomposition")
    D = analysis_D(df, inter)

    log("E: leave-one-language-out classification")
    E_i = analysis_E(z, inter, "interpretable")
    E_a = analysis_E(z, cols, "all_incl_mfcc")
    gain_free = [c for c in inter if c not in GAIN_SENSITIVE]
    E_g = analysis_E(z, gain_free, "gain_invariant")
    B_g = analysis_B(prof, gain_free, langs, tag="_gain_invariant")

    log("F/G: geometry and confound checks")
    F = analysis_F(z, inter)
    G = analysis_G(df, z, inter)

    top = {}
    for e in EMOTIONS:
        sub = C[(C.emotion == e)].sort_values("p_binom").head(12)
        top[e] = sub[["feature", "group", "mean_z", "agree_frac", "n_agree",
                      "n_langs", "direction", "p_binom", "q_binom"]].to_dict("records")

    summary = dict(
        config=dict(languages={l: dict(language=SOURCES[l]["language"],
                                       family=SOURCES[l]["family"],
                                       corpus=SOURCES[l]["corpus"])
                               for l in langs},
                    emotions=EMOTIONS, anchor_emotion=ANCHOR,
                    n_clips=int(len(df)),
                    clips_per_cell=df.groupby(["lang", "emotion"]).size()
                    .unstack().to_dict(),
                    n_features=len(cols), n_interpretable=len(inter)),
        B_profile_correlation=B,
        C_top_consistent_features=top,
        D_variance=dict(
            emotion_dominant=D[D.ratio > 1].feature.tolist(),
            language_dominant=D[D.ratio <= 1].feature.tolist(),
            mean_eta2_emotion=float(D.eta2_emotion.mean()),
            mean_eta2_language=float(D.eta2_language.mean()),
            top10=D.head(10)[["feature", "eta2_emotion", "eta2_language"]]
            .to_dict("records")),
        E_leave_one_language_out=dict(interpretable=E_i, all_incl_mfcc=E_a),
        F_clip_geometry=F,
        G_language_confound=G,
        H_robustness_gain_invariant=dict(
            n_features=len(gain_free),
            excluded=[c for c in inter if c in GAIN_SENSITIVE],
            profile_correlation=B_g["same_vs_different"],
            lolo_logreg=E_g["logreg"]["pooled_accuracy"],
            lolo_logreg_f1=E_g["logreg"]["pooled_macro_f1"]),
    )
    with open(RESULTS / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    log(f"\nwrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
