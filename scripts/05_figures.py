"""Figures for the cross-lingual emotion experiment.

Colour roles follow one validated system: categorical slots 1-3 (blue / orange /
aqua) for the three emotions, a blue->red diverging ramp with a neutral grey
midpoint for signed z-scores and correlations, and a single blue sequential ramp
for counts. Grid and axes stay recessive; every panel with two or more series
carries a legend.
"""
import json

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import EMOTIONS, FEATURES, FIGURES, RESULTS, SOURCES
from feature_defs import GROUPS, INTERPRETABLE, PRETTY

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
EMO_C = {"sad": "#2a78d6", "angry": "#eb6834", "happy": "#1baf7a"}
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#0d366b", "#256abf", "#9ec5f4", "#f0efec",
                 "#f0a6a5", "#e34948", "#8e1f1f"])
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab",
                 "#0d366b"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 130, "savefig.dpi": 130,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})

ORDERED = [f for g in ("pitch", "energy", "quality", "spectral", "temporal")
           for f in GROUPS[g]]


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name, flush=True)


def lang_label(l):
    return f"{l}  {SOURCES[l]['language']}"


# ---------------------------------------------------------------------------
def fig_profiles(prof, langs, cols):
    fig, axes = plt.subplots(len(EMOTIONS), 1,
                             figsize=(max(12, 0.42 * len(cols)),
                                      1.1 + 0.33 * len(langs) * len(EMOTIONS)))
    for ax, emo in zip(axes, EMOTIONS):
        sub = prof[prof.emotion == emo].set_index("lang").reindex(langs)
        M = sub[cols].to_numpy(float)
        M = np.vstack([M, np.nanmean(M, axis=0)])
        im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1.5, 1.5),
                       aspect="auto")
        ax.set_yticks(range(len(langs) + 1),
                      [lang_label(l) for l in langs] + ["MEAN"], fontsize=7.5)
        ax.get_yticklabels()[-1].set_color(INK)
        ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([PRETTY.get(c, c) for c in cols], rotation=90,
                           fontsize=6.5) if emo == EMOTIONS[-1] else \
            ax.set_xticklabels([])
        ax.axhline(len(langs) - 0.5, color=INK, lw=1.2)
        ax.set_ylim(len(langs) + 0.5, -0.5)
        ax.grid(False)
        ax.set_title(f"{emo.upper()} — deviation from each language's own norm",
                     color=EMO_C[emo])
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.016)
        cb.set_label("z within language", color=INK2, fontsize=7)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Acoustic emotion profiles, 11 languages × 3 emotions",
                 fontsize=12, fontweight="bold", y=1.005)
    save(fig, "fig1_emotion_profiles.png")


def fig_corr_matrices(langs):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, emo in zip(axes, EMOTIONS):
        M = pd.read_csv(RESULTS / f"B_corrmatrix_{emo}.csv", index_col=0)
        M = M.reindex(index=langs, columns=langs)
        A = M.to_numpy(float)
        off = A[~np.eye(len(langs), dtype=bool)]
        im = ax.imshow(A, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1))
        ax.set_xticks(range(len(langs)), langs, fontsize=8)
        ax.set_yticks(range(len(langs)), langs, fontsize=8)
        ax.grid(False)
        ax.set_title(f"{emo}  |  mean r = {np.nanmean(off):+.2f}",
                     color=EMO_C[emo])
        for i in range(len(langs)):
            for j in range(len(langs)):
                if i != j and np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:+.1f}".replace("0.", "."),
                            ha="center", va="center", fontsize=5.5,
                            color=INK if abs(A[i, j]) < 0.55 else "#ffffff")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label(
            "profile correlation r", fontsize=7, color=INK2)
    fig.suptitle("Do two languages express the same emotion the same way?",
                 fontsize=12, fontweight="bold")
    save(fig, "fig2_cross_language_correlation.png")


def fig_same_vs_diff():
    df = pd.read_csv(RESULTS / "B_profile_correlations.csv")
    same = df[df.emo_a == df.emo_b].r.to_numpy()
    diff = df[df.emo_a != df.emo_b].r.to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bins = np.linspace(-1, 1, 41)
    ax.hist(same, bins=bins, color=EMO_C["sad"], alpha=0.85,
            label=f"same emotion, different language (n={len(same)})")
    ax.hist(diff, bins=bins, color="#898781", alpha=0.6,
            label=f"different emotion, different language (n={len(diff)})")
    for v, c, t in ((same.mean(), EMO_C["sad"], "same"),
                    (diff.mean(), "#52514e", "diff")):
        ax.axvline(v, color=c, lw=2, ls="--")
        ax.text(v, ax.get_ylim()[1] * 0.96, f" mean {v:+.2f}", color=c,
                fontsize=8, fontweight="bold", ha="left", va="top")
    ax.set_xlabel("correlation between two languages' acoustic profiles")
    ax.set_ylabel("number of language pairs")
    ax.set_title("Same-emotion profiles agree across languages; "
                 "different-emotion profiles do not")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig3_same_vs_different_emotion.png")


def fig_consistency(C):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True)
    for ax, emo in zip(axes, EMOTIONS):
        sub = (C[C.emotion == emo].assign(a=lambda d: d.mean_z.abs())
               .sort_values("a", ascending=False).head(14)
               .sort_values("mean_z"))
        y = np.arange(len(sub))
        ax.hlines(y, sub.ci_lo, sub.ci_hi, color=EMO_C[emo], lw=2, alpha=0.5)
        ax.plot(sub.mean_z, y, "o", ms=8, color=EMO_C[emo],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axvline(0, color=AXIS, lw=1.2)
        ax.set_yticks(y, [PRETTY.get(f, f) for f in sub.feature], fontsize=8)
        ax.set_xlim(-1.25, 1.25)
        for yy, (k, n, q) in enumerate(zip(sub.n_agree, sub.n_langs,
                                           sub.q_binom)):
            ax.text(1.22, yy, f"{k}/{n}" + ("*" if q < 0.05 else ""),
                    va="center", ha="right", fontsize=7.5, color=INK2)
        ax.set_title(f"{emo.upper()}  —  mean across the 11 languages, 95% CI",
                     color=EMO_C[emo], fontsize=9.5)
        ax.set_xlabel("mean z within language")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Which acoustic changes are shared by all 11 languages?  "
                 "(k/n = languages agreeing on the direction, * = FDR q<0.05)",
                 fontsize=11, fontweight="bold")
    save(fig, "fig4_feature_consistency.png")


def fig_key_features(prof, langs, keys):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, f in zip(axes.ravel(), keys):
        for emo in EMOTIONS:
            sub = prof[prof.emotion == emo].set_index("lang").reindex(langs)
            ax.plot(range(len(langs)), sub[f].to_numpy(float), "o", ms=8,
                    color=EMO_C[emo], markeredgecolor=SURFACE,
                    markeredgewidth=1.4, label=emo)
        ax.axhline(0, color=AXIS, lw=1.2)
        ax.set_xticks(range(len(langs)), langs, fontsize=8)
        ax.set_title(PRETTY.get(f, f))
        ax.set_ylabel("z within language", fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(fontsize=8, ncols=3, loc="upper left")
    fig.suptitle("Per-language emotion means for six headline features "
                 "(each point is one language)", fontsize=12,
                 fontweight="bold")
    save(fig, "fig5_key_features_by_language.png")


def fig_lolo(summary, langs):
    per = pd.read_csv(RESULTS / "E_lolo_per_language_interpretable.csv")
    per = per[per.model == "logreg"].set_index("held_out").reindex(langs)
    pooled = summary["E_leave_one_language_out"]["interpretable"]["logreg"][
        "pooled_accuracy"]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    bars = ax.bar(range(len(langs)), per.accuracy, width=0.64,
                  color=EMO_C["sad"], label="accuracy on the held-out language")
    for b, v in zip(bars, per.accuracy):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", fontsize=8, color=INK2)
    ax.axhline(1 / 3, color="#e34948", lw=2, ls="--", label="chance (1/3)")
    ax.axhline(pooled, color=INK2, lw=1.6, ls=":",
               label=f"pooled accuracy {pooled:.2f}")
    ax.set_xticks(range(len(langs)), [lang_label(l) for l in langs],
                  rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("3-way emotion accuracy")
    ax.set_title("Train on 10 languages, test on the 11th "
                 "(logistic regression, 33 interpretable features)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "fig6_leave_one_language_out.png")


def fig_confusion():
    cm = pd.read_csv(RESULTS / "E_confusion_interpretable_logreg.csv",
                     index_col=0)
    N = cm.to_numpy(float)
    R = N / N.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(R, cmap=SEQ_BLUE, vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{R[i, j]:.0%}\n{int(N[i, j])}", ha="center",
                    va="center", fontsize=9,
                    color="#ffffff" if R[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(cm)), cm.columns)
    ax.set_yticks(range(len(cm)), cm.index)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true emotion")
    ax.grid(False)
    ax.set_title("Pooled leave-one-language-out confusion")
    fig.colorbar(im, ax=ax, fraction=0.045).set_label("row share",
                                                      fontsize=8, color=INK2)
    save(fig, "fig7_confusion_matrix.png")


def fig_embedding(z, cols, langs):
    X = StandardScaler().fit_transform(z[cols].to_numpy(float))
    P = PCA(n_components=2, random_state=0).fit(X)
    Y = P.transform(X)
    fig = plt.figure(figsize=(14.5, 5.6))
    ax = fig.add_subplot(1, 2, 1)
    for emo in EMOTIONS:
        m = (z.emotion == emo).to_numpy()
        ax.plot(Y[m, 0], Y[m, 1], "o", ms=5.5, alpha=0.8, color=EMO_C[emo],
                markeredgecolor=SURFACE, markeredgewidth=0.6, label=emo)
    ax.set_xlabel(f"PC1 ({P.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({P.explained_variance_ratio_[1]:.0%})")
    ax.set_title("All clips from 11 languages, coloured by emotion")
    ax.legend(fontsize=9)
    gs = fig.add_gridspec(3, 8, left=0.50, right=0.99, hspace=0.45, wspace=0.25)
    for k, lg in enumerate(langs):
        a = fig.add_subplot(gs[k // 4, 4 + k % 4])
        m = (z.lang == lg).to_numpy()
        a.plot(Y[:, 0], Y[:, 1], "o", ms=1.6, color="#dedcd4")
        for emo in EMOTIONS:
            mm = m & (z.emotion == emo).to_numpy()
            a.plot(Y[mm, 0], Y[mm, 1], "o", ms=4, color=EMO_C[emo])
        a.set_title(lang_label(lg), fontsize=7.5)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.text(0.76, 1.0, "the same space, one language at a time",
             ha="center", fontsize=10, fontweight="bold")
    fig.savefig(FIGURES / "fig8_pca_embedding.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig8_pca_embedding.png", flush=True)


def fig_variance(D):
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.plot(D.eta2_language, D.eta2_emotion, "o", ms=8,
            color=EMO_C["sad"], markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="one acoustic feature")
    lim = max(D.eta2_language.max(), D.eta2_emotion.max()) * 1.15
    ax.plot([0, lim], [0, lim], color=AXIS, lw=1.2, ls="--")
    ax.text(lim * 0.97, lim * 0.9, "equal split", color=MUTED, fontsize=8,
            ha="right")
    lab = pd.concat([D.nlargest(6, "eta2_emotion"),
                     D.nlargest(4, "eta2_language")]).drop_duplicates("feature")
    for r in lab.itertuples():
        ax.annotate(PRETTY.get(r.feature, r.feature),
                    (r.eta2_language, r.eta2_emotion), fontsize=7.5,
                    color=INK2, xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("variance explained by language (η²)")
    ax.set_ylabel("variance explained by emotion (η²)")
    ax.set_title("Is a feature driven by which emotion it is, or which "
                 "language it is?")
    ax.legend(fontsize=8)
    save(fig, "fig9_variance_decomposition.png")


def main():
    z = pd.read_csv(FEATURES / "features_zlang.csv")
    prof = pd.read_csv(RESULTS / "A_emotion_profiles_zlang.csv")
    C = pd.read_csv(RESULTS / "C_feature_consistency.csv")
    D = pd.read_csv(RESULTS / "D_variance_decomposition.csv")
    summary = json.load(open(RESULTS / "summary.json"))
    langs = sorted(z.lang.unique())
    cols = [c for c in ORDERED if c in z.columns]

    fig_profiles(prof, langs, cols)
    fig_corr_matrices(langs)
    fig_same_vs_diff()
    fig_consistency(C)
    keys = [f for f in ["f0_mean_st", "f0_std_st", "rms_range_db", "hnr_db",
                        "syl_rate_hz", "alpha_ratio_db"] if f in z.columns]
    fig_key_features(prof, langs, keys)
    fig_lolo(summary, langs)
    fig_confusion()
    fig_embedding(z, cols, langs)
    fig_variance(D)
    print("figures in", FIGURES)


if __name__ == "__main__":
    main()
