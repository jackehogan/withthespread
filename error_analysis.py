"""
Error analysis for backtest predictions.

Loads backtest CSVs and produces a PDF report characterising where the model
fails.  Run after backtest.py has generated the prediction and feature CSVs.

Sections
--------
1.  Residual distribution
2.  Residual vs predspread_diff  (did the model's confidence correlate with error?)
3.  High-confidence busts        (wrong predictions with |coverprob - 0.5| > 0.15)
4.  Team-level bias              (mean residual per team, ranked)
5.  Period curve                 (accuracy + MAE over time in the season)
6.  Spread segmentation          (if spread column available)
7.  Home / Away / B2B split      (if those columns available)
8.  Residual vs each feature     (if features CSV joinable)
9.  Sigma_diff calibration       (did larger sigma -> harder games?)

Usage
-----
    python error_analysis.py --season 2023
    python error_analysis.py --season 2024
    python error_analysis.py --season 2023 2024   (combined)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(season: int, prefix: str = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions and features CSVs for a season."""
    p_path = Path(prefix) / f"backtest_nba_{season}_predictions.csv"
    f_path = Path(prefix) / f"backtest_nba_{season}_features.csv"

    preds = pd.read_csv(p_path)
    preds["residual"] = preds["predspread"] - preds["spreadscore"]
    preds["predicted_cover"] = (preds["coverprob"] >= 0.5).astype(int)
    preds["correct"] = (preds["predicted_cover"] == preds["covered"]).astype(int)
    preds["confidence"] = (preds["coverprob"] - 0.5).abs()
    preds["season"] = season

    feats = pd.read_csv(f_path) if f_path.exists() else pd.DataFrame()
    return preds, feats


def _add_title_page(pdf: PdfPages, title: str) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.55, title, ha="center", va="center", fontsize=22, fontweight="bold")
    fig.text(0.5, 0.45, "backtest error analysis", ha="center", va="center", fontsize=14, color="grey")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _section_page(pdf: PdfPages, text: str) -> None:
    fig = plt.figure(figsize=(11, 2))
    fig.text(0.05, 0.5, text, ha="left", va="center", fontsize=16, fontweight="bold", color="#333333")
    fig.patch.set_facecolor("#f0f0f0")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _savefig(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 1: Residual distribution
# ---------------------------------------------------------------------------

def plot_residual_distribution(df: pd.DataFrame, pdf: PdfPages) -> None:
    _section_page(pdf, "1.  Residual distribution  (predspread - actual spreadscore)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    ax = axes[0]
    ax.hist(df["residual"], bins=40, edgecolor="white", color="#4C72B0", alpha=0.85)
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.axvline(df["residual"].mean(), color="red", lw=1.5, ls="-", label=f"mean={df['residual'].mean():+.2f}")
    ax.set_xlabel("Residual (predspread - spreadscore)")
    ax.set_ylabel("Count")
    ax.set_title("Residual histogram")
    ax.legend()

    # Summary stats text
    ax = axes[1]
    ax.axis("off")
    stats = {
        "N predictions": len(df),
        "Seasons": ", ".join(str(s) for s in sorted(df["season"].unique())),
        "Mean residual (bias)": f"{df['residual'].mean():+.3f}",
        "Std residual": f"{df['residual'].std():.3f}",
        "Median residual": f"{df['residual'].median():+.3f}",
        "|err| <= 5": f"{(df['residual'].abs() <= 5).mean():.1%}",
        "|err| <= 10": f"{(df['residual'].abs() <= 10).mean():.1%}",
        "|err| > 20": f"{(df['residual'].abs() > 20).mean():.1%}",
        "Overall accuracy": f"{df['correct'].mean():.3f}",
        "Base cover rate": f"{df['covered'].mean():.3f}",
    }
    y = 0.95
    for k, v in stats.items():
        ax.text(0.05, y, k, fontsize=11, transform=ax.transAxes, va="top")
        ax.text(0.65, y, v, fontsize=11, transform=ax.transAxes, va="top", fontweight="bold")
        y -= 0.09
    ax.set_title("Summary statistics")

    fig.tight_layout()
    _savefig(pdf, fig)


# ---------------------------------------------------------------------------
# Section 2: Residual vs predspread_diff
# ---------------------------------------------------------------------------

def plot_residual_vs_confidence(df: pd.DataFrame, pdf: PdfPages) -> None:
    _section_page(pdf, "2.  Residual vs model confidence  (spread_diff and coverprob)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Residual vs spread_diff
    ax = axes[0]
    ax.scatter(df["spread_diff"], df["residual"], alpha=0.25, s=12, color="#4C72B0")
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    corr = df["spread_diff"].corr(df["residual"])
    ax.set_xlabel("spread_diff  (predspread_A - predspread_B)")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residual vs spread_diff  (corr={corr:.3f})")

    # Mean abs residual by confidence decile
    ax = axes[1]
    df["conf_decile"] = pd.qcut(df["confidence"], q=10, duplicates="drop")
    grp = df.groupby("conf_decile", observed=True).agg(
        mean_abs_res=("residual", lambda x: x.abs().mean()),
        acc=("correct", "mean"),
        n=("correct", "size"),
    )
    x = range(len(grp))
    ax2 = ax.twinx()
    ax.bar(x, grp["mean_abs_res"], color="#4C72B0", alpha=0.7, label="MAE")
    ax2.plot(x, grp["acc"], "o-", color="red", lw=2, label="Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i.right)[:5] for i in grp.index], rotation=45, fontsize=8)
    ax.set_xlabel("Confidence decile (|coverprob - 0.5|)")
    ax.set_ylabel("Mean |residual|", color="#4C72B0")
    ax2.set_ylabel("Accuracy", color="red")
    ax.set_title("MAE and accuracy by confidence decile")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    _savefig(pdf, fig)


# ---------------------------------------------------------------------------
# Section 3: High-confidence busts
# ---------------------------------------------------------------------------

def plot_busts(df: pd.DataFrame, pdf: PdfPages, threshold: float = 0.15) -> None:
    _section_page(pdf, f"3.  High-confidence wrong predictions  (|coverprob - 0.5| > {threshold})")

    busts = df[(df["confidence"] > threshold) & (df["correct"] == 0)].copy()
    busts = busts.sort_values("confidence", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Distribution of bust residuals
    ax = axes[0]
    ax.hist(busts["residual"], bins=30, color="#DD4444", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("Residual on busts")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals on high-confidence wrong ({len(busts)} predictions)")

    # Bust rate by period
    ax = axes[1]
    period_df = df.groupby("period").agg(
        total=("correct", "size"),
        busts=("correct", lambda x: ((df.loc[x.index, "confidence"] > threshold) & (x == 0)).sum()),
    )
    period_df["bust_rate"] = period_df["busts"] / period_df["total"]
    ax.bar(period_df.index, period_df["bust_rate"], color="#DD4444", alpha=0.7)
    ax.set_xlabel("Period")
    ax.set_ylabel("Bust rate")
    ax.set_title("High-confidence bust rate by period")

    fig.tight_layout()
    _savefig(pdf, fig)

    # Top 30 busts table
    cols = ["team", "season", "period", "coverprob", "spread_diff", "predspread",
            "spreadscore", "residual", "covered"]
    cols = [c for c in cols if c in busts.columns]
    top = busts[cols].head(30)

    fig, ax = plt.subplots(figsize=(13, max(4, len(top) * 0.28 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=top.round(2).values,
        colLabels=top.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(range(len(top.columns)))
    ax.set_title(f"Top {len(top)} high-confidence wrong predictions (sorted by confidence)", pad=12)
    fig.tight_layout()
    _savefig(pdf, fig)


# ---------------------------------------------------------------------------
# Section 4: Team-level bias
# ---------------------------------------------------------------------------

def plot_team_bias(df: pd.DataFrame, pdf: PdfPages) -> None:
    _section_page(pdf, "4.  Team-level bias  (mean residual per team)")

    team_stats = (
        df.groupby("team")
        .agg(
            mean_res=("residual", "mean"),
            std_res=("residual", "std"),
            mae=("residual", lambda x: x.abs().mean()),
            acc=("correct", "mean"),
            n=("correct", "size"),
        )
        .sort_values("mean_res")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(team_stats) * 0.22 + 1)))

    # Mean residual bar (sorted)
    ax = axes[0]
    colors = ["#DD4444" if v < 0 else "#4C72B0" for v in team_stats["mean_res"]]
    ax.barh(team_stats["team"], team_stats["mean_res"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Mean residual  (positive = model over-predicted)")
    ax.set_title("Mean residual by team")
    ax.tick_params(axis="y", labelsize=8)

    # MAE + accuracy scatter
    ax = axes[1]
    sc = ax.scatter(team_stats["mae"], team_stats["acc"],
                    c=team_stats["mean_res"], cmap="RdBu_r",
                    s=60, vmin=-10, vmax=10, edgecolors="grey", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="mean residual")
    for _, row in team_stats.iterrows():
        ax.annotate(row["team"].split()[-1], (row["mae"], row["acc"]),
                    fontsize=6, ha="center", va="bottom")
    ax.axhline(df["correct"].mean(), color="grey", ls="--", lw=0.8, label="overall acc")
    ax.set_xlabel("MAE")
    ax.set_ylabel("Accuracy")
    ax.set_title("MAE vs accuracy by team")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _savefig(pdf, fig)


# ---------------------------------------------------------------------------
# Section 5: Period curve
# ---------------------------------------------------------------------------

def plot_period_curve(df: pd.DataFrame, pdf: PdfPages) -> None:
    _section_page(pdf, "5.  Season-progression curve  (accuracy and MAE by period)")

    period_df = df.groupby("period").agg(
        acc=("correct", "mean"),
        mae=("residual", lambda x: x.abs().mean()),
        n=("correct", "size"),
    ).reset_index()

    # Rolling 5-period smoothing
    period_df["acc_smooth"] = period_df["acc"].rolling(5, min_periods=1, center=True).mean()
    period_df["mae_smooth"] = period_df["mae"].rolling(5, min_periods=1, center=True).mean()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax = axes[0]
    ax.plot(period_df["period"], period_df["acc"], color="#cccccc", lw=1, label="raw")
    ax.plot(period_df["period"], period_df["acc_smooth"], color="#4C72B0", lw=2, label="5-period MA")
    ax.axhline(df["correct"].mean(), color="red", ls="--", lw=1, label=f"overall {df['correct'].mean():.3f}")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy over the season")
    ax.legend()
    ax.set_ylim(0.3, 0.8)

    ax = axes[1]
    ax.plot(period_df["period"], period_df["mae"], color="#cccccc", lw=1, label="raw")
    ax.plot(period_df["period"], period_df["mae_smooth"], color="#DD8844", lw=2, label="5-period MA")
    ax.set_ylabel("MAE")
    ax.set_xlabel("Period (game number)")
    ax.set_title("Regression MAE over the season")
    ax.legend()

    fig.tight_layout()
    _savefig(pdf, fig)


# ---------------------------------------------------------------------------
# Section 6: Spread segmentation (if spread column available)
# ---------------------------------------------------------------------------

def plot_spread_segments(df: pd.DataFrame, pdf: PdfPages) -> None:
    if "spread" not in df.columns or df["spread"].isna().mean() > 0.5:
        return

    _section_page(pdf, "6.  Spread size segmentation  (model accuracy by spread magnitude)")

    df = df.copy()
    df["spread_abs"] = df["spread"].abs()
    bins = [0, 2, 4, 6, 9, 100]
    labels = ["0-2", "2-4", "4-6", "6-9", "9+"]
    df["spread_bucket"] = pd.cut(df["spread_abs"], bins=bins, labels=labels, right=False)

    grp = df.groupby("spread_bucket", observed=True).agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        mae=("residual", lambda x: x.abs().mean()),
        cover_rate=("covered", "mean"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    x = range(len(grp))

    for ax, col, label, color in [
        (axes[0], "acc", "Accuracy", "#4C72B0"),
        (axes[1], "mae", "MAE", "#DD8844"),
        (axes[2], "cover_rate", "Actual cover rate", "#44AA66"),
    ]:
        ax.bar(x, grp[col], color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("|Spread|")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by spread size")
        for i, (v, n) in enumerate(zip(grp[col], grp["n"])):
            ax.text(i, v + 0.005, f"n={n}", ha="center", fontsize=8)

    fig.tight_layout()
    _savefig(pdf, fig)


# ---------------------------------------------------------------------------
# Section 7: Home / Away / B2B
# ---------------------------------------------------------------------------

def plot_context_splits(df: pd.DataFrame, pdf: PdfPages) -> None:
    ctx_cols = [c for c in ["home", "is_b2b"] if c in df.columns and df[c].notna().mean() > 0.3]
    if not ctx_cols:
        return

    _section_page(pdf, "7.  Game-context splits  (home/away, back-to-back)")

    fig, axes = plt.subplots(1, len(ctx_cols) * 2, figsize=(6 * len(ctx_cols) * 2 / 2 + 1, 5))
    if len(ctx_cols) * 2 == 1:
        axes = [axes]
    axes = list(axes)

    idx = 0
    for col in ctx_cols:
        sub = df[df[col].notna()].copy()
        sub[col] = sub[col].round().astype(int)
        grp = sub.groupby(col).agg(
            acc=("correct", "mean"),
            mae=("residual", lambda x: x.abs().mean()),
            n=("correct", "size"),
        )

        for metric, ylabel in [("acc", "Accuracy"), ("mae", "MAE")]:
            ax = axes[idx]
            labels = ["Away/No" if col == "home" else "No", "Home/Yes" if col == "home" else "Yes"]
            vals = grp[metric].values
            ax.bar(range(len(vals)), vals,
                   color=["#DD8844", "#4C72B0"][:len(vals)], alpha=0.85)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels[:len(vals)])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} by {col}")
            for i, (v, n) in enumerate(zip(vals, grp["n"].values)):
                ax.text(i, v * 1.01, f"n={n}", ha="center", fontsize=8)
            idx += 1

    fig.tight_layout()
    _savefig(pdf, fig)


# ---------------------------------------------------------------------------
# Section 8: Residual vs each feature (from features CSV)
# ---------------------------------------------------------------------------

def plot_feature_residuals(df: pd.DataFrame, feats: pd.DataFrame, pdf: PdfPages) -> None:
    if feats.empty or "team" not in feats.columns:
        return
    # Check if team column is string (joinable)
    if feats["team"].dtype != object:
        return

    _section_page(pdf, "8.  Residual vs each feature  (model already has these — checks for misfit)")

    merged = df.merge(feats, on=["team", "period"], how="inner", suffixes=("", "_feat"))
    if merged.empty:
        return

    skip = {"team", "period", "season", "covered", "spreadscore", "predspread",
            "spread_diff", "coverprob", "correct", "predicted_cover", "confidence",
            "residual", "lb", "elo_k", "sigma_diff", "decile", "conf_decile",
            "spread_bucket", "spread_abs"}
    feat_cols = [c for c in feats.columns if c not in skip
                 and pd.api.types.is_numeric_dtype(feats[c])
                 and merged[c].notna().sum() > 30]

    n_cols = 3
    n_rows = (len(feat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else list(axes)

    for i, col in enumerate(feat_cols):
        ax = axes[i]
        x = merged[col].values
        y = merged["residual"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        ax.scatter(x, y, alpha=0.15, s=8, color="#4C72B0")
        ax.axhline(0, color="black", lw=0.8, ls="--")

        # Binned mean residual
        try:
            bins = pd.qcut(x, q=8, duplicates="drop")
            bin_mean = pd.Series(y).groupby(bins).mean()
            bin_x = [interval.mid for interval in bin_mean.index]
            ax.plot(bin_x, bin_mean.values, "o-", color="red", lw=1.5, ms=4)
        except Exception:
            pass

        corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
        ax.set_title(f"{col}  (corr={corr:+.3f})", fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Residual", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Residual vs each feature (red = binned mean — slope suggests misfit)", fontsize=11)
    fig.tight_layout()
    _savefig(pdf, fig)


# ---------------------------------------------------------------------------
# Section 9: Sigma calibration
# ---------------------------------------------------------------------------

def plot_sigma_calibration(df: pd.DataFrame, pdf: PdfPages) -> None:
    if "sigma_diff" not in df.columns:
        return

    _section_page(pdf, "9.  Sigma calibration  (did larger sigma predict harder-to-call games?)")

    df = df.copy()
    df["sigma_decile"] = pd.qcut(df["sigma_diff"], q=8, duplicates="drop")
    grp = df.groupby("sigma_decile", observed=True).agg(
        acc=("correct", "mean"),
        mae=("residual", lambda x: x.abs().mean()),
        n=("correct", "size"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(grp))
    labels = [f"{i.right:.1f}" for i in grp.index]

    for ax, col, ylabel, color in [
        (axes[0], "acc", "Accuracy", "#4C72B0"),
        (axes[1], "mae", "MAE", "#DD8844"),
    ]:
        ax.bar(x, grp[col], color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel("sigma_diff decile (right edge)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by sigma_diff decile")
        for i, (v, n) in enumerate(zip(grp[col], grp["n"].values)):
            ax.text(i, v * 1.01, f"n={n}", ha="center", fontsize=7)

    fig.tight_layout()
    _savefig(pdf, fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(seasons: list[int], output_path: str = "error_analysis.pdf") -> None:
    all_preds = []
    all_feats = []
    for s in seasons:
        p, f = _load(s)
        all_preds.append(p)
        if not f.empty:
            all_feats.append(f)
        print(f"  Loaded season {s}: {len(p)} predictions, features joinable={not f.empty and f['team'].dtype == object}")

    df = pd.concat(all_preds, ignore_index=True)
    feats = pd.concat(all_feats, ignore_index=True) if all_feats else pd.DataFrame()

    season_str = "+".join(str(s) for s in seasons)
    print(f"\nGenerating error_analysis PDF -> {output_path}")

    with PdfPages(output_path) as pdf:
        _add_title_page(pdf, f"NBA Model Error Analysis — {season_str}")

        plot_residual_distribution(df, pdf)
        plot_residual_vs_confidence(df, pdf)
        plot_busts(df, pdf)
        plot_team_bias(df, pdf)
        plot_period_curve(df, pdf)
        plot_spread_segments(df, pdf)
        plot_context_splits(df, pdf)
        plot_feature_residuals(df, feats, pdf)
        plot_sigma_calibration(df, pdf)

    print(f"Done. {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest error analysis report.")
    parser.add_argument("--season", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out = args.output or f"error_analysis_nba_{'_'.join(str(s) for s in args.season)}.pdf"
    run_analysis(args.season, out)
