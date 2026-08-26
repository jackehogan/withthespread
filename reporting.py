"""
Email report building blocks: stake sizing, yesterday's graded results, and the
nightly ROI chart.

Stake sizing lives here rather than inside the email builder because two places
need the identical rule -- today's recommendation and the reconstruction of what
yesterday's recommendation would have been. If they drift, the reported dollar
results stop describing the bets that were actually advised.
"""
from __future__ import annotations

import datetime
import io

import matplotlib
matplotlib.use("Agg")           # no display in the GitHub Actions runner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The moneyline-only, market-blind model running under GitHub Actions. Earlier
# predictions came from a different model and are not comparable.
PIPELINE_START = "2026-08-05"

BANKROLL = 100.0
KELLY_MULT = 0.10
THRESHOLDS = [0.0, 0.05, 0.10]
WINDOW = 7

PANELS = [
    ("0.00", "all bets", "#2a78d6"),
    ("0.05", "EV > 0.05", "#eb6834"),
    ("0.10", "EV > 0.10", "#1baf7a"),
]


def payout(odds: float) -> float:
    """Decimal profit per 1 unit staked at an American price."""
    odds = float(odds)
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0


def kelly_bet(ev, odds) -> float | None:
    """Recommended stake on a $100 bankroll at 0.10x fractional Kelly."""
    try:
        ev, odds = float(ev), float(odds)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ev) or not np.isfinite(odds) or ev <= 0:
        return None
    b = payout(odds)
    if b <= 0:
        return None
    return round(BANKROLL * KELLY_MULT * ev / b, 2)


# ---------------------------------------------------------------------------
# Yesterday's results
# ---------------------------------------------------------------------------

def yesterday_results(preds: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    The bets advised for the day before `target_date`, with their outcome.

    Returns one row per advised bet (ev > 0), including any still ungraded, so a
    postponed or late-finishing game shows as pending rather than vanishing.
    Columns: team, opponent, ml_odds, ev, stake, result, dollars.
    """
    if preds.empty or "prediction_date" not in preds.columns:
        return pd.DataFrame()

    prev = (pd.to_datetime(target_date).date() - datetime.timedelta(days=1)).isoformat()
    d = preds.copy()
    d["prediction_date"] = d["prediction_date"].astype(str)
    d = d[d["prediction_date"] == prev]
    if d.empty:
        return pd.DataFrame()

    for col in ("ev", "ml_odds", "pnl"):
        d[col] = pd.to_numeric(d.get(col), errors="coerce")
    d = d[d["ev"] > 0].copy()
    if d.empty:
        return pd.DataFrame()

    d["stake"] = [kelly_bet(e, o) for e, o in zip(d["ev"], d["ml_odds"])]
    d = d[d["stake"].notna()]

    def _result(r):
        if pd.isna(r["pnl"]):
            return "PENDING"
        if r["pnl"] == 0:
            return "PUSH"
        return "WIN" if r["pnl"] > 0 else "LOSS"

    d["result"] = d.apply(_result, axis=1)
    # pnl is profit per unit staked, so dollars follows from the stake directly.
    d["dollars"] = np.where(d["pnl"].notna(), d["stake"] * d["pnl"], np.nan)

    cols = ["team", "opponent", "ml_odds", "ev", "stake", "result", "dollars"]
    for c in cols:
        if c not in d.columns:
            d[c] = None
    return d[cols].sort_values("stake", ascending=False).reset_index(drop=True)


def yesterday_html(res: pd.DataFrame, target_date: str) -> str:
    """Render yesterday's advised bets as an HTML table."""
    prev = (pd.to_datetime(target_date).date() - datetime.timedelta(days=1)).isoformat()
    if res.empty:
        return (f"<h3>Yesterday &mdash; {prev}</h3>"
                f"<p style='color:#777'>No advised bets on {prev}.</p>")

    tint = {"WIN": "#e8f5e9", "LOSS": "#fdecea", "PUSH": "#f4f4f4", "PENDING": "#fff8e1"}
    rows = []
    for _, r in res.iterrows():
        amt = ("&#8212;" if pd.isna(r["dollars"])
               else f"<b>{'+' if r['dollars'] >= 0 else '&minus;'}"
                    f"${abs(r['dollars']):.2f}</b>")
        colour = ("#1b7f3b" if r["result"] == "WIN"
                  else "#b3261e" if r["result"] == "LOSS" else "#555")
        rows.append(
            f"<tr style='background:{tint.get(r['result'], '')}'>"
            f"<td><b>{str(r['team']).split()[-1]} ML</b></td>"
            f"<td>vs {r['opponent']}</td>"
            f"<td>{int(r['ml_odds']):+d}</td>"
            f"<td>{r['ev']:+.3f}</td>"
            f"<td>${r['stake']:.2f}</td>"
            f"<td style='color:{colour};font-weight:bold'>{r['result']}</td>"
            f"<td style='color:{colour}'>{amt}</td></tr>"
        )

    graded = res[res["dollars"].notna()]
    n_w = int((graded["result"] == "WIN").sum())
    n_l = int((graded["result"] == "LOSS").sum())
    staked = graded["stake"].sum()
    net = graded["dollars"].sum()
    if len(graded):
        foot = (f"<tr><td colspan='4' style='text-align:right'><b>{n_w}-{n_l}</b>"
                f" on {len(graded)} graded</td>"
                f"<td><b>${staked:.2f}</b></td><td>staked</td>"
                f"<td style='color:{'#1b7f3b' if net >= 0 else '#b3261e'};"
                f"font-weight:bold'>{'+' if net >= 0 else '&minus;'}${abs(net):.2f}"
                f" ({net / staked * 100:+.1f}%)</td></tr>")
    else:
        foot = ("<tr><td colspan='7' style='color:#777'>None graded yet "
                "&mdash; results pending.</td></tr>")

    return (
        f"<h3>Yesterday &mdash; {prev}</h3>"
        "<table class='picks'>"
        "<tr><th>Bet</th><th>Matchup</th><th>Line</th><th>EV</th>"
        "<th>Bet ($100 BR)</th><th>Result</th><th>Amount</th></tr>"
        + "".join(rows) + foot + "</table>"
    )


# ---------------------------------------------------------------------------
# ROI chart
# ---------------------------------------------------------------------------

def daily_roi_series(preds: pd.DataFrame, start: str = PIPELINE_START) -> dict | None:
    """
    Daily ROI at each EV threshold, with a rolling mean and +/-1 SE band.

    The band is the SE of a mean over the bets inside each window, so it narrows
    as volume accumulates rather than being a fixed width.
    """
    if preds.empty or "pnl" not in preds.columns:
        return None
    g = preds.copy()
    g["date"] = g["prediction_date"].astype(str)
    for col in ("pnl", "ev"):
        g[col] = pd.to_numeric(g.get(col), errors="coerce")
    g = g.dropna(subset=["pnl", "ev"])
    g = g[g["date"] >= start]
    if g.empty:
        return None

    per_bet_sd = float(g["pnl"].std())
    dates = sorted(g["date"].unique())
    series = {}
    for thr in THRESHOLDS:
        sub = g[g["ev"] > thr]
        daily = (sub.groupby("date")["pnl"]
                    .agg(bets="size", pnl="sum")
                    .reindex(dates, fill_value=0))
        roi = np.where(daily["bets"] > 0,
                       daily["pnl"] / daily["bets"].replace(0, np.nan) * 100, np.nan)
        roll_pnl = daily["pnl"].rolling(WINDOW, min_periods=1).sum()
        roll_n = daily["bets"].rolling(WINDOW, min_periods=1).sum()
        series[f"{thr:.2f}"] = {
            "daily_roi": roi,
            "daily_bets": daily["bets"].to_numpy(),
            "roll_roi": np.where(roll_n > 0, roll_pnl / roll_n * 100, np.nan),
            "roll_se": np.where(roll_n > 0, per_bet_sd / np.sqrt(roll_n) * 100, np.nan),
            "total_bets": int(len(sub)),
            "total_roi": float(sub["pnl"].sum() / max(len(sub), 1) * 100),
        }
    return {"dates": dates, "series": series}


def roi_chart_png(preds: pd.DataFrame, start: str = PIPELINE_START) -> bytes | None:
    """Render the nightly ROI tracker as a PNG. Returns None if there is nothing to plot."""
    d = daily_roi_series(preds, start)
    if not d:
        return None
    dates, series = d["dates"], d["series"]
    n = len(dates)
    xs = np.arange(n)

    fig, axes = plt.subplots(len(PANELS), 1, sharex=True, figsize=(7.0, 5.4))
    for ax, (key, label, col) in zip(axes, PANELS):
        s = series[key]
        roll, se = s["roll_roi"], s["roll_se"]
        ok = ~np.isnan(roll)
        if ok.any():
            ax.fill_between(xs[ok], (roll - se)[ok], (roll + se)[ok],
                            color=col, alpha=0.14, linewidth=0)
            ax.plot(xs[ok], roll[ok], color=col, linewidth=2, zorder=3)

        ax.axhline(0, color="#b5b3a8", linewidth=1, zorder=1)
        ax.axhline(s["total_roi"], color=col, linewidth=1.4, linestyle=(0, (6, 4)),
                   alpha=0.85, zorder=2)
        ax.text(0.2, s["total_roi"] + (3 if s["total_roi"] >= 0 else -9),
                f"avg {s['total_roi']:+.1f}%", color=col, fontsize=8)

        pts = ~np.isnan(s["daily_roi"])
        ax.scatter(xs[pts], s["daily_roi"][pts],
                   s=14 + np.minimum(s["daily_bets"][pts], 14) * 4,
                   color=col, alpha=0.42, edgecolors="white", linewidths=0.9, zorder=4)

        ax.set_ylim(-105, 45)
        ax.set_yticks([-100, -75, -50, -25, 0, 25])
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="#e3e2db", linestyle=":", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{label}  —  {s['total_bets']} bets, {s['total_roi']:+.1f}% to date",
                     fontsize=9.5, fontweight="bold", loc="left", color="#1a1a2e")

    step = max(1, n // 8)
    axes[-1].set_xticks(xs[::step])
    axes[-1].set_xticklabels([d[5:] for d in dates[::step]], fontsize=8)
    axes[-1].set_xlim(-0.6, n - 0.4)

    fig.tight_layout(h_pad=1.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return buf.getvalue()
