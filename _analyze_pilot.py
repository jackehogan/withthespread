"""
Does the inverse-steam pattern survive a full season -- on the run line, and on
the moneyline the model actually bets?

The original pilot only ever measured the RUN LINE (label: spreadscore > 0).
The model bets the moneyline exclusively, so that arm is the one that decides
anything. The h2h prices ride along in the same odds-api request and used to be
discarded; both arms now cost the same credits.

Each arm compares the 5am price against a genuine late price from the SAME
source and the SAME book, so a 5am -> late difference isolates time rather than
venue. Prices are de-vigged within each game before comparison.

    python _check_pilot_usable.py && python _analyze_pilot.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import db


def raw(v):
    """American odds -> implied probability, vig included."""
    v = float(v)
    return abs(v) / (abs(v) + 100.0) if v < 0 else 100.0 / (v + 100.0)


def devig(frame, col, key="game_pk"):
    """Implied probability normalised so the two sides of a game sum to 1."""
    r = frame[col].apply(raw)
    return r / frame.groupby(key)[col].transform(lambda s: s.apply(raw).sum())


def arm(pairs, title, price_5am, price_late, outcome_col, extra=None):
    d = pairs.dropna(subset=[price_5am, price_late, outcome_col]).copy()
    d = d.groupby("game_pk").filter(lambda x: len(x) == 2)
    if len(d) < 100:
        print(f"\n=== {title} ===\n  only {len(d)} paired rows — skipping.")
        return
    y = (d[outcome_col] > 0).astype(int)
    p5 = devig(d, price_5am)
    pl = devig(d, price_late)

    print(f"\n=== {title} ===")
    print(f"  {len(d)} rows / {d['game_pk'].nunique()} games"
          f"   (~SE on AUC {1/np.sqrt(len(d)):.4f})")
    print(f"  {'5am price':<34}AUC {roc_auc_score(y, p5):.4f}")
    print(f"  {'late price (~pre-game)':<34}AUC {roc_auc_score(y, pl):.4f}")
    if extra:
        for lbl, col in extra:
            if col in d.columns and d[col].notna().all():
                print(f"  {lbl:<34}AUC {roc_auc_score(y, devig(d, col)):.4f}")

    mv = pl - p5
    print(f"  movement 5am -> late: mean |move| {mv.abs().mean()*100:.2f} pts, "
          f"p90 {mv.abs().quantile(.9)*100:.2f}")
    print(f"  {'AUC of the MOVE itself':<34}AUC {roc_auc_score(y, mv):.4f}"
          "   (>0.5 = follow steam, <0.5 = fade)")
    for thr in (0.02, 0.04):
        big = d[mv.abs() > thr]
        if len(big) < 60:
            continue
        bm = mv.loc[big.index]
        toward = y.loc[big.index][bm > 0].mean() * 100
        away = y.loc[big.index][bm < 0].mean() * 100
        n_t = int((bm > 0).sum()); n_a = int((bm < 0).sum())
        se = 1 / np.sqrt(min(n_t, n_a)) * 50
        print(f"    moves >{thr*100:.0f}pts (n={len(big)}): "
              f"hit when line moved TOWARD {toward:5.1f}% (n={n_t})   "
              f"AWAY {away:5.1f}% (n={n_a})   ~SE {se:.1f}")


def main():
    p = pd.read_csv("_pilot_closing_usable.csv")
    client = db.connect()
    try:
        games = db.fetch_games(client, "mlb", 2025)
    finally:
        client.close()
    games["game_pk"] = games["game_pk"].astype(str)
    j = p.merge(games[["team", "season", "period", "game_pk", "opponent"]],
                on=["team", "season", "period"], how="left").dropna(subset=["game_pk"])
    pairs = j.groupby("game_pk").filter(lambda d: len(d) == 2).copy()
    print(f"pilot rows {len(p)}  ->  both sides captured: {len(pairs)} "
          f"({pairs['game_pk'].nunique()} games)")
    if "book" in pairs.columns:
        print(f"books: {pairs['book'].value_counts().head(4).to_dict()}")
    print(f"median minutes before first pitch: "
          f"{pairs['minutes_before_start'].median():.0f}")

    # `price_sbr_close` is NOT an independent source: odds_source reads
    # odds-api-historical on every row, so it is the same feed, not SBR.
    arm(pairs, "RUN LINE  (label: covered)", "price_5am", "late_price",
        "spreadscore", extra=[("legacy `moneyline` column", "price_sbr_close")])
    arm(pairs, "MONEYLINE  (label: won) — the market the model bets",
        "price_5am_ml", "late_ml_price", "diff")

    print("\nOriginal 35-date pilot, run line only: TOWARD 40.8%  AWAY 59.2%")


if __name__ == "__main__":
    main()
