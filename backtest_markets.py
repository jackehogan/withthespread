"""
Backtest three betting strategies on out-of-sample data.

    A. run line only   -- bet when EV(coverprob, spread_juice) clears a threshold
    B. moneyline only  -- bet when EV(win_prob,  ml_odds)      clears it
    C. max of both     -- bet whichever EV is higher (what predict_mlb does now)

If C beats A and B, market selection carries real signal. If C lands between
them or below, the selection is picking noise -- the winner's-curse case.

Run on 2026, which the model has never seen: training stops at 2025.
Bets are priced against the ACTUAL stored odds and graded on the outcome that
market settles (cover for the run line, win for the moneyline).
"""
import numpy as np
import pandas as pd

import db
import data_pipeline as dp
import model as ml
from config import MLB

TEST_SEASON = 2026
THRESHOLDS = [0.0, 0.02, 0.05, 0.10]


def payout(odds):
    """Profit per 1 unit staked on a winning bet at American odds."""
    o = float(odds)
    return 100.0 / abs(o) if o < 0 else o / 100.0


def ev(prob, odds):
    if np.isnan(prob) or np.isnan(odds):
        return np.nan
    return prob * payout(odds) - (1.0 - prob)


bundle = ml.load_model()
clf, win_clf = bundle["clf"], bundle.get("win_clf")
lookback, best_k = bundle["best_lookback"], bundle["best_k"]
if win_clf is None:
    raise SystemExit("bundle has no win model — retrain first")

client = db.connect()
try:
    allg = db.fetch_games(client, "mlb")
finally:
    client.close()
seasons = sorted(allg["season"].unique())
allg = pd.concat(
    [dp.filter_regular_season(allg[allg["season"] == s], MLB, s) for s in seasons],
    ignore_index=True,
)

# Put 2026 in the eval split so it is scored, never trained on.
# _precompute directly gives the cached fast path (which carries row keys)
# without paying for the lookback/K search we do not need here.
print("precomputing features (this takes a minute)...")
cache = ml._precompute(allg, 200, TEST_SEASON, [best_k])
out = ml.build_features(allg, 200, lookback, TEST_SEASON,
                        MLB.eval_split_period, best_k, _cache=cache)
X_test, X_val, extra = out[1], out[4], out[7]
X = pd.concat([X_test, X_val], ignore_index=True)
keys = list(extra["keys_test"]) + list(extra["keys_val"])
print(f"scored rows: {len(X)}  keys: {len(keys)}")
if len(X) != len(keys):
    raise SystemExit("key/row misalignment")

feats = list(clf.get_booster().feature_names)
pred = pd.DataFrame(keys, columns=["team", "season", "period"])
pred["coverprob"] = clf.predict_proba(X[feats])[:, 1]
pred["win_prob"] = win_clf.predict_proba(X[list(win_clf.get_booster().feature_names)])[:, 1]

# Join actual prices and outcomes
g = allg[allg["season"] == TEST_SEASON][
    ["team", "season", "period", "spread_juice", "ml_odds", "spreadscore", "diff"]]
d = pred.merge(g, on=["team", "season", "period"], how="inner")
d = d[(d["spreadscore"].notna()) & (d["spreadscore"] != 0) & (d["diff"] != 0)]
print(f"rows with prices and outcomes: {len(d)}\n")

d["ev_spread"] = [ev(p, o) for p, o in zip(d["coverprob"], d["spread_juice"])]
d["ev_ml"] = [ev(p, o) for p, o in zip(d["win_prob"], d["ml_odds"])]
d["covered"] = (d["spreadscore"] > 0).astype(int)
d["won"] = (d["diff"] > 0).astype(int)
d["pnl_spread"] = np.where(d["covered"] == 1,
                           [payout(o) for o in d["spread_juice"]], -1.0)
d["pnl_ml"] = np.where(d["won"] == 1,
                       [payout(o) for o in d["ml_odds"]], -1.0)

print(f"{'strategy':<22}{'thresh':>8}{'bets':>7}{'W-L':>12}{'P&L':>10}{'ROI':>9}")
print("-" * 70)
summary = []
for thr in THRESHOLDS:
    a = d[d["ev_spread"] > thr]
    b = d[d["ev_ml"] > thr]
    take_ml = d["ev_ml"] > d["ev_spread"]
    c_ev = np.where(take_ml, d["ev_ml"], d["ev_spread"])
    c = d[c_ev > thr].copy()
    c["pnl"] = np.where(take_ml[c_ev > thr], c["pnl_ml"], c["pnl_spread"])
    c["won_bet"] = np.where(take_ml[c_ev > thr], c["won"], c["covered"])

    for name, sub, pnl_col, win_col in (
        ("A run line only", a, "pnl_spread", "covered"),
        ("B moneyline only", b, "pnl_ml", "won"),
        ("C max of both", c, "pnl", "won_bet"),
    ):
        if len(sub) == 0:
            print(f"{name:<22}{thr:>8.2f}{0:>7}")
            continue
        pnl = sub[pnl_col].sum()
        roi = pnl / len(sub) * 100
        w = int(sub[win_col].sum())
        rec = f"{w}-{len(sub) - w}"
        print(f"{name:<22}{thr:>8.2f}{len(sub):>7}{rec:>12}{pnl:>10.1f}{roi:>8.1f}%")
        summary.append((thr, name, len(sub), roi))
    print()

print("Standard error on ROI is roughly 110% / sqrt(bets), so at 500 bets a")
print("difference under ~5 points is noise. If C does not clearly beat both")
print("A and B, selecting on EV is not adding information.")
