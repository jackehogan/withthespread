"""
Same three-strategy backtest, but with a MARKET-BLIND model.

The production model takes ml_implied_prob, derived from the run-line price,
so its apparent edge could simply be market information fed back at itself.
This retrains cover and win classifiers with every market-derived feature
removed -- only Elo, rolling SpreadScore, pitching, bullpen and game context
remain -- then runs the identical strategies.

If the moneyline edge survives, the model has genuinely independent signal.
If it vanishes, the edge was the market's, not the model's.

Trains on 2022-2025, scores 2026, which is never trained on.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import db
import data_pipeline as dp
import model as ml
from config import MLB

TEST_SEASON = 2026
THRESHOLDS = [0.0, 0.02, 0.05, 0.10]
# Anything carrying a price. `spread` is the run line itself; the ablation
# showed it contributes 0.0000 anyway, and keeping it would leak favourite
# status back in.
MARKET_COLS = ["ml_implied_prob", "spread"]


def payout(o):
    o = float(o)
    return 100.0 / abs(o) if o < 0 else o / 100.0


def ev(p, o):
    if np.isnan(p) or np.isnan(o):
        return np.nan
    return p * payout(o) - (1.0 - p)


bundle = ml.load_model()
lookback, best_k = bundle["best_lookback"], bundle["best_k"]
params = dict(bundle["clf"].get_xgb_params())
params.update({"enable_categorical": True, "tree_method": "hist",
               "random_state": 42})

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

print("precomputing...")
cache = ml._precompute(allg, 200, TEST_SEASON, [best_k])
out = ml.build_features(allg, 200, lookback, TEST_SEASON,
                        MLB.eval_split_period, best_k, _cache=cache)
X_tr, X_te, y_tr, y_te, X_va, y_va, _, extra = out

X_eval = pd.concat([X_te, X_va], ignore_index=True)
y_eval = pd.concat([y_te, y_va], ignore_index=True)
w_tr = extra["train"]
w_eval = pd.concat([extra["test"], extra["val"]], ignore_index=True)
keys = list(extra["keys_test"]) + list(extra["keys_val"])

drop = [c for c in MARKET_COLS if c in X_tr.columns]
Xtr_b = X_tr.drop(columns=drop)
Xev_b = X_eval.drop(columns=drop)
print(f"train {Xtr_b.shape}   eval {Xev_b.shape}")
print(f"removed: {drop}")
print(f"remaining features: {list(Xev_b.columns)}\n")

cover = XGBClassifier(**params).fit(Xtr_b, y_tr)
m = w_tr.notna().values
win = XGBClassifier(**params).fit(Xtr_b[m], w_tr[m])

d = pd.DataFrame(keys, columns=["team", "season", "period"])
d["coverprob"] = cover.predict_proba(Xev_b)[:, 1]
d["win_prob"] = win.predict_proba(Xev_b)[:, 1]

g = allg[allg["season"] == TEST_SEASON][
    ["team", "season", "period", "game_pk", "spread_juice", "ml_odds",
     "spreadscore", "diff"]]
d = d.merge(g, on=["team", "season", "period"], how="inner")
d = d[(d["spreadscore"].notna()) & (d["spreadscore"] != 0) & (d["diff"] != 0)]
d = d.dropna(subset=["spread_juice", "ml_odds"])
d["covered"] = (d["spreadscore"] > 0).astype(int)
d["won"] = (d["diff"] > 0).astype(int)

print("Market-blind model quality on 2026:")
print(f"  coverprob -> covering  AUC {roc_auc_score(d['covered'], d['coverprob']):.4f}"
      f"   (with market: 0.6168)")
print(f"  win_prob  -> winning   AUC {roc_auc_score(d['won'], d['win_prob']):.4f}"
      f"   (with market: 0.5808)")

def raw(v):
    v = float(v)
    return abs(v) / (abs(v) + 100.0) if v < 0 else 100.0 / (v + 100.0)

d["game_pk"] = d["game_pk"].astype(str)
pp = d.groupby("game_pk").filter(lambda x: len(x) == 2)
if len(pp):
    r = pp["ml_odds"].apply(raw)
    mk = r / pp.groupby("game_pk")["ml_odds"].transform(
        lambda s: s.apply(raw).sum())
    print(f"  market h2h price       AUC {roc_auc_score(pp['won'], mk):.4f}")

d["ev_spread"] = [ev(p, o) for p, o in zip(d["coverprob"], d["spread_juice"])]
d["ev_ml"] = [ev(p, o) for p, o in zip(d["win_prob"], d["ml_odds"])]
d["pnl_spread"] = np.where(d["covered"] == 1,
                           [payout(o) for o in d["spread_juice"]], -1.0)
d["pnl_ml"] = np.where(d["won"] == 1, [payout(o) for o in d["ml_odds"]], -1.0)

print(f"\n{'strategy':<22}{'thresh':>8}{'bets':>7}{'P&L':>10}{'ROI':>9}"
      f"{'  (with market)':>16}")
print("-" * 76)
WITH_MARKET = {  # from backtest_markets.py, for side-by-side
    (0.0, "A"): 4.3, (0.0, "B"): 7.6, (0.0, "C"): 3.9,
    (0.02, "A"): 5.3, (0.02, "B"): 9.9, (0.02, "C"): 5.0,
    (0.05, "A"): 5.9, (0.05, "B"): 11.7, (0.05, "C"): 6.3,
    (0.10, "A"): 10.4, (0.10, "B"): 16.7, (0.10, "C"): 10.7,
}
for thr in THRESHOLDS:
    a = d[d["ev_spread"] > thr]
    b = d[d["ev_ml"] > thr]
    take_ml = d["ev_ml"] > d["ev_spread"]
    c_ev = np.where(take_ml, d["ev_ml"], d["ev_spread"])
    sel = c_ev > thr
    c = d[sel].copy()
    c["pnl"] = np.where(take_ml[sel], c["pnl_ml"], c["pnl_spread"])

    for tag, name, sub, col in (("A", "A run line only", a, "pnl_spread"),
                                ("B", "B moneyline only", b, "pnl_ml"),
                                ("C", "C max of both", c, "pnl")):
        if len(sub) == 0:
            continue
        roi = sub[col].sum() / len(sub) * 100
        prev = WITH_MARKET.get((thr, tag))
        print(f"{name:<22}{thr:>8.2f}{len(sub):>7}{sub[col].sum():>10.1f}"
              f"{roi:>8.1f}%{prev:>15.1f}%")
    print()

print("If the market-blind ROIs collapse toward -4% (the vig), the earlier")
print("edge came from the market feature rather than from the model.")
