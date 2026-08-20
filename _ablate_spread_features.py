"""
Do the spread-derived features carry real signal for the MONEYLINE bet, or is
their contribution market information leaking back into a supposedly
market-blind model?

spreadscore = diff + spread, and the run line's sign is the book's
favourite/underdog call. So every spreadscore-derived feature smuggles in a
market signal that _MARKET_BLIND believes it removed.

Trains the win classifier three ways on 2022-2025, scores 2026:
  FULL      every feature as production has it
  NO-SPREAD the six spreadscore-derived features dropped
  NO-RATINGS also drops plain Elo, leaving only pitching, bullpen and
            context -- the floor with no rating features at all

Reports AUC, log loss, Brier and flat-stake backtest ROI at several EV cuts.
"""
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

import db, data_pipeline as dp, model as ml
from config import MLB

TEST_SEASON = 2026
SPREAD_FEATS = ["ats_elo_diff", "ats_opp_elo", "1_ago_ss", "ss_mean_5",
                "cover_streak", "fade_streak"]
THRESH = [0.0, 0.05, 0.10]

bundle = ml.load_model()
lookback, best_k = bundle["best_lookback"], bundle["best_k"]
params = dict(bundle["clf"].get_xgb_params())
params.update({"enable_categorical": True, "tree_method": "hist", "random_state": 42})

c = db.connect()
try: allg = db.fetch_games(c, "mlb")
finally: c.close()
seasons = sorted(allg["season"].unique())
allg = pd.concat([dp.filter_regular_season(allg[allg["season"]==s], MLB, s) for s in seasons],
                 ignore_index=True)

print("precomputing...")
cache = ml._precompute(allg, 200, TEST_SEASON, [best_k])
X_tr, X_te, y_tr, y_te, X_va, y_va, _, extra = ml.build_features(
    allg, 200, lookback, TEST_SEASON, MLB.eval_split_period, best_k, _cache=cache)
X_eval = pd.concat([X_te, X_va], ignore_index=True)
w_tr = extra["train"]
keys = list(extra["keys_test"]) + list(extra["keys_val"])

g = allg[allg["season"]==TEST_SEASON][["team","season","period","ml_odds","diff"]]
base = pd.DataFrame(keys, columns=["team","season","period"]).merge(
    g, on=["team","season","period"], how="left")
ok = base["diff"].notna() & (base["diff"]!=0) & base["ml_odds"].notna()
base = base[ok].reset_index(drop=True)
b_pay = np.where(base["ml_odds"]<0, 100/base["ml_odds"].abs(), base["ml_odds"]/100)
won = (base["diff"]>0).astype(int).values

m = w_tr.notna().values
present = [f for f in SPREAD_FEATS if f in X_tr.columns]
print(f"dropping: {present}\n")

VARIANTS = {
    "FULL":      [],
    "NO-SPREAD": present,
    "NO-RATINGS": present + [c for c in ["elo_diff","opponent_elo"] if c in X_tr.columns],
}

rows = []
for name, drop in VARIANTS.items():
    Xtr = X_tr.drop(columns=drop); Xev = X_eval.drop(columns=drop)
    clf = XGBClassifier(**params).fit(Xtr[m], w_tr[m])
    p_all = clf.predict_proba(Xev)[:, 1]
    p = p_all[ok.values]
    auc = roc_auc_score(won, p)
    ev = p*b_pay - (1-p)
    pnl = np.where(won==1, b_pay, -1.0)
    r = {"variant": name, "feats": Xtr.shape[1], "auc": auc,
         "logloss": log_loss(won, p), "brier": brier_score_loss(won, p)}
    for t in THRESH:
        s = ev > t
        r[f"roi_{t}"] = pnl[s].mean()*100 if s.sum() else np.nan
        r[f"n_{t}"] = int(s.sum())
    rows.append(r)

d = pd.DataFrame(rows)
print(f"{'variant':<11}{'feats':>6}{'AUC':>8}{'logloss':>9}{'Brier':>8}", end="")
for t in THRESH: print(f"{'ROI>'+str(t):>12}", end="")
print()
print("-"*79)
for _, r in d.iterrows():
    line = f"{r.variant:<11}{int(r.feats):>6}{r.auc:>8.4f}{r.logloss:>9.4f}{r.brier:>8.4f}"
    for t in THRESH:
        roi = r["roi_%s" % t]
        n = int(r["n_%s" % t])
        line += f"{'%+.1f%% (%d)' % (roi, n):>12}"
    print(line)
