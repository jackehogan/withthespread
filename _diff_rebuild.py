"""
Rebuild the six spreadscore-derived features on `diff` and test whether the
win model survives without the market signal embedded in the run line.

  1_ago_ss      -> 1_ago_diff       last game's run differential
  ss_mean_5     -> diff_mean_5      mean run differential, last 5
  cover_streak  -> win_streak       consecutive wins going in
  fade_streak   -> loss_streak      consecutive losses going in
  ats_elo_diff  -> long_elo_diff    Elo on diff at window 40 (ATS Elo's window)
  ats_opp_elo   -> long_opp_elo

All are strictly pre-game: shift(1) before any rolling window.

Variants: FULL (as production), NO-SPREAD (six dropped), DIFF-REBUILD
(six dropped, six diff-based added), PLUS-MARKET (rebuild plus an explicit
recent-favourite-rate feature, to price what market-blindness actually costs).
"""
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

import db, data_pipeline as dp, model as ml, elo as elo_mod
from config import MLB

TEST_SEASON = 2026
SPREAD_FEATS = ["ats_elo_diff","ats_opp_elo","1_ago_ss","ss_mean_5",
                "cover_streak","fade_streak"]
LONG_WINDOW = 40
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

# ---- diff-based replacements, keyed (team, season, period) ----
d = allg.copy()
for col in ("diff","spread"):
    d[col] = pd.to_numeric(d[col], errors="coerce")
d = d.sort_values(["team","season","period"])
grp = d.groupby(["team","season"])
d["1_ago_diff"]  = grp["diff"].shift(1)
d["diff_mean_5"] = grp["diff"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
d["fav_rate_5"]  = grp["spread"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())

def streaks(s):
    """Consecutive wins / losses strictly before each game."""
    w = (s > 0).astype(float).shift(1)
    out_w, out_l, cw, cl = [], [], 0, 0
    for v in w:
        out_w.append(cw); out_l.append(cl)
        if pd.isna(v): cw = cl = 0
        elif v == 1:   cw, cl = cw + 1, 0
        else:          cw, cl = 0, cl + 1
    return pd.Series(out_w, index=s.index), pd.Series(out_l, index=s.index)

ws, ls = [], []
for _, sub in d.groupby(["team","season"], sort=False):
    a, b = streaks(sub["diff"])
    ws.append(a); ls.append(b)
d["win_streak"], d["loss_streak"] = pd.concat(ws), pd.concat(ls)

long_elo = elo_mod.compute(d.dropna(subset=["diff"]), k=best_k, window=LONG_WINDOW,
                           value_col="diff")
d = d.set_index(["team","season","period"])
d["long_elo_diff"] = long_elo["elo_diff"].reindex(d.index)
d["long_opp_elo"]  = long_elo["opp_elo"].reindex(d.index)

NEW = ["1_ago_diff","diff_mean_5","win_streak","loss_streak",
       "long_elo_diff","long_opp_elo"]
extra_feats = d[NEW + ["fav_rate_5"]]

print("precomputing...")
cache = ml._precompute(allg, 200, TEST_SEASON, [best_k])
X_tr, X_te, y_tr, y_te, X_va, y_va, _, extra = ml.build_features(
    allg, 200, lookback, TEST_SEASON, MLB.eval_split_period, best_k, _cache=cache)
X_eval = pd.concat([X_te, X_va], ignore_index=True)
w_tr = extra["train"]
k_tr = pd.MultiIndex.from_tuples(extra["keys_train"])
k_ev = pd.MultiIndex.from_tuples(list(extra["keys_test"]) + list(extra["keys_val"]))

def attach(X, keys, cols):
    X = X.copy().reset_index(drop=True)
    add = extra_feats.reindex(keys)[cols].reset_index(drop=True)
    for c_ in cols: X[c_] = pd.to_numeric(add[c_], errors="coerce").values
    return X

g2 = allg[allg["season"]==TEST_SEASON][["team","season","period","ml_odds","diff"]]
base = pd.DataFrame(list(k_ev), columns=["team","season","period"]).merge(
    g2, on=["team","season","period"], how="left")
ok = (base["diff"].notna() & (base["diff"]!=0) & base["ml_odds"].notna()).values
pay = np.where(base["ml_odds"]<0, 100/base["ml_odds"].abs(), base["ml_odds"]/100)[ok]
won = (base["diff"]>0).astype(int).values[ok]
m = w_tr.notna().values

VARIANTS = {
    "FULL":         (SPREAD_FEATS and [], []),
    "NO-SPREAD":    (SPREAD_FEATS, []),
    "DIFF-REBUILD": (SPREAD_FEATS, NEW),
    "PLUS-MARKET":  (SPREAD_FEATS, NEW + ["fav_rate_5"]),
}

print(f"{'variant':<14}{'feats':>6}{'AUC':>8}{'logloss':>9}{'Brier':>8}"
      + "".join(f"{'ROI>'+str(t):>13}" for t in THRESH))
print("-" * 84)
for name, (drop, add) in VARIANTS.items():
    Xtr = X_tr.drop(columns=[c_ for c_ in drop if c_ in X_tr.columns])
    Xev = X_eval.drop(columns=[c_ for c_ in drop if c_ in X_eval.columns])
    if add:
        Xtr, Xev = attach(Xtr, k_tr, add), attach(Xev, k_ev, add)
    clf = XGBClassifier(**params).fit(Xtr[m], w_tr[m])
    p = clf.predict_proba(Xev)[:, 1][ok]
    ev = p*pay - (1-p); pnl = np.where(won==1, pay, -1.0)
    line = (f"{name:<14}{Xtr.shape[1]:>6}{roc_auc_score(won,p):>8.4f}"
            f"{log_loss(won,p):>9.4f}{brier_score_loss(won,p):>8.4f}")
    for t in THRESH:
        s = ev > t
        line += f"{'%+.1f%% (%d)' % (pnl[s].mean()*100, s.sum()):>13}" if s.sum() else f"{'—':>13}"
    print(line)
