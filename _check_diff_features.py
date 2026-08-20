"""Validate the diff-based feature rebuild before retraining production."""
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import db, data_pipeline as dp, model as ml
from config import MLB

NEW = ["1_ago_diff","diff_mean_5","win_streak","loss_streak",
       "long_elo_diff","long_opp_elo"]
BANNED = ["diff","spreadscore","score","opp_score","won","covered"]

c = db.connect()
try: allg = db.fetch_games(c, "mlb")
finally: c.close()
seasons = sorted(allg["season"].unique())
allg = pd.concat([dp.filter_regular_season(allg[allg["season"]==s], MLB, s) for s in seasons],
                 ignore_index=True)
b = ml.load_model()
cache = ml._precompute(allg, 200, 2026, [b["best_k"]])
X_tr, X_te, y_tr, y_te, X_va, y_va, _, extra = ml.build_features(
    allg, 200, b["best_lookback"], 2026, MLB.eval_split_period, b["best_k"], _cache=cache)

print(f"\n[1] new features present   {X_tr.shape[1]} cols total")
for f in NEW:
    ok = f in X_tr.columns
    nn = X_tr[f].notna().mean()*100 if ok else 0
    print(f"    {f:<16}{'yes' if ok else 'MISSING':>8}   non-null {nn:>5.1f}%")
print("    RESULT:", "PASS" if all(f in X_tr.columns for f in NEW) else "FAIL")

print("\n[2] no outcome leaked into features")
bad = [c_ for c_ in X_tr.columns if c_ in BANNED]
print(f"    banned columns present: {bad or 'none'}")
print("    RESULT:", "FAIL" if bad else "PASS")

print("\n[3] streaks and lags are strictly pre-game")
ctx = ml._compute_context(allg)
j = allg.set_index(["team","season","period"])[["diff"]].join(
    ctx[["1_ago_diff","win_streak","loss_streak","diff_mean_5"]])
j = j.dropna(subset=["diff"])
# a win_streak > 0 must never coincide with the CURRENT game being a loss by construction
corr = j["win_streak"].corr((j["diff"]>0).astype(float))
print(f"    corr(win_streak, current game won) = {corr:+.4f}  (near 0 = no leak)")
srt = allg.sort_values(["team","season","period"])
lag = srt.groupby(["team","season"])["diff"].shift(1)
chk = pd.DataFrame({"expect": lag.values},
                   index=pd.MultiIndex.from_frame(srt[["team","season","period"]]))
got = ctx["1_ago_diff"].reindex(chk.index)
m = chk["expect"].notna() & got.notna()
mism = int((chk["expect"][m].values != got[m].values).sum())
print(f"    1_ago_diff matches previous game's diff: {int(m.sum())-mism}/{int(m.sum())}")
print("    RESULT:", "PASS" if mism == 0 and abs(corr) < 0.05 else "FAIL")

print("\n[4] win model would train on the narrower set")
drop = [c_ for c_ in ml._SPREAD_DERIVED_FEATURES if c_ in X_tr.columns]
print(f"    excluded: {drop}")
print(f"    cover model features: {X_tr.shape[1]}")
print(f"    win   model features: {X_tr.shape[1]-len(drop)}")
print("    RESULT:", "PASS" if len(drop) == 6 else "FAIL")
