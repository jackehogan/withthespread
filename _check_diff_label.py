"""
Verify the win label now reads `diff` directly, and that carrying `diff`
through the context frame did not leak it into the feature matrix.

Three checks:
  1. `diff` (and any other outcome column) is absent from every feature matrix.
  2. The new label agrees with the legacy `spreadscore - spread` route on every
     row where the legacy route could produce an answer at all.
  3. The win label matches the raw game result from the DB, row for row.
"""
import numpy as np
import pandas as pd

import db
import data_pipeline as dp
import model as ml
from config import MLB

TEST_SEASON = 2026
BANNED = ["diff", "spreadscore", "score", "opp_score", "won", "covered"]

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

bundle = ml.load_model()
lookback, best_k = bundle["best_lookback"], bundle["best_k"]

print("precomputing...")
cache = ml._precompute(allg, 200, TEST_SEASON, [best_k])
X_tr, X_te, y_tr, y_te, X_va, y_va, _, extra = ml.build_features(
    allg, 200, lookback, TEST_SEASON, MLB.eval_split_period, best_k, _cache=cache)

# --- 1. no outcome column reached the features ---
print("\n[1] leakage check")
bad = []
for name, X in (("train", X_tr), ("test", X_te), ("val", X_va)):
    hits = [c for c in X.columns if c in BANNED]
    print(f"  {name:<6} {X.shape[1]:>3} cols   banned present: {hits or 'none'}")
    bad += hits
print("  RESULT:", "FAIL" if bad else "PASS")

# --- 2. new label vs the legacy spreadscore - spread route ---
print("\n[2] new label vs legacy reconstruction")
ctx = ml._compute_context(allg)
print(f"  context carries 'diff': {'diff' in ctx.columns}")

ss = pd.to_numeric(allg["spreadscore"], errors="coerce").values
sp = pd.to_numeric(allg["spread"], errors="coerce").values
dv = pd.to_numeric(allg["diff"], errors="coerce").values

legacy = np.where(np.isnan(ss - sp), np.nan, ((ss - sp) > 0).astype(float))
new    = np.where(np.isnan(dv), np.nan, (dv > 0).astype(float))

both = ~np.isnan(legacy) & ~np.isnan(new)
disagree = int((legacy[both] != new[both]).sum())
print(f"  rows both routes label : {int(both.sum())}")
print(f"  disagreements          : {disagree}")
print(f"  legacy NaN, new usable : {int((np.isnan(legacy) & ~np.isnan(new)).sum())}")
print(f"  new NaN, legacy usable : {int((~np.isnan(legacy) & np.isnan(new)).sum())}")
print("  RESULT:", "FAIL" if disagree else "PASS")

# --- 3. label matches the actual game result ---
print("\n[3] win label vs raw score")
sc = pd.to_numeric(allg["score"], errors="coerce").values
op = pd.to_numeric(allg["opp_score"], errors="coerce").values
m = ~np.isnan(sc) & ~np.isnan(op) & ~np.isnan(new)
truth = (sc[m] - op[m] > 0).astype(float)
mism = int((truth != new[m]).sum())
print(f"  rows compared : {int(m.sum())}")
print(f"  mismatches    : {mism}")
print("  RESULT:", "FAIL" if mism else "PASS")

# --- how many rows the win model actually trains on now ---
w_tr = extra["train"]
print(f"\nwin-label coverage in train split: "
      f"{int(w_tr.notna().sum())}/{len(w_tr)} "
      f"({w_tr.notna().mean()*100:.2f}%)")
