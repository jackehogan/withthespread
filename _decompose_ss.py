"""
spreadscore = diff + spread, and spread is +/-1.5 by favourite status.
So every rolling spreadscore feature is a PERFORMANCE term plus a MARKET term:

    ss_mean_5 = diff_mean_5  +  1.5 * (2 * dog_rate_5 - 1)

The second term is nothing but "how often has the book made this team an
underdog in its last 5 games". Quantify how much of the feature that is.
"""
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import db, data_pipeline as dp
from config import MLB

c = db.connect()
try: g = db.fetch_games(c, "mlb")
finally: c.close()
seasons = sorted(g["season"].unique())
g = pd.concat([dp.filter_regular_season(g[g["season"]==s], MLB, s) for s in seasons],
              ignore_index=True)
for col in ("diff","spread","spreadscore"):
    g[col] = pd.to_numeric(g[col], errors="coerce")
g = g.dropna(subset=["diff","spread","spreadscore"]).sort_values(["team","season","period"])

grp = g.groupby(["team","season"])
g["diff_m5"]   = grp["diff"].transform(lambda s: s.shift(1).rolling(5).mean())
g["spread_m5"] = grp["spread"].transform(lambda s: s.shift(1).rolling(5).mean())
g["ss_m5"]     = grp["spreadscore"].transform(lambda s: s.shift(1).rolling(5).mean())
d = g.dropna(subset=["diff_m5","spread_m5","ss_m5"])

print(f"team-games with a full 5-game history: {len(d)}\n")
print("variance decomposition of ss_mean_5 = diff_mean_5 + spread_mean_5")
vd, vs, vt = d["diff_m5"].var(), d["spread_m5"].var(), d["ss_m5"].var()
cov = d["diff_m5"].cov(d["spread_m5"])
print(f"  var(performance term)  {vd:.4f}   {vd/vt*100:>5.1f}% of total")
print(f"  var(market term)       {vs:.4f}   {vs/vt*100:>5.1f}%")
print(f"  2*cov                  {2*cov:.4f}   {2*cov/vt*100:>5.1f}%")
print(f"  var(ss_mean_5)         {vt:.4f}")
print()
print(f"corr(ss_mean_5, performance only) : {d['ss_m5'].corr(d['diff_m5']):.3f}")
print(f"corr(ss_mean_5, market only)      : {d['ss_m5'].corr(d['spread_m5']):.3f}")
print(f"corr(performance, market)         : {d['diff_m5'].corr(d['spread_m5']):.3f}")
print()
print("spread_mean_5 is just recent underdog rate:")
d = d.copy(); d["dog_rate_5"] = (d["spread_m5"] + 1.5) / 3.0
print(f"  corr(spread_mean_5, dog_rate_5)  : {d['spread_m5'].corr(d['dog_rate_5']):.3f} (identity)")
print(f"  dog_rate_5 range                 : {d['dog_rate_5'].min():.2f} .. {d['dog_rate_5'].max():.2f}")
print()
print("Does the market term predict the NEXT game's result on its own?")
d["won"] = (d["diff"] > 0).astype(int)
from sklearn.metrics import roc_auc_score
print(f"  AUC of -spread_mean_5 (fewer dog games = better team): "
      f"{roc_auc_score(d['won'], -d['spread_m5']):.4f}")
print(f"  AUC of  diff_mean_5                                 : "
      f"{roc_auc_score(d['won'],  d['diff_m5']):.4f}")
print(f"  AUC of  ss_mean_5                                   : "
      f"{roc_auc_score(d['won'],  d['ss_m5']):.4f}")
