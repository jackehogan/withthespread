"""
Is the SBR closing column trustworthy, and does inverse-steam survive?

Uses the 442 pilot rows where the odds-api snapshot was a genuine same-day
pre-game price (median 45 min before first pitch), from the SAME source as the
5am price -- so a 5am -> late comparison isolates time rather than book.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import db


def raw(v):
    v = float(v)
    return abs(v) / (abs(v) + 100.0) if v < 0 else 100.0 / (v + 100.0)


p = pd.read_csv("_pilot_closing_usable.csv")

client = db.connect()
try:
    games = db.fetch_games(client, "mlb", 2025)
finally:
    client.close()
games["game_pk"] = games["game_pk"].astype(str)
j = p.merge(games[["team", "season", "period", "game_pk", "opponent"]],
            on=["team", "season", "period"], how="left")
j = j.dropna(subset=["game_pk"])

# keep only games where BOTH sides were captured (needed to strip vig)
pairs = j.groupby("game_pk").filter(lambda d: len(d) == 2).copy()
print(f"pilot rows: {len(p)}   with both sides captured: {len(pairs)} "
      f"({pairs['game_pk'].nunique()} games)\n")
if len(pairs) < 100:
    print("too few complete games to analyse.")
    raise SystemExit

for col, name in [("price_5am", "p5am"), ("late_price", "plate"),
                  ("price_sbr_close", "psbr")]:
    r = pairs[col].apply(raw)
    pairs[name] = r / pairs.groupby("game_pk")[col].transform(lambda s: s.apply(raw).sum())

y = (pairs["spreadscore"] > 0).astype(int)
n = len(pairs)
se = 1 / np.sqrt(n)

print("1. Which price best predicts covering? (same games, same vig treatment)")
for name, label in [("p5am", "5am (odds-api)"),
                    ("plate", "late (odds-api, ~45min pre-game)"),
                    ("psbr", "SBR 'closing'")]:
    print(f"   {label:<36}AUC {roc_auc_score(y, pairs[name]):.4f}")

print(f"\n2. Do odds-api late and SBR 'closing' agree? (n={n}, ~SE {se:.4f})")
d = (pairs["plate"] - pairs["psbr"]).abs()
print(f"   correlation of implied probs : "
      f"{np.corrcoef(pairs['plate'], pairs['psbr'])[0,1]:.4f}")
print(f"   mean |difference|            : {d.mean()*100:.2f} prob points")
print(f"   median |difference|          : {d.median()*100:.2f}")
print(f"   share differing > 5 points   : {(d > 0.05).mean()*100:.1f}%")
d5 = (pairs["plate"] - pairs["p5am"]).abs()
print(f"   for reference, |late - 5am|  : mean {d5.mean()*100:.2f}, "
      f"median {d5.median()*100:.2f}")

print("\n3. Inverse-steam test, using odds-api late instead of SBR")
for mv_col, label in [("mv_api", "5am -> late  (odds-api, clean)"),
                      ("mv_sbr", "5am -> SBR close (original)")]:
    pairs["mv_api"] = pairs["plate"] - pairs["p5am"]
    pairs["mv_sbr"] = pairs["psbr"] - pairs["p5am"]
    mv = pairs[mv_col]
    auc = roc_auc_score(y, mv)
    big = pairs[mv.abs() > 0.02]
    if len(big) < 60:
        print(f"   {label:<34}AUC {auc:.4f}   (only {len(big)} big moves)")
        continue
    toward = y[big[big[mv_col] > 0].index].mean() * 100
    away = y[big[big[mv_col] < 0].index].mean() * 100
    print(f"   {label:<34}AUC {auc:.4f}")
    print(f"      moves >2pts: n={len(big):>4}   "
          f"cover when line moved TOWARD {toward:5.1f}%   AWAY {away:5.1f}%")

print(f"\n   Original 2025 finding (SBR, n=1900): TOWARD 42.9%  AWAY 57.2%")
print(f"   ~SE on a cover rate at n~200 is about {1/np.sqrt(200)*50:.1f} points.")
