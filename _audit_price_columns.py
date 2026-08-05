"""
Classify what the `moneyline` column prices, per season, from the data itself.

A price implies a probability. If that probability predicts WINNING better than
COVERING, the column holds the h2h moneyline; if it predicts covering better,
it holds the run-line juice. Sign heuristics are ambiguous for heavy
favourites, so this uses AUC against both outcomes.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import db
import data_pipeline as dp
from config import MLB


def raw_prob(v):
    v = float(v)
    return abs(v) / (abs(v) + 100.0) if v < 0 else 100.0 / (v + 100.0)


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

g = allg[allg["game_pk"].notna() & (allg["game_pk"].astype(str) != "")].copy()
g["game_pk"] = g["game_pk"].astype(str)


def novig(frame, col):
    f = frame.dropna(subset=[col]).copy()
    f = f.groupby("game_pk").filter(lambda d: len(d) == 2)
    if f.empty:
        return f
    f["_raw"] = f[col].apply(raw_prob)
    tot = f.groupby("game_pk")["_raw"].transform("sum")
    f["_p"] = f["_raw"] / tot
    return f


print("What does each season's `moneyline` column price?\n")
print(f"{'season':<8}{'n':>7}{'AUC vs WIN':>13}{'AUC vs COVER':>15}   verdict")
print("-" * 68)

verdicts = {}
for season in seasons:
    f = novig(g[g["season"] == season], "moneyline")
    f = f.dropna(subset=["diff", "spreadscore"])
    f = f[f["spreadscore"] != 0]
    if len(f) < 200:
        print(f"{season:<8}{len(f):>7}   insufficient")
        continue
    win = (f["diff"] > 0).astype(int)
    cov = (f["spreadscore"] > 0).astype(int)
    a_win = roc_auc_score(win, f["_p"])
    a_cov = roc_auc_score(cov, f["_p"])
    verdict = "h2h MONEYLINE" if a_win > a_cov else "run-line JUICE"
    verdicts[int(season)] = verdict
    print(f"{season:<8}{len(f):>7}{a_win:>13.4f}{a_cov:>15.4f}   {verdict}")

print("\n\nSame test on open_moneyline (SBR-sourced):")
print(f"{'season':<8}{'n':>7}{'AUC vs WIN':>13}{'AUC vs COVER':>15}   verdict")
print("-" * 68)
for season in seasons:
    f = novig(g[g["season"] == season], "open_moneyline")
    f = f.dropna(subset=["diff", "spreadscore"])
    f = f[f["spreadscore"] != 0]
    if len(f) < 200:
        continue
    win = (f["diff"] > 0).astype(int)
    cov = (f["spreadscore"] > 0).astype(int)
    a_win = roc_auc_score(win, f["_p"])
    a_cov = roc_auc_score(cov, f["_p"])
    print(f"{season:<8}{len(f):>7}{a_win:>13.4f}{a_cov:>15.4f}   "
          f"{'h2h MONEYLINE' if a_win > a_cov else 'run-line JUICE'}")

print("\n\nVerdict by season:")
for s, v in verdicts.items():
    print(f"  {s}: {v}")
print("\nSeasons that can supply a RUN-LINE price for cover EV:")
print("  " + ", ".join(str(s) for s, v in verdicts.items() if "JUICE" in v) or "  none")
print("Seasons that can supply an H2H price for win EV:")
print("  " + ", ".join(str(s) for s, v in verdicts.items() if "MONEYLINE" in v) or "  none")
