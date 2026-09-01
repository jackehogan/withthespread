"""
Re-test inverse steam after dropping mispaired rows.

The MLB run line is fixed at +/-1.5; only the price moves. So if a row's late
snapshot carries a DIFFERENT signed point than its 5am row, the two prices are
opposite sides of the bet, not a move -- 188 rows (8%) are like that. The
capture pairs a date's late price with cand[0], the team's FIRST game that day,
so on doubleheaders the price of one game is attached to the other, whose
favourite can differ because the starters differ.

Those rows generate the largest spurious "moves", which is exactly where the
fade signal was strongest -- so they have to be excluded before the result
can be believed.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import db


def raw(v):
    v = float(v)
    return abs(v) / (abs(v) + 100.0) if v < 0 else 100.0 / (v + 100.0)


def devig(f, col):
    r = f[col].apply(raw)
    return r / f.groupby("game_pk")[col].transform(lambda s: s.apply(raw).sum())


p = pd.read_csv("_pilot_closing_usable.csv")
client = db.connect()
try:
    games = db.fetch_games(client, "mlb", 2025)
finally:
    client.close()
games["game_pk"] = games["game_pk"].astype(str)
games["spread"] = pd.to_numeric(games["spread"], errors="coerce")
j = p.merge(games[["team", "season", "period", "game_pk", "spread"]],
            on=["team", "season", "period"], how="left").dropna(subset=["game_pk"])

j["ok_side"] = j["late_point"] == j["spread"]
# a doubleheader = the team appears twice on the same calendar date
j["dh"] = j.groupby(["team", "date"])["game_pk"].transform("nunique") > 1
print(f"rows {len(j)}   side-mismatched {int((~j['ok_side']).sum())} "
      f"({(~j['ok_side']).mean()*100:.1f}%)")
print(f"  of the mismatched, on a doubleheader date: "
      f"{int((~j['ok_side'] & j['dh']).sum())} / {int((~j['ok_side']).sum())}")
print(f"  of the matched,    on a doubleheader date: "
      f"{int((j['ok_side'] & j['dh']).sum())} / {int(j['ok_side'].sum())}")


def run(d, title):
    d = d.groupby("game_pk").filter(lambda x: len(x) == 2).copy()
    d = d.dropna(subset=["price_5am", "late_price", "price_5am_ml",
                         "late_ml_price", "diff", "spreadscore"])
    d = d.groupby("game_pk").filter(lambda x: len(x) == 2)
    if len(d) < 100:
        print(f"\n{title}: only {len(d)} rows"); return
    print(f"\n=== {title} ===")
    print(f"  {len(d)} rows / {d['game_pk'].nunique()} games  ~SE {1/np.sqrt(len(d)):.4f}")
    for mk, p5, pl, lab in [("spreadscore", "price_5am", "late_price", "RUN LINE (cover)"),
                            ("diff", "price_5am_ml", "late_ml_price", "MONEYLINE (win)")]:
        y = (d[mk] > 0).astype(int)
        a, b = devig(d, p5), devig(d, pl)
        mv = b - a
        print(f"  {lab}")
        print(f"    5am AUC {roc_auc_score(y,a):.4f}   late AUC {roc_auc_score(y,b):.4f}"
              f"   move AUC {roc_auc_score(y,mv):.4f}")
        for thr in (0.02, 0.04):
            big = d[mv.abs() > thr]
            if len(big) < 60:
                continue
            bm = mv.loc[big.index]
            t = y.loc[big.index][bm > 0].mean()*100
            aw = y.loc[big.index][bm < 0].mean()*100
            nt, na = int((bm > 0).sum()), int((bm < 0).sum())
            print(f"      >{thr*100:.0f}pts n={len(big):>4}  TOWARD {t:5.1f}% (n={nt})"
                  f"  AWAY {aw:5.1f}% (n={na})  ~SE {1/np.sqrt(min(nt,na))*50:.1f}")


run(j, "ALL rows (what I reported)")
run(j[j["ok_side"]], "CLEAN — same side at both snapshots")
run(j[~j["ok_side"]], "MISPAIRED ONLY — should be noise")
