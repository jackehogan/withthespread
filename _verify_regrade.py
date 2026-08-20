"""Confirm every re-graded prediction now matches the actual game result."""
import numpy as np, pandas as pd, db

CUTOVER = "2026-08-05"
c = db.connect()
try:
    p = db.fetch_predictions(c, "mlb")
    g = db.fetch_games(c, "mlb")
finally:
    c.close()

p["date"] = p["prediction_date"].astype(str)
g["date"] = g["date"].astype(str)
for col in ("ev", "pnl", "ml_odds", "moneyline"):
    p[col] = pd.to_numeric(p[col], errors="coerce")
gr = p[p["pnl"].notna()].copy()

gg = g[["team", "date", "opponent", "diff", "spreadscore", "ml_odds", "spread_juice"]].copy()
for col in ("diff", "spreadscore", "ml_odds", "spread_juice"):
    gg[col] = pd.to_numeric(gg[col], errors="coerce")
gg = gg.rename(columns={"ml_odds": "odds_true", "spread_juice": "juice_true",
                        "diff": "diff_true", "spreadscore": "ss_true"})
gg = gg.drop_duplicates(subset=["team", "date"])

m = gr.merge(gg, on=["team", "date"], how="left", suffixes=("", "_g"))
m = m.dropna(subset=["diff_true"])

m["won_actual"] = np.where(m["bet"] == "ML", m["diff_true"] > 0, m["ss_true"] > 0)
m["won_recorded"] = m["pnl"] > 0
m["push"] = np.where(m["bet"] == "ML", m["diff_true"] == 0, m["ss_true"] == 0)
chk = m[~m["push"]]
agree = int((chk["won_actual"] == chk["won_recorded"]).sum())
print(f"graded predictions checked : {len(chk)}")
print(f"outcome matches real game  : {agree}/{len(chk)}")
print("RESULT:", "PASS" if agree == len(chk) else f"FAIL ({len(chk)-agree} wrong)")

if agree != len(chk):
    bad = chk[chk["won_actual"] != chk["won_recorded"]]
    print(bad[["team","date","bet","diff_true","ss_true","pnl"]].head(10).to_string(index=False))

print("\n\nCorrected record by era")
print(f"{'era':<28}{'bets':>7}{'ROI':>9}{'SE':>7}")
for lab, sub in (("run line (pre 08-05)", gr[gr["date"] < CUTOVER]),
                 ("moneyline (08-05 on)", gr[gr["date"] >= CUTOVER]),
                 ("ALL 2026", gr)):
    if sub.empty: continue
    roi = sub["pnl"].mean()*100
    se = sub["pnl"].std(ddof=1)/np.sqrt(len(sub))*100
    print(f"{lab:<28}{len(sub):>7}{roi:>8.1f}%{se:>6.1f}")

print("\nModerneyline era by EV threshold")
new = gr[gr["date"] >= CUTOVER]
for t in (0.0, 0.05, 0.10):
    s = new[new["ev"] > t]
    if len(s) < 3: continue
    roi = s["pnl"].mean()*100
    se = s["pnl"].std(ddof=1)/np.sqrt(len(s))*100
    print(f"  EV>{t:.2f}: {len(s):>4} bets  ROI {roi:+6.1f}%  (SE {se:.1f})")

# ungraded leftovers
un = p[p["pnl"].isna() & (p["ev"] > 0)]
print(f"\npositive-EV predictions still ungraded: {len(un)}")
if len(un):
    print(un.groupby(un["date"] >= CUTOVER).size().rename({False:"pre-cutover",True:"post-cutover"}).to_string())
    print("  most recent dates:", sorted(un["date"].unique())[-5:])
