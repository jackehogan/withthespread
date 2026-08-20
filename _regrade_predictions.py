"""
Re-score 2026 MLB predictions after the date-keyed lookup fix.

update_prediction_results only touches documents missing `covered`, so the
stored grades must be cleared before it will recompute them. The existing
values are written to a timestamped JSON backup first -- this is a bulk write
to live data and must be reversible.

    python _regrade_predictions.py            # dry run: report, change nothing
    python _regrade_predictions.py --apply    # back up, clear, re-grade
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import db

SPORT, SEASON = "mlb", 2026
APPLY = "--apply" in sys.argv
FIELDS = ["covered", "bet_won", "pnl", "result_spreadscore", "result_period"]

client = db.connect()
pred_col = client[db._MONGO_DB]["predictions"]

q = {"sport": SPORT, "season": SEASON, "covered": {"$exists": True}}
before = list(pred_col.find(q, {"_id": 0, "team": 1, "prediction_date": 1,
                                "period": 1, "ev": 1, "bet": 1, **{f: 1 for f in FIELDS}}))
print(f"graded 2026 predictions currently stored: {len(before)}")

bdf = pd.DataFrame(before)
if not bdf.empty:
    bdf["pnl"] = pd.to_numeric(bdf["pnl"], errors="coerce")
    n = int(bdf["pnl"].notna().sum())
    print(f"  ROI as currently recorded: {bdf['pnl'].mean()*100:+.2f}%  ({n} bets)")

if not APPLY:
    print("\nDRY RUN — nothing changed. Re-run with --apply to back up and re-grade.")
    client.close()
    sys.exit(0)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = f"_predictions_grades_backup_{stamp}.json"
with open(backup, "w", encoding="utf-8") as fh:
    json.dump(before, fh, indent=1, default=str)
print(f"\nbacked up {len(before)} graded docs -> {backup}")

res = pred_col.update_many(
    {"sport": SPORT, "season": SEASON},
    {"$unset": {f: "" for f in FIELDS}},
)
print(f"cleared grades on {res.modified_count} documents")

updated = db.update_prediction_results(client, SPORT, SEASON)
print(f"re-graded {updated} predictions")

after = list(pred_col.find(q, {"_id": 0, "team": 1, "prediction_date": 1,
                               "period": 1, "result_period": 1, "pnl": 1, "ev": 1}))
adf = pd.DataFrame(after)
adf["pnl"] = pd.to_numeric(adf["pnl"], errors="coerce")
print(f"\ngraded after re-run: {len(adf)}")
print(f"  ROI now recorded : {adf['pnl'].mean()*100:+.2f}%  ({int(adf['pnl'].notna().sum())} bets)")

if "result_period" in adf.columns:
    m = adf["result_period"].notna()
    same = int((adf.loc[m, "period"] == adf.loc[m, "result_period"]).sum())
    print(f"  pred.period == graded game period: {same}/{int(m.sum())} "
          f"(the rest are what the old period-keyed lookup got wrong)")

client.close()
