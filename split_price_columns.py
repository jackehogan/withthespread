"""
Split the overloaded `moneyline` column into two explicit ones.

`moneyline` has held different quantities depending on which feed seeded the
season. Classified empirically by whether its implied probability predicts
winning or covering:

    2022  h2h moneyline    (AUC vs win 0.5436 / vs cover 0.5133)
    2023  h2h moneyline    (0.5249 / 0.5239 -- weak either way)
    2024  run-line juice   (0.4667 / 0.6028)
    2025  run-line juice   (0.5061 / 0.6417)
    2026  h2h moneyline    (0.5693 / 0.4429)

This matters because ml_implied_prob -- the model's most influential feature --
is derived from it, so the feature silently means P(win) in some seasons and
P(cover) in others.

Writes two new fields and leaves `moneyline` untouched, so nothing breaks and
the change is reversible:

    spread_juice : price on the run line   -> cover EV, ml_implied_prob
    ml_odds      : head-to-head moneyline  -> win EV

open_moneyline is run-line sourced (SBR) in every season that has it.

    python split_price_columns.py           # dry run
    python split_price_columns.py --apply
"""
import argparse

import pandas as pd
from pymongo import UpdateOne

import db

# Empirically classified — see _audit_price_columns.py
SEASON_MEANING = {
    2022: "ml_odds",
    2023: "ml_odds",
    2024: "spread_juice",
    2025: "spread_juice",
    2026: "ml_odds",
}

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()

client = db.connect()
try:
    col = client[db._MONGO_DB]["games"]
    docs = list(col.find(
        {"sport": "mlb"},
        {"_id": 1, "season": 1, "moneyline": 1, "open_moneyline": 1,
         "spread_juice": 1, "ml_odds": 1},
    ))
    df = pd.DataFrame(docs)
    print(f"game rows: {len(df)}\n")

    ops, counts = [], {}
    for r in docs:
        season = int(r.get("season", 0))
        target = SEASON_MEANING.get(season)
        if target is None:
            continue
        patch = {}
        if r.get("moneyline") is not None and r.get(target) is None:
            patch[target] = float(r["moneyline"])
        # open_moneyline is run-line priced wherever it exists
        if r.get("open_moneyline") is not None and "spread_juice" not in patch:
            pass  # left as-is; it is already a distinct column
        if patch:
            ops.append(UpdateOne({"_id": r["_id"]}, {"$set": patch}))
            k = (season, target)
            counts[k] = counts.get(k, 0) + 1

    print("Rows that would gain an explicit price column:")
    for (season, target), n in sorted(counts.items()):
        print(f"  {season} -> {target:<13} {n:>6}")
    print(f"\n  total: {len(ops)}")

    print("\nResulting availability:")
    print("  cover EV possible (spread_juice) : "
          + ", ".join(str(s) for s, t in SEASON_MEANING.items() if t == "spread_juice"))
    print("  win EV possible   (ml_odds)      : "
          + ", ".join(str(s) for s, t in SEASON_MEANING.items() if t == "ml_odds"))
    print("  both prices                      : none")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
    elif ops:
        res = col.bulk_write(ops)
        print(f"\nwrote {res.modified_count} rows")
    else:
        print("\nNothing to write.")
finally:
    client.close()
