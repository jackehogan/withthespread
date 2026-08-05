"""
Replace synthesised ESPN lines with observed SBR run lines and prices.

ESPN's pickcenter exposes no run-line price, so _parse_espn_event invents the
line from a favorite boolean (-1.5 / +1.5) and stores the H2H moneyline. Any
ESPN-seeded season therefore has:

  * a spread that was inferred, never observed (2026 has zero alternate lines)
  * spreadscore -- the training target -- built on that inferred line
  * ml_implied_prob meaning P(win) rather than P(cover)

SBR publishes the real run line, the run-line juice, and the opening line.
This refreshes them regardless of whether spreadscore is already populated,
which is why seed_mlb_odds_sbr_web cannot be reused: it only fills gaps, so
ESPN's wrong-but-present values keep it from ever running.

Writes spread, spread_juice, spreadscore, open_spread, open_moneyline.
Leaves ml_odds alone -- ESPN's H2H price is still valid for win EV.

    python refresh_sbr_prices.py --season 2026
    python refresh_sbr_prices.py --season 2026 --apply
"""
import argparse
import time

import numpy as np
import pandas as pd
from pymongo import UpdateOne

import db
import data_pipeline as dp

parser = argparse.ArgumentParser()
parser.add_argument("--season", type=int, required=True)
parser.add_argument("--apply", action="store_true")
parser.add_argument("--delay", type=float, default=0.4)
parser.add_argument("--limit-dates", type=int, default=None,
                    help="only process the first N dates (for a quick look)")
args = parser.parse_args()

client = db.connect()
try:
    games = db.fetch_games(client, "mlb", args.season)
finally:
    client.close()

games = games.dropna(subset=["diff"]).copy()
games["date_s"] = games["date"].astype(str).str[:10]
dates = sorted(games["date_s"].unique())
if args.limit_dates:
    dates = dates[: args.limit_dates]
print(f"season {args.season}: {len(games)} rows across {len(dates)} dates\n")

# (team, date) -> rows ordered by period, to place doubleheaders correctly
lookup: dict[tuple, list] = {}
for _, r in games.iterrows():
    lookup.setdefault((r["team"], r["date_s"]), []).append(r)
for v in lookup.values():
    v.sort(key=lambda r: int(r["period"]))

ops = []
stats = {"matched": 0, "no_sbr": 0, "unmatched": 0,
         "spread_changed": 0, "label_flipped": 0, "alt_lines": 0}

for i, d in enumerate(dates, 1):
    try:
        odds = dp.fetch_mlb_odds_sbr_web(d)
    except Exception as exc:
        print(f"  [{i}/{len(dates)}] {d}: FAILED {exc}")
        time.sleep(args.delay)
        continue
    if odds.empty:
        stats["no_sbr"] += 1
        time.sleep(args.delay)
        continue

    consumed: dict[tuple, int] = {}
    for _, o in odds.iterrows():
        key = (o["team"], d)
        rows = lookup.get(key)
        if not rows:
            stats["unmatched"] += 1
            continue
        idx = consumed.get(key, 0)
        if idx >= len(rows):
            continue
        consumed[key] = idx + 1
        row = rows[idx]

        new_spread = float(o["run_line"])
        new_ss = round(float(row["diff"]) + new_spread, 4)
        patch = {
            "spread": new_spread,
            "spread_juice": float(o["moneyline"]),
            "spreadscore": new_ss,
        }
        if pd.notna(o.get("open_spread")):
            patch["open_spread"] = float(o["open_spread"])
        if pd.notna(o.get("open_moneyline")):
            patch["open_moneyline"] = float(o["open_moneyline"])

        stats["matched"] += 1
        if abs(new_spread) != 1.5:
            stats["alt_lines"] += 1
        old_spread = row.get("spread")
        if pd.notna(old_spread) and float(old_spread) != new_spread:
            stats["spread_changed"] += 1
        old_ss = row.get("spreadscore")
        if pd.notna(old_ss) and (float(old_ss) > 0) != (new_ss > 0):
            stats["label_flipped"] += 1

        ops.append(UpdateOne(
            {"sport": "mlb", "team": row["team"],
             "season": int(row["season"]), "period": int(row["period"])},
            {"$set": patch}, upsert=False,
        ))

    if i % 20 == 0:
        print(f"  [{i}/{len(dates)}] matched={stats['matched']} "
              f"flips={stats['label_flipped']}")
    time.sleep(args.delay)

print(f"\nmatched rows          : {stats['matched']}")
print(f"dates with no SBR data: {stats['no_sbr']}")
print(f"SBR rows unmatched    : {stats['unmatched']}")
print(f"spread value changed  : {stats['spread_changed']}")
print(f"alternate lines found : {stats['alt_lines']}  (ESPN could never produce these)")
print(f"COVER LABEL FLIPPED   : {stats['label_flipped']}"
      f"  ({stats['label_flipped']/max(stats['matched'],1)*100:.1f}% of matched)")

if not args.apply:
    print("\nDRY RUN — nothing written. Re-run with --apply.")
elif ops:
    client = db.connect()
    try:
        res = client[db._MONGO_DB]["games"].bulk_write(ops)
        print(f"\nwrote {res.modified_count} rows")
    finally:
        client.close()
