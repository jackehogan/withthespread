"""
Renumber period to a contiguous 1..N per (team, season).

Deleting duplicate rows left holes where the inflated period numbers had been
-- 832 of them in 2026, across 29 of 30 teams. period indexes the SpreadScore
pivot, so a hole becomes a phantom NaN lag and degrades 1_ago_ss.

Ordering is by (date, game_pk). game_pk ascending puts game 1 of a doubleheader
before game 2, matching how backfill_game_pk assigns them.

Two-phase write: periods are first parked in a high range, then written to
their final values. A direct update would transiently collide with the unique
index on (sport, team, season, period).

Predictions store period to resolve results, so they are remapped through
game_pk in the same run.

    python renumber_periods.py           # dry run
    python renumber_periods.py --apply
"""
import argparse

import pandas as pd
from pymongo import UpdateOne

import db

PARK = 100000   # temporary offset to dodge the unique index

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
parser.add_argument("--seasons", type=int, nargs="*", default=None)
args = parser.parse_args()

client = db.connect()
try:
    col = client[db._MONGO_DB]["games"]
    q = {"sport": "mlb"}
    if args.seasons:
        q["season"] = {"$in": args.seasons}

    docs = list(col.find(q, {"_id": 1, "team": 1, "season": 1, "period": 1,
                             "date": 1, "game_pk": 1}))
    df = pd.DataFrame(docs)
    print(f"game rows under consideration: {len(df)}")

    df["date_s"] = df["date"].astype(str).str[:10]
    df["pk_s"] = df["game_pk"].astype(str)
    df = df.sort_values(["season", "team", "date_s", "pk_s"])
    df["new_period"] = df.groupby(["season", "team"]).cumcount() + 1

    changed = df[df["period"] != df["new_period"]]
    print(f"rows whose period changes: {len(changed)}\n")

    print("Per season:")
    summary = df.groupby("season").agg(
        rows=("_id", "size"),
        changing=("period", lambda s: 0),
    )
    summary["changing"] = changed.groupby("season").size().reindex(summary.index).fillna(0).astype(int)
    summary["new_max"] = df.groupby("season")["new_period"].max()
    summary["old_max"] = df.groupby("season")["period"].max()
    print(summary.to_string())

    if not changed.empty:
        print("\nsample of changes:")
        print(changed[["season", "team", "date_s", "period", "new_period"]]
              .head(10).to_string(index=False))

    # Remap predictions through game_pk
    old_key_to_pk = {
        (r["team"], int(r["season"]), int(r["period"])): r["pk_s"]
        for r in df.to_dict("records")
    }
    pk_to_new = {
        (r["team"], int(r["season"]), r["pk_s"]): int(r["new_period"])
        for r in df.to_dict("records")
    }

    pcol = client[db._MONGO_DB]["predictions"]
    pdocs = list(pcol.find({"sport": "mlb"},
                           {"_id": 1, "team": 1, "season": 1, "period": 1}))
    pred_ops, unmatched = [], 0
    for p in pdocs:
        k = (p.get("team"), int(p.get("season", 0)), int(p.get("period", 0)))
        pk = old_key_to_pk.get(k)
        if pk is None:
            unmatched += 1
            continue
        new_p = pk_to_new.get((p["team"], int(p["season"]), pk))
        if new_p is not None and new_p != p.get("period"):
            pred_ops.append(UpdateOne({"_id": p["_id"]},
                                      {"$set": {"period": int(new_p)}}))
    print(f"\npredictions: {len(pdocs)} total, "
          f"{len(pred_ops)} need remapping, {unmatched} unmatched")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
    else:
        # phase 1: park everything out of the way
        col.bulk_write([
            UpdateOne({"_id": r["_id"]},
                      {"$set": {"period": int(r["new_period"]) + PARK}})
            for r in df.to_dict("records")
        ])
        # phase 2: settle on final values
        col.bulk_write([
            UpdateOne({"_id": r["_id"]}, {"$set": {"period": int(r["new_period"])}})
            for r in df.to_dict("records")
        ])
        print(f"\nrenumbered {len(df)} game rows")
        if pred_ops:
            pcol.bulk_write(pred_ops)
            print(f"remapped {len(pred_ops)} prediction rows")
finally:
    client.close()
