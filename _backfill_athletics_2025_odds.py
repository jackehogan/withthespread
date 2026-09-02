"""
Re-fetch 2025 Athletics moneylines.

2025 was the club's first season keyed as "Athletics" in the DB, while the
odds-api snapshots for that season still said "Oakland Athletics". The seeder
matched odds on (team, date) with no alias handling, so the A's side never
matched: 3 of 162 games carry a moneyline, against 96% on the opponent row of
the very same games. It is the only team-season in the DB below 50%.

The join now resolves franchise aliases; this refills the rows that were lost
before the fix. Only rows currently missing a price are written.

    python _backfill_athletics_2025_odds.py            # dry run, shows cost
    python _backfill_athletics_2025_odds.py --apply
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from pymongo import UpdateOne

import data_pipeline as dp
import db

APPLY = "--apply" in sys.argv
MAX_CREDITS = 3600


def main():
    client = db.connect()
    try:
        g = db.fetch_games(client, "mlb")
        g["ml_odds"] = pd.to_numeric(g["ml_odds"], errors="coerce")
        a = g[(g["season"] == 2025) & (g["team"] == "Athletics")]
        need = a[a["ml_odds"].isna()]
        dates = sorted(need["date"].astype(str).str[:10].unique())
        print(f"Athletics 2025: {len(a)} rows, {len(need)} missing a moneyline")
        print(f"dates to fetch: {len(dates)}  ->  ~{len(dates) * 20} credits "
              f"(cap {MAX_CREDITS})")
        if len(dates) * 20 > MAX_CREDITS:
            print("ABORT: over the credit cap."); return
        if not APPLY:
            print("\nDRY RUN - rerun with --apply to spend.")
            return

        odds = dp.fetch_mlb_odds_api(dates, request_delay=0.3, max_dates=None)
        if odds.empty:
            print("no odds returned"); return
        omap = (odds.drop_duplicates(subset=["team", "date"], keep="first")
                    .set_index(["team", "date"])
                    [["run_line", "spread_juice", "ml_odds", "snapshot_ts"]]
                    .to_dict("index"))
        print(f"fetched {len(odds)} odds rows across {odds['date'].nunique()} dates")

        ops, filled, alias = [], 0, 0
        for _, r in need.iterrows():
            d = str(r["date"])[:10]
            rec = omap.get((r["team"], d))
            if rec is None:
                for alt in dp._FRANCHISE_ALIASES.get(r["team"], []):
                    rec = omap.get((alt, d))
                    if rec is not None:
                        alias += 1
                        break
            if rec is None or rec.get("ml_odds") is None:
                continue
            payload = {k: v for k, v in rec.items() if v is not None and v == v}
            payload["odds_source"] = "odds-api-historical"
            ops.append(UpdateOne(
                {"sport": "mlb", "team": r["team"],
                 "season": int(r["season"]), "period": int(r["period"])},
                {"$set": payload}))
            filled += 1
        print(f"rows to fill: {filled} (of which {alias} matched via an alias)")
        if ops:
            client[db._MONGO_DB]["games"].bulk_write(ops)
            print(f"applied {len(ops)} updates")
    finally:
        client.close()


if __name__ == "__main__":
    main()
