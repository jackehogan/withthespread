"""
Backfill observed run lines and both market prices from the paid odds-api.

Samples one historical snapshot per date at the SAME UTC hour the nightly
prediction job runs (09:00Z = 5am EDT). That matters for two reasons:

  1. EV must be priced against the odds actually obtainable at decision time.
     Backtesting on closing lines computes EV on prices never received.
  2. Closing lines encode lineups, scratches, weather and late money -- none of
     which exist at 5am. Using them in a 5am model is lookahead bias.

Games absent from the 09:00Z board were unbettable at decision time, so
skipping them makes a backtest match reality rather than flatter it.

Writes, per team-row:
    spread        observed run line (replaces ESPN's synthesised +/-1.5)
    spread_juice  price on the run line   -> cover EV, ml_implied_prob
    ml_odds       head-to-head moneyline  -> win EV
    spreadscore   recomputed as diff + spread
    odds_source   provenance, so a future mismatch is queryable
    snapshot_ts   the exact snapshot used

Cost is 20 credits per date (h2h + spreads, us region). Enforces --max-credits
and stops cleanly rather than silently exhausting the quota.

    python backfill_odds_api.py --seasons 2022 2023
    python backfill_odds_api.py --seasons 2022 2023 --apply --max-credits 8000
"""
import argparse
import json
import time

import pandas as pd
import requests
from pymongo import UpdateOne

import db

BASE = "https://api.the-odds-api.com/v4"
SNAPSHOT_HOUR = "09:00:00Z"          # matches the nightly cron (0 9 * * *)
PREFERRED = ["fanduel", "draftkings", "betmgm", "caesars", "pointsbet"]

# odds-api team naming vs statsapi
NAME_FIX = {"Oakland Athletics": "Oakland Athletics", "Athletics": "Athletics"}


def pick_bookmaker(game: dict) -> dict | None:
    """Prefer a major book that actually carries a spreads market."""
    bms = game.get("bookmakers", [])
    def has_spreads(b):
        return any(m.get("key") == "spreads" for m in b.get("markets", []))
    for want in PREFERRED:
        for b in bms:
            if b.get("key") == want and has_spreads(b):
                return b
    for b in bms:
        if has_spreads(b):
            return b
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-credits", type=int, default=8000)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--limit-dates", type=int, default=None)
    args = ap.parse_args()

    key = json.load(open("data/config.txt"))["spreads"]["key_paid"]

    client = db.connect()
    try:
        games = pd.concat(
            [db.fetch_games(client, "mlb", s) for s in args.seasons],
            ignore_index=True,
        )
    finally:
        client.close()
    games = games.dropna(subset=["diff"]).copy()
    games["date_s"] = games["date"].astype(str).str[:10]

    lookup: dict[tuple, list] = {}
    for _, r in games.iterrows():
        lookup.setdefault((r["team"], r["date_s"]), []).append(r)
    for v in lookup.values():
        v.sort(key=lambda r: int(r["period"]))

    dates = sorted(games["date_s"].unique())
    if args.limit_dates:
        dates = dates[: args.limit_dates]
    est = len(dates) * 20
    print(f"seasons {args.seasons}: {len(games)} rows, {len(dates)} dates")
    print(f"estimated cost: {est:,} credits (budget {args.max_credits:,})\n")
    if est > args.max_credits:
        print(f"  NOTE: estimate exceeds budget; will stop at the cap.\n")

    ops, spent, stats = [], 0, {
        "dates_ok": 0, "dates_empty": 0, "matched": 0, "unmatched": 0,
        "spread_changed": 0, "label_flipped": 0, "alt_lines": 0, "no_book": 0,
    }
    remaining = None

    for i, d in enumerate(dates, 1):
        if spent + 20 > args.max_credits:
            print(f"\n  credit cap reached after {i-1} dates — stopping cleanly.")
            break
        try:
            r = requests.get(
                f"{BASE}/historical/sports/baseball_mlb/odds",
                params={"apiKey": key, "regions": "us", "markets": "h2h,spreads",
                        "oddsFormat": "american", "date": f"{d}T{SNAPSHOT_HOUR}"},
                timeout=30,
            )
        except Exception as exc:
            print(f"  [{i}/{len(dates)}] {d}: request failed {exc}")
            time.sleep(args.delay)
            continue

        spent += int(r.headers.get("x-requests-last", 20) or 20)
        remaining = r.headers.get("x-requests-remaining", remaining)
        if r.status_code != 200:
            print(f"  [{i}/{len(dates)}] {d}: HTTP {r.status_code} {r.text[:120]}")
            time.sleep(args.delay)
            continue

        payload = r.json()
        snap_ts = payload.get("timestamp")
        board = payload.get("data", [])
        if not board:
            stats["dates_empty"] += 1
            time.sleep(args.delay)
            continue
        stats["dates_ok"] += 1

        # Order the board by start time so game 1 of a doubleheader is consumed
        # before game 2, matching the period ordering of the DB rows.
        board = sorted(board, key=lambda x: x.get("commence_time") or "")
        consumed: dict[tuple, int] = {}

        for g in board:
            bm = pick_bookmaker(g)
            if bm is None:
                stats["no_book"] += 1
                continue
            markets = {m["key"]: m for m in bm.get("markets", [])}
            spreads = markets.get("spreads")
            h2h = markets.get("h2h")
            if not spreads:
                continue

            h2h_price = {}
            if h2h:
                for o in h2h.get("outcomes", []):
                    h2h_price[NAME_FIX.get(o["name"], o["name"])] = o.get("price")

            for o in spreads.get("outcomes", []):
                team = NAME_FIX.get(o["name"], o["name"])
                point, price = o.get("point"), o.get("price")
                if point is None or price is None:
                    continue
                rows = lookup.get((team, d))
                if not rows:
                    stats["unmatched"] += 1
                    continue
                # Consume rows in period order: game 1 of a doubleheader takes
                # the earlier board entry, game 2 the later one.
                slot = consumed.get((team, d), 0)
                if slot >= len(rows):
                    stats["unmatched"] += 1
                    continue
                consumed[(team, d)] = slot + 1
                row = rows[slot]

                new_ss = round(float(row["diff"]) + float(point), 4)
                stats["matched"] += 1
                if abs(float(point)) != 1.5:
                    stats["alt_lines"] += 1
                if pd.notna(row.get("spread")) and float(row["spread"]) != float(point):
                    stats["spread_changed"] += 1
                if pd.notna(row.get("spreadscore")) and \
                        (float(row["spreadscore"]) > 0) != (new_ss > 0):
                    stats["label_flipped"] += 1

                patch = {
                    "spread": float(point),
                    "spread_juice": float(price),
                    "spreadscore": new_ss,
                    "odds_source": "odds-api-historical",
                    "snapshot_ts": snap_ts,
                }
                if team in h2h_price and h2h_price[team] is not None:
                    patch["ml_odds"] = float(h2h_price[team])

                ops.append(UpdateOne(
                    {"sport": "mlb", "team": row["team"],
                     "season": int(row["season"]), "period": int(row["period"])},
                    {"$set": patch}, upsert=False,
                ))

        if i % 25 == 0:
            print(f"  [{i}/{len(dates)}] matched={stats['matched']} "
                  f"flips={stats['label_flipped']} spent={spent} left={remaining}")
        time.sleep(args.delay)

    print(f"\ncredits spent        : {spent:,}   remaining: {remaining}")
    print(f"dates with a board   : {stats['dates_ok']}   empty: {stats['dates_empty']}")
    print(f"matched rows         : {stats['matched']}")
    print(f"unmatched odds rows  : {stats['unmatched']}")
    print(f"games with no spreads book: {stats['no_book']}")
    print(f"spread value changed : {stats['spread_changed']}")
    print(f"alternate lines      : {stats['alt_lines']}")
    print(f"COVER LABEL FLIPPED  : {stats['label_flipped']}"
          f"  ({stats['label_flipped']/max(stats['matched'],1)*100:.1f}%)")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
    elif ops:
        client = db.connect()
        try:
            res = client[db._MONGO_DB]["games"].bulk_write(ops)
            print(f"\nwrote {res.modified_count} rows")
        finally:
            client.close()


if __name__ == "__main__":
    main()
