"""
Pilot: is the stored SBR "closing" price trustworthy, and is the inverse-steam
pattern real?

2025 analysis found that when the line moved TOWARD a side between 5am and
close, that side covered 42.9% of the time, versus 57.2% when it moved away.
Following steam would be a large losing edge and fading it a large winning one
-- which is backwards from how these markets are usually described.

That result rests entirely on the legacy `moneyline` column, which is the least
trustworthy field in the DB:
  * SBR's opening line scores a HIGHER AUC than its closing line, which is not
    physically sensible.
  * The 5am -> close move has a p90 of 16.9 probability points, far larger than
    real overnight run-line movement.
  * 5am (odds-api/fanduel) and closing (SBR) come from different sources, so
    "movement" conflates time with book and with any SBR matching errors.

This fetches a genuine LATE price from the SAME source as the 5am price, so the
comparison isolates time. One snapshot per date at 23:00Z (~7pm ET), keeping
only games whose first pitch is AFTER the snapshot, which guarantees a pre-game
price rather than an in-play one.

Cost 20 credits per date, hard-capped by --max-credits so it can never eat the
reserve needed for nightly runs.

    python pilot_closing_lines.py --dates 35            # dry run, shows plan
    python pilot_closing_lines.py --dates 35 --apply
"""
import argparse
import datetime
import json
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import db

BASE = "https://api.the-odds-api.com/v4"
SNAPSHOT_HOUR = "23:00:00Z"          # ~7pm ET: after day games, before most evening starts
PREFERRED = ["fanduel", "draftkings", "betmgm", "caesars", "pointsbet"]
ET = ZoneInfo("America/New_York")


def has_spreads(bm):
    return any(m.get("key") == "spreads" for m in bm.get("markets", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--dates", type=int, default=35)
    ap.add_argument("--max-credits", type=int, default=700)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--out", default="_pilot_closing.csv")
    args = ap.parse_args()

    key = json.load(open("data/config.txt"))["spreads"]["key_paid"]

    client = db.connect()
    try:
        games = db.fetch_games(client, "mlb", args.season)
    finally:
        client.close()

    g = games.dropna(subset=["spread_juice", "moneyline", "spreadscore", "diff"]).copy()
    g = g[g["spreadscore"] != 0]
    g["date_s"] = g["date"].astype(str).str[:10]
    # Spread evenly across the season rather than taking one stretch
    all_dates = sorted(g["date_s"].unique())
    if len(all_dates) > args.dates:
        idx = np.linspace(0, len(all_dates) - 1, args.dates).astype(int)
        dates = [all_dates[i] for i in sorted(set(idx))]
    else:
        dates = all_dates

    est = len(dates) * 20
    print(f"season {args.season}: {len(all_dates)} candidate dates, sampling {len(dates)}")
    print(f"estimated cost {est} credits (cap {args.max_credits})\n")

    lookup = {}
    for _, r in g.iterrows():
        lookup.setdefault((r["team"], r["date_s"]), []).append(r)
    for v in lookup.values():
        v.sort(key=lambda r: int(r["period"]))

    rows, spent, remaining = [], 0, None
    skipped_inplay = 0
    skipped_other_day = 0

    for i, d in enumerate(dates, 1):
        if spent + 20 > args.max_credits:
            print(f"\n  credit cap reached after {i-1} dates — stopping.")
            break
        try:
            r = requests.get(
                f"{BASE}/historical/sports/baseball_mlb/odds",
                params={"apiKey": key, "regions": "us", "markets": "h2h,spreads",
                        "oddsFormat": "american", "date": f"{d}T{SNAPSHOT_HOUR}"},
                timeout=30,
            )
        except Exception as exc:
            print(f"  {d}: request failed {exc}")
            continue
        spent += int(r.headers.get("x-requests-last", 20) or 20)
        remaining = r.headers.get("x-requests-remaining", remaining)
        if r.status_code != 200:
            print(f"  {d}: HTTP {r.status_code}")
            time.sleep(args.delay)
            continue

        payload = r.json()
        snap_iso = payload.get("timestamp")
        snap_dt = datetime.datetime.fromisoformat(snap_iso.replace("Z", "+00:00"))

        for game in payload.get("data", []):
            ct = game.get("commence_time")
            if not ct:
                continue
            start = datetime.datetime.fromisoformat(ct.replace("Z", "+00:00"))
            # The board lists several days ahead. Without this the snapshot gets
            # matched to a NEXT-DAY game whenever a team plays both days --
            # 60% of the first pilot run was wasted that way.
            if start.astimezone(ET).strftime("%Y-%m-%d") != d:
                skipped_other_day += 1
                continue
            # Only pre-game prices; an in-play line is worse than no line.
            if start <= snap_dt:
                skipped_inplay += 1
                continue
            bms = game.get("bookmakers", [])
            bm = next((b for w in PREFERRED for b in bms
                       if b.get("key") == w and has_spreads(b)), None) \
                or next((b for b in bms if has_spreads(b)), None)
            if bm is None:
                continue
            mk = {m["key"]: m for m in bm.get("markets", [])}
            if "spreads" not in mk:
                continue
            for o in mk["spreads"].get("outcomes", []):
                if o.get("point") is None or o.get("price") is None:
                    continue
                cand = lookup.get((o["name"], d))
                if not cand:
                    continue
                row = cand[0]
                rows.append({
                    "date": d, "team": o["name"],
                    "season": int(row["season"]), "period": int(row["period"]),
                    "diff": float(row["diff"]),
                    "spreadscore": float(row["spreadscore"]),
                    "late_point": float(o["point"]),
                    "late_price": float(o["price"]),
                    "price_5am": float(row["spread_juice"]),
                    "price_sbr_close": float(row["moneyline"]),
                    "minutes_before_start": round(
                        (start - snap_dt).total_seconds() / 60, 1),
                    "snapshot_ts": snap_iso,
                })
        if i % 10 == 0:
            print(f"  [{i}/{len(dates)}] rows={len(rows)} spent={spent} left={remaining}")
        time.sleep(args.delay)

    print(f"\ncredits spent : {spent}   remaining: {remaining}")
    print(f"rows captured : {len(rows)}")
    print(f"skipped (already in play at snapshot): {skipped_inplay}")
    print(f"skipped (board entry for another day): {skipped_other_day}")

    if not rows:
        print("nothing captured.")
        return

    out = pd.DataFrame(rows)
    if args.apply:
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    else:
        print("\nDRY RUN — data fetched but not written. Re-run with --apply to save.")
    print(f"\nmedian minutes before first pitch: {out['minutes_before_start'].median():.0f}")


if __name__ == "__main__":
    main()
