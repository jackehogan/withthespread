"""
Backfill NBA advanced team ratings (off/def/net rating) into the games collection.

For each (season, period), fetches cumulative ratings from stats.nba.com as of
the last game date of the *previous* period — so features contain no leakage
when the model predicts period T.

Usage
-----
    python seed_ratings.py --seasons 2022 2023 2024
    python seed_ratings.py --seasons 2024            # single season
"""

from __future__ import annotations

import argparse
import time

import db
import data_pipeline as dp
from config import SPORTS
from pipeline import _fetch_games_filtered


def seed_ratings(season: int, request_delay: float = 0.7) -> None:
    sport = SPORTS["nba"]
    client = db.connect()

    try:
        season_games = _fetch_games_filtered(client, sport, season)
    except Exception as exc:
        client.close()
        print(f"  Could not fetch games for {season}: {exc}")
        return

    if season_games.empty:
        client.close()
        print(f"  No games found for {season}. Skipping.")
        return

    # Build period -> last date of the PREVIOUS period mapping
    # For period T, we want ratings accumulated through end of T-1
    periods = sorted(season_games["period"].dropna().unique().astype(int))
    period_last_date: dict[int, str] = {}
    for p in periods:
        prev = season_games[season_games["period"] == p - 1]["date"].dropna()
        if not prev.empty:
            period_last_date[p] = str(prev.max())[:10]  # YYYY-MM-DD

    print(f"\nSeeding NBA ratings for {season} ({len(periods)} periods, "
          f"{len(period_last_date)} with prior data)...")

    total_updated = 0
    for period in periods:
        date_to = period_last_date.get(period)
        if date_to is None:
            # Period 1 — no prior games, skip (ratings will be NaN)
            continue

        ratings = dp.fetch_nba_ratings(season, date_to)
        time.sleep(request_delay)  # be polite to stats.nba.com

        if ratings.empty:
            print(f"  Period {period:3d}: no ratings returned (date_to={date_to})")
            continue

        n = db.upsert_game_ratings(client, "nba", season, int(period), ratings)
        total_updated += n
        print(f"  Period {period:3d}: {n:2d} game docs updated  (as of {date_to})")

    client.close()
    print(f"  Done. {total_updated} total game docs updated for {season}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill NBA team ratings.")
    parser.add_argument(
        "--seasons", type=int, nargs="+", required=True,
        help="Season start years to seed (e.g. 2022 2023 2024)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.7,
        help="Seconds between API calls (default 0.7)"
    )
    args = parser.parse_args()

    for season in args.seasons:
        seed_ratings(season, request_delay=args.delay)
