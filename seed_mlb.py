"""
Backfill MLB game data into MongoDB.

Steps per season
----------------
1. Fetch game results from statsapi (free, no key required).
2. Load run lines from a local SBR Excel/CSV file.
3. Fetch prior-season pitcher stats from statsapi and attach to each game row.
4. Compute SpreadScore = run_diff + run_line.
5. Upsert everything into the games collection.

SBR file setup
--------------
Download one Excel file per season from:
  https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlboddsarchives.htm

Save to data/sbr/ with names like:
  mlb_2021.xlsx, mlb_2022.xlsx, mlb_2023.xlsx, mlb_2024.xlsx

Usage
-----
    python seed_mlb.py --seasons 2021 2022 2023 2024
    python seed_mlb.py --seasons 2024 --sbr-dir data/sbr
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

import db
import data_pipeline as dp
from config import SPORTS


def seed_mlb_season(
    season: int,
    sbr_dir: str = "data/sbr",
    request_delay: float = 0.5,
) -> None:
    """
    Seed one MLB season into MongoDB.

    Parameters
    ----------
    season        : Calendar year (e.g. 2023 for the 2023 MLB season).
    sbr_dir       : Directory containing downloaded SBR Excel/CSV files.
    request_delay : Seconds between statsapi calls (politeness).
    """
    print(f"\n{'=' * 60}")
    print(f"Seeding MLB season {season}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 1. Game results from statsapi
    # ------------------------------------------------------------------
    print("[1/4] Fetching game results from statsapi...")
    try:
        raw = dp.fetch_season_games_mlb(season)
    except Exception as exc:
        print(f"  fetch_season_games_mlb failed: {exc}")
        return

    if not raw:
        print("  No completed games returned.")
        return

    game_df = dp.parse_game_results_mlb(raw, season)
    if game_df.empty:
        print("  No valid games after filtering.")
        return

    periods = sorted(game_df["period"].dropna().unique().astype(int))
    print(f"  {len(game_df)} game rows across {len(periods)} periods "
          f"(periods {min(periods)}-{max(periods)}).")

    # ------------------------------------------------------------------
    # 2. Run lines from SBR CSV
    # ------------------------------------------------------------------
    print(f"[2/4] Loading SBR run lines from {sbr_dir}/...")
    sbr_df = dp.load_sbr_mlb(season, sbr_dir=sbr_dir)

    if not sbr_df.empty:
        # Join run line and moneyline to game_df by (team, date)
        sbr_indexed = sbr_df.set_index(["team", "date"])[["run_line", "moneyline"]]
        game_df["run_line"]  = game_df.set_index(["team", "date"]).index.map(
            sbr_indexed.get("run_line", pd.Series(dtype=float))
        ).values
        game_df["moneyline"] = game_df.set_index(["team", "date"]).index.map(
            sbr_indexed.get("moneyline", pd.Series(dtype=float))
        ).values
        n_with_lines = game_df["run_line"].notna().sum()
        print(f"  Matched run lines for {n_with_lines}/{len(game_df)} game rows.")
    else:
        game_df["run_line"]  = None
        game_df["moneyline"] = None
        print("  No SBR file found — storing games without run lines.")

    # Compute SpreadScore = run_diff + run_line
    game_df["spread"] = game_df["run_line"]   # 'spread' is the model's field name
    game_df["spreadscore"] = (
        game_df["diff"] + game_df["run_line"]
        if "run_line" in game_df.columns
        else None
    )
    mask = game_df["run_line"].notna()
    game_df.loc[mask, "spreadscore"] = (
        game_df.loc[mask, "diff"] + game_df.loc[mask, "run_line"]
    )
    game_df.loc[~mask, "spreadscore"] = None

    n_scored = game_df["spreadscore"].notna().sum()
    print(f"  SpreadScore computed for {n_scored}/{len(game_df)} game rows.")

    # ------------------------------------------------------------------
    # 3. Pitcher stats (prior season) — attach to each game row
    # ------------------------------------------------------------------
    print(f"[3/4] Fetching pitcher stats for {season - 1} (prior season)...")
    time.sleep(request_delay)
    pitcher_stats = dp.fetch_mlb_pitcher_stats(season - 1)

    if pitcher_stats.empty:
        print("  No pitcher stats returned — sp_era/sp_whip will be NaN.")
    else:
        print(f"  Loaded stats for {len(pitcher_stats)} pitchers.")

    def _lookup_pitcher(name: str, col: str) -> float | None:
        if not name or pitcher_stats.empty or name not in pitcher_stats.index:
            return None
        val = pitcher_stats.loc[name, col]
        return float(val) if pd.notna(val) else None

    game_df["sp_era"]  = game_df["sp_name"].apply(lambda n: _lookup_pitcher(n, "era"))
    game_df["sp_whip"] = game_df["sp_name"].apply(lambda n: _lookup_pitcher(n, "whip"))
    game_df["sp_k9"]   = game_df["sp_name"].apply(lambda n: _lookup_pitcher(n, "k9"))

    n_sp = game_df["sp_era"].notna().sum()
    print(f"  Pitcher ERA matched for {n_sp}/{len(game_df)} game rows.")

    # ------------------------------------------------------------------
    # 4. Upsert to MongoDB
    # ------------------------------------------------------------------
    print("[4/4] Upserting to MongoDB...")
    client = db.connect()
    try:
        records = game_df.to_dict("records")
        db.upsert_games(client, records)
        print(f"  Upserted {len(records)} records for MLB {season}.")
    finally:
        client.close()


def seed_mlb_pitcher_stats(
    season: int,
    request_delay: float = 0.5,
) -> None:
    """
    Refresh pitcher stats on existing game rows for a season.

    Useful if you want to update pitcher stats after seed_mlb_season() was
    run without them, or to update with in-season stats as the season progresses.

    Uses PRIOR SEASON stats (season - 1) to avoid leakage.
    """
    print(f"\nRefreshing pitcher stats for MLB {season} (using {season - 1} stats)...")
    pitcher_stats = dp.fetch_mlb_pitcher_stats(season - 1)
    time.sleep(request_delay)

    if pitcher_stats.empty:
        print("  No pitcher stats returned.")
        return

    client = db.connect()
    try:
        games_df = db.fetch_games(client, "mlb", season)
        if games_df.empty or "sp_name" not in games_df.columns:
            print("  No MLB games found in DB for this season.")
            return

        updates = []
        for _, row in games_df.iterrows():
            sp = row.get("sp_name", "")
            if not sp or sp not in pitcher_stats.index:
                continue
            stats = pitcher_stats.loc[sp]
            update = {
                "sport": "mlb", "team": row["team"],
                "season": row["season"], "period": row["period"],
            }
            for col in ["era", "whip", "k9"]:
                if col in stats.index and pd.notna(stats[col]):
                    update[f"sp_{col}"] = float(stats[col])
            if len(update) > 4:
                updates.append(update)

        if updates:
            db.upsert_games(client, updates)
            print(f"  Updated pitcher stats on {len(updates)} game rows.")
        else:
            print("  No pitcher matches found.")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed MLB historical game data.")
    parser.add_argument(
        "--seasons", type=int, nargs="+", required=True,
        help="Calendar years to seed, e.g. 2021 2022 2023 2024. "
             "Training starts from 2021 (post-COVID; 2020 was a 60-game anomaly)."
    )
    parser.add_argument(
        "--sbr-dir", type=str, default="data/sbr",
        dest="sbr_dir",
        help="Directory containing SBR Excel files (default: data/sbr)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between API calls (default: 0.5)"
    )
    parser.add_argument(
        "--pitcher-refresh", action="store_true", dest="pitcher_refresh",
        help="Only refresh pitcher stats on existing game rows, skip game seeding"
    )
    args = parser.parse_args()

    for season in sorted(args.seasons):
        if args.pitcher_refresh:
            seed_mlb_pitcher_stats(season, request_delay=args.delay)
        else:
            seed_mlb_season(season, sbr_dir=args.sbr_dir, request_delay=args.delay)
        time.sleep(args.delay)
