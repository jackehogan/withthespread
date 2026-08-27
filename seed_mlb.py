"""
Backfill MLB game data into MongoDB.

Data sources (all free, no API key required)
--------------------------------------------
- **statsapi** (stats.mlb.com)   : game scores, dates, probable pitchers
- **ESPN internal API**          : run lines, moneylines, over/under
- **statsapi stats endpoint**    : prior-season pitcher ERA/WHIP/K9

Steps per season
----------------
1. Fetch game results + probable pitchers from statsapi.
2. Fetch run lines and moneylines from ESPN (one call per game, ~15 min/season).
3. Merge scores + odds by (team, date).
4. Fetch prior-season pitcher stats and attach to each game row.
5. Compute SpreadScore = run_diff + run_line.
6. Upsert everything into the games collection.

Usage
-----
    python seed_mlb.py --seasons 2022 2023 2024 2025
    python seed_mlb.py --seasons 2025                 # current season only
    python seed_mlb.py --seasons 2024 --skip-odds     # scores + pitchers only
    python seed_mlb.py --seasons 2024 --pitcher-refresh  # refresh pitcher stats only
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
    skip_odds: bool = False,
    request_delay: float = 0.25,
    since: str | None = None,
    max_odds_dates: int | None = 5,
) -> None:
    """
    Seed one MLB season into MongoDB.

    Parameters
    ----------
    season        : Calendar year (e.g. 2023).
    skip_odds     : If True, skip ESPN odds fetch (only seed scores + pitchers).
    request_delay : Seconds between ESPN API calls (default 0.25).
    since         : Optional 'YYYY-MM-DD' lower bound — only fetch games on or
                    after this date. Use for nightly incremental updates.
    """
    print(f"\n{'=' * 60}")
    print(f"Seeding MLB season {season}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 1. Game results + probable pitchers from statsapi
    # ------------------------------------------------------------------
    print("[1/4] Fetching game results from statsapi...")
    try:
        raw = dp.fetch_season_games_mlb(season, since=since)
    except Exception as exc:
        print(f"  fetch_season_games_mlb failed: {exc}")
        return

    if not raw:
        print("  No completed games returned.")
        return

    # Incremental (--since) runs always overlap games that are already seeded,
    # because the since-date is the last date in the DB. Pass the existing
    # (team, game_pk) -> period map so those games keep their stored period
    # instead of being re-numbered into duplicate rows; period_offsets then
    # only sets where genuinely new games start. A full-season reseed needs
    # neither — cumcount already numbers every game correctly from 1.
    period_offsets: dict[str, int] = {}
    known_periods: dict[tuple[str, str], int] = {}
    if since:
        _offset_client = db.connect()
        try:
            _existing = db.fetch_games(_offset_client, "mlb", season)
        finally:
            _offset_client.close()
        if not _existing.empty and "period" in _existing.columns:
            period_offsets = _existing.groupby("team")["period"].max().to_dict()
            if "game_pk" in _existing.columns:
                _pk_rows = _existing[
                    _existing["game_pk"].notna()
                    & (_existing["game_pk"].astype(str) != "")
                ]
                known_periods = {
                    (t, str(pk)): int(p)
                    for t, pk, p in zip(
                        _pk_rows["team"], _pk_rows["game_pk"], _pk_rows["period"]
                    )
                }

    game_df = dp.parse_game_results_mlb(
        raw, season, period_offsets=period_offsets, known_periods=known_periods
    )
    if game_df.empty:
        print("  No valid games after filtering.")
        return

    periods = sorted(game_df["period"].dropna().unique().astype(int))
    print(f"  {len(game_df)} game rows across {len(periods)} periods "
          f"(periods {min(periods)}-{max(periods)}).")

    # ------------------------------------------------------------------
    # 2. Run lines + moneylines from ESPN — only for dates not yet in DB
    # ------------------------------------------------------------------
    if skip_odds:
        print("[2/4] Skipping odds fetch (--skip-odds).")
        game_df["spread"]      = None
        game_df["moneyline"]   = None
        game_df["over_under"]  = None
        game_df["spreadscore"] = None
    else:
        # Check which specific (team, period) games are already seeded in MongoDB.
        # Only fetch odds for dates that contain genuinely missing games.
        _check_client = db.connect()
        try:
            existing_df = db.fetch_games(_check_client, "mlb", season)
        finally:
            _check_client.close()

        if not existing_df.empty and "spreadscore" in existing_df.columns:
            seeded_keys = set(
                zip(
                    existing_df.loc[existing_df["spreadscore"].notna(), "team"],
                    existing_df.loc[existing_df["spreadscore"].notna(), "period"].astype(int),
                )
            )
        else:
            seeded_keys = set()

        # Filter to only (team, period) rows that are genuinely missing
        game_df = game_df[
            [(t, int(p)) not in seeded_keys for t, p in zip(game_df["team"], game_df["period"])]
        ].copy()

        if game_df.empty:
            print(f"  All completed games already seeded — nothing to do.")
            return

        missing_dates = set(game_df["date"].dropna().astype(str).unique())

        # A nightly run must only pay for the days it is actually seeding.
        # Without this, one stale date anywhere in the season would pull the
        # whole backlog at 20 credits each.
        if since:
            missing_dates = {d for d in missing_dates if d >= since}

        print(f"  {len(game_df)} game row(s) missing spreadscore across {len(missing_dates)} date(s).")

        # Odds come from the paid odds-api, not ESPN. ESPN's pickcenter exposes
        # no run-line price, so the old path synthesised the line from a
        # `favorite` boolean and stored the H2H price in the run-line slot —
        # which silently made ml_implied_prob mean P(win) instead of P(cover).
        # The snapshot is taken at the same hour the prediction job runs, so
        # stored prices are the ones actually obtainable at decision time.
        # Nightly (--since) runs are capped hard; a deliberate full-season
        # reseed must opt in, since it costs ~2,600 credits for 2026.
        _max_dates = 5 if since else max_odds_dates
        print(f"[2/4] Fetching run lines from odds-api for {len(missing_dates)} date(s)"
              f"  (~{min(len(missing_dates), _max_dates or 10**6) * 20} credits)...")
        odds_df = dp.fetch_mlb_odds_api(
            missing_dates, request_delay=request_delay, max_dates=_max_dates
        )

        if not odds_df.empty:
            odds_map = (
                odds_df.drop_duplicates(subset=["team", "date"], keep="first")
                       .set_index(["team", "date"])
                       [["run_line", "spread_juice", "ml_odds", "snapshot_ts"]]
                       .to_dict("index")
            )
            keys = list(zip(game_df["team"], game_df["date"]))
            game_df["spread"]       = [odds_map.get(k, {}).get("run_line")     for k in keys]
            game_df["spread_juice"] = [odds_map.get(k, {}).get("spread_juice") for k in keys]
            game_df["ml_odds"]      = [odds_map.get(k, {}).get("ml_odds")      for k in keys]
            game_df["snapshot_ts"]  = [odds_map.get(k, {}).get("snapshot_ts")  for k in keys]
            game_df["odds_source"]  = [
                "odds-api-historical" if k in odds_map else None for k in keys
            ]
            # over_under is no longer fetched — nothing models totals, and
            # dropping it keeps the request at 2 markets instead of 3.
            game_df["over_under"] = None

            n_lines = game_df["spread"].notna().sum()
            print(f"  Run lines matched for {n_lines}/{len(game_df)} game rows "
                  f"({n_lines/len(game_df):.1%}).")
        else:
            print("  No odds data returned — storing games without run lines.")
            for c in ("spread", "spread_juice", "ml_odds",
                      "snapshot_ts", "odds_source", "over_under"):
                game_df[c] = None

        # Compute SpreadScore = run_diff + run_line
        mask = game_df["spread"].notna()
        game_df["spreadscore"] = None
        game_df.loc[mask, "spreadscore"] = (
            game_df.loc[mask, "diff"] + game_df.loc[mask, "spread"]
        )
        n_ss = game_df["spreadscore"].notna().sum()
        print(f"  SpreadScore computed for {n_ss}/{len(game_df)} game rows.")

    # ------------------------------------------------------------------
    # 3. Pitcher + bullpen stats (prior season) — attach to each game row
    # ------------------------------------------------------------------
    # Prior-season pitcher/bullpen stats: load from DB if already seeded,
    # only hit the statsapi (slow, no timeout) if genuinely missing.
    # These stats never change once the prior season is over, so re-fetching
    # them every nightly run is wasteful and the API call can hang.
    print(f"[3/4] Loading pitcher & bullpen stats for {season - 1} (prior season)...")
    _check_client2 = db.connect()
    try:
        _existing_all = db.fetch_games(_check_client2, "mlb", season)
    finally:
        _check_client2.close()

    # Build pitcher lookup from rows already in the DB that have sp_era populated
    pitcher_stats = pd.DataFrame()
    if not _existing_all.empty and "sp_era" in _existing_all.columns:
        _sp_rows = _existing_all.dropna(subset=["sp_era", "sp_name"])
        if not _sp_rows.empty:
            _sp_cols = [c for c in ["sp_era", "sp_whip", "sp_k9", "sp_ip_per_start"] if c in _sp_rows.columns]
            _sp_df = (
                _sp_rows[["sp_name"] + _sp_cols]
                .drop_duplicates("sp_name")
                .rename(columns={"sp_era": "era", "sp_whip": "whip", "sp_k9": "k9", "sp_ip_per_start": "ip_per_start"})
            )
            # Use normalized name as index so lookups are encoding-safe
            _sp_df["name_key"] = _sp_df["sp_name"].apply(dp._normalize_pitcher_name)
            pitcher_stats = _sp_df.drop(columns=["sp_name"]).set_index("name_key")

    # Build bullpen lookup from rows already in the DB that have bp_era populated
    bullpen_stats = pd.DataFrame()
    if not _existing_all.empty and "bp_era" in _existing_all.columns:
        _bp_rows = _existing_all.dropna(subset=["bp_era", "team"])
        if not _bp_rows.empty:
            _bp_cols = [c for c in ["bp_era", "bp_whip", "bp_k9", "bp_hr9", "bp_ip_per_game"] if c in _bp_rows.columns]
            bullpen_stats = (
                _bp_rows[["team"] + _bp_cols]
                .drop_duplicates("team")
                .set_index("team")
            )

    # Fall back to API only if DB has no stats at all
    if pitcher_stats.empty:
        print(f"  No pitcher stats in DB — fetching from statsapi (may be slow)...")
        time.sleep(0.5)
        pitcher_stats = dp.fetch_mlb_pitcher_stats(season - 1)
        time.sleep(0.5)
    else:
        print(f"  Loaded stats for {len(pitcher_stats)} starters from DB.")

    if bullpen_stats.empty:
        print(f"  No bullpen stats in DB — fetching from statsapi (may be slow)...")
        bullpen_stats = dp.fetch_mlb_bullpen_stats(season - 1)
    else:
        print(f"  Loaded bullpen stats for {len(bullpen_stats)} teams from DB.")

    if pitcher_stats.empty:
        print("  No starter stats available — sp_era/sp_whip will be NaN.")
    if bullpen_stats.empty:
        print("  No bullpen stats available — bp_era/bp_whip will be NaN.")

    # Starting pitcher lookup — normalize names on both sides to handle
    # accent/encoding mismatches (e.g. "Eury Pérez" vs "Eury Perez").
    def _lookup(name: str, col: str) -> float | None:
        if not name or pitcher_stats.empty:
            return None
        key = dp._normalize_pitcher_name(name)
        if key not in pitcher_stats.index:
            return None
        val = pitcher_stats.loc[key, col]
        return float(val) if pd.notna(val) else None

    game_df["sp_era"]          = game_df["sp_name"].apply(lambda n: _lookup(n, "era"))
    game_df["sp_whip"]         = game_df["sp_name"].apply(lambda n: _lookup(n, "whip"))
    game_df["sp_k9"]           = game_df["sp_name"].apply(lambda n: _lookup(n, "k9"))
    game_df["sp_ip_per_start"] = game_df["sp_name"].apply(lambda n: _lookup(n, "ip_per_start"))

    n_sp = game_df["sp_era"].notna().sum()
    n_has_name = (game_df["sp_name"].notna() & (game_df["sp_name"] != "")).sum()
    n_no_name  = len(game_df) - n_has_name
    print(f"  Starter ERA matched for {n_sp}/{len(game_df)} game rows "
          f"({n_sp/len(game_df):.1%})."
          f"  [no pitcher: {n_no_name}, name-no-match: {n_has_name - n_sp}]")

    # Team bullpen lookup
    def _bp_lookup(team: str, col: str) -> float | None:
        if not team or bullpen_stats.empty or team not in bullpen_stats.index:
            return None
        val = bullpen_stats.loc[team, col]
        return float(val) if pd.notna(val) else None

    game_df["bp_era"]         = game_df["team"].apply(lambda t: _bp_lookup(t, "bp_era"))
    game_df["bp_whip"]        = game_df["team"].apply(lambda t: _bp_lookup(t, "bp_whip"))
    game_df["bp_k9"]          = game_df["team"].apply(lambda t: _bp_lookup(t, "bp_k9"))
    game_df["bp_hr9"]         = game_df["team"].apply(lambda t: _bp_lookup(t, "bp_hr9"))
    game_df["bp_ip_per_game"] = game_df["team"].apply(lambda t: _bp_lookup(t, "bp_ip_per_game"))

    n_bp = game_df["bp_era"].notna().sum()
    print(f"  Bullpen ERA matched for {n_bp}/{len(game_df)} game rows "
          f"({n_bp/len(game_df):.1%}).")

    # ------------------------------------------------------------------
    # 4. Upsert to MongoDB + resolve pending prediction results
    # ------------------------------------------------------------------
    print("[4/4] Upserting to MongoDB...")
    client = db.connect()
    try:
        records = game_df.to_dict("records")
        db.upsert_games(client, records)
        print(f"  Upserted {len(records)} records for MLB {season}.")

        # Data quality report on the full season after upsert
        full_season = db.fetch_games(client, "mlb", season)
        dp.data_quality_report(full_season, label=f"MLB {season} (full season after upsert)")

        # Resolve any predictions whose games are now complete
        n_resolved = db.update_prediction_results(client, "mlb", season)
        if n_resolved:
            roi = db.fetch_roi_summary(client, "mlb", season)
            print(f"  Resolved {n_resolved} prediction result(s).")
            if roi["n_bets"] > 0:
                print(f"  Season ROI: {roi['wins']}-{roi['losses']} "
                      f"({roi['win_rate']*100:.1f}% win rate), "
                      f"P&L {roi['total_pnl']:+.2f}u, ROI {roi['roi_pct']:+.1f}%")
    finally:
        client.close()

    # Automatically seed boxscore stats for any new rows.
    # These are safe to re-run — both functions skip rows that already have data.
    seed_mlb_bp_fatigue(season, request_delay=request_delay)
    seed_mlb_sp_game_stats(season, request_delay=request_delay)


def seed_mlb_pitcher_stats(
    season: int,
    request_delay: float = 0.5,
) -> None:
    """
    Refresh prior-season starter and bullpen stats on existing game rows.

    Uses stats from season-1 to avoid leakage.  Safe to re-run — all
    writes are upserts against the existing game documents.
    """
    print(f"\nRefreshing pitcher & bullpen stats for MLB {season} "
          f"(using {season - 1} prior-season stats)...")

    pitcher_stats = dp.fetch_mlb_pitcher_stats(season - 1)
    time.sleep(request_delay)
    bullpen_stats = dp.fetch_mlb_bullpen_stats(season - 1)
    time.sleep(request_delay)

    if pitcher_stats.empty:
        print("  No starter stats returned.")
    else:
        print(f"  Loaded starter stats for {len(pitcher_stats)} pitchers.")

    if bullpen_stats.empty:
        print("  No bullpen stats returned.")
    else:
        print(f"  Loaded bullpen stats for {len(bullpen_stats)} teams.")

    if pitcher_stats.empty and bullpen_stats.empty:
        return

    client = db.connect()
    try:
        games_df = db.fetch_games(client, "mlb", season)
        if games_df.empty:
            print("  No MLB games found in DB for this season.")
            return

        updates = []
        for _, row in games_df.iterrows():
            update = {
                "sport": "mlb", "team": row["team"],
                "season": row["season"], "period": int(row["period"]),
            }
            # Starter stats — normalize name to handle accent/encoding mismatches
            sp = row.get("sp_name", "")
            sp_key = dp._normalize_pitcher_name(sp) if sp else ""
            if sp_key and not pitcher_stats.empty and sp_key in pitcher_stats.index:
                stats = pitcher_stats.loc[sp_key]
                # Include ip_per_start — was added after 2022-2025 initial seed;
                # this refresh ensures historical seasons are backfilled.
                for col in ["era", "whip", "k9", "ip_per_start"]:
                    if col in stats.index and pd.notna(stats[col]):
                        field = f"sp_{col}" if col != "ip_per_start" else "sp_ip_per_start"
                        update[field] = float(stats[col])
            # Bullpen stats (team-level)
            team = row.get("team", "")
            if team and not bullpen_stats.empty and team in bullpen_stats.index:
                bp = bullpen_stats.loc[team]
                # Include bp_ip_per_game — was added after 2022-2025 initial seed.
                for col in ["bp_era", "bp_whip", "bp_k9", "bp_hr9", "bp_ip_per_game"]:
                    if col in bp.index and pd.notna(bp[col]):
                        update[col] = float(bp[col])
            if len(update) > 4:
                updates.append(update)

        if updates:
            db.upsert_games(client, updates)
            print(f"  Updated pitcher/bullpen stats on {len(updates)} game rows.")
        else:
            print("  No matches found.")
    finally:
        client.close()


def seed_mlb_bp_fatigue(season: int, request_delay: float = 0.25) -> None:
    """
    Fetch per-game bullpen IP from statsapi boxscores and store in MongoDB.

    Run this after seed_mlb_season — it adds 'bp_ip_game' to each game
    document, which model.py uses to compute rolling 14-day bullpen fatigue.

    One statsapi call per game (~2400/season, ~10 min at default delay).
    Safe to re-run — only updates rows missing 'bp_ip_game'.
    """
    print(f"\n{'=' * 60}")
    print(f"Seeding bullpen fatigue (bp_ip_game) for MLB {season}")
    print(f"{'=' * 60}")

    client = db.connect()
    try:
        # Find games missing bp_ip_game
        col = client[db._MONGO_DB]["games"]
        pending = list(col.find(
            {"sport": "mlb", "season": season,
             "bp_ip_game": {"$exists": False},
             "game_pk": {"$exists": True, "$ne": ""}},
            {"_id": 0, "team": 1, "period": 1, "game_pk": 1},
        ))

        if not pending:
            print("  All games already have bp_ip_game.")
            return

        # Unique game_pks (each game has 2 rows — home + away)
        unique_pks = list(dict.fromkeys(r["game_pk"] for r in pending if r.get("game_pk")))
        print(f"  {len(unique_pks)} games to fetch ({len(pending)} rows)...")

        bp_data = dp.fetch_season_bp_ip(season, unique_pks, request_delay=request_delay)

        if bp_data.empty:
            print("  No bullpen IP data returned.")
            return

        # Upsert bp_ip_game back by matching (team, game_pk)
        from pymongo import UpdateOne
        ops = []
        for _, row in bp_data.iterrows():
            ops.append(UpdateOne(
                {"sport": "mlb", "season": season,
                 "team": row["team"], "game_pk": row["game_pk"]},
                {"$set": {"bp_ip_game": float(row["bp_ip_game"])}},
                upsert=False,
            ))
        if ops:
            col.bulk_write(ops)
            print(f"  Updated bp_ip_game for {len(ops)} game rows.")
    finally:
        client.close()


def seed_mlb_sp_game_stats(season: int, request_delay: float = 0.25) -> None:
    """
    Fetch per-game starting pitcher stats from statsapi boxscores.

    Stores 'sp_ip_game' (innings pitched) and 'sp_er_game' (earned runs) on
    each game document.  model.py uses these to compute rolling 5-start ERA
    for each starting pitcher, replacing the static prior-season sp_era.

    One statsapi call per unique game_pk (~2400/season, ~10 min at default delay).
    Safe to re-run — only processes rows missing sp_ip_game.
    """
    print(f"\n{'=' * 60}")
    print(f"Seeding SP game stats (sp_ip_game, sp_er_game) for MLB {season}")
    print(f"{'=' * 60}")

    from pymongo import UpdateOne

    client = db.connect()
    try:
        col = client[db._MONGO_DB]["games"]
        pending = list(col.find(
            # Keyed on sp_k_game, not sp_ip_game: rows seeded before the
            # k/bb/h/hr extraction already have sp_ip_game and would otherwise
            # be skipped forever.
            {"sport": "mlb", "season": season,
             "sp_k_game": {"$exists": False},
             "game_pk": {"$exists": True, "$ne": ""}},
            {"_id": 0, "team": 1, "period": 1, "game_pk": 1},
        ))

        if not pending:
            print("  All games already have the per-game starter line.")
            return

        unique_pks = list(dict.fromkeys(r["game_pk"] for r in pending if r.get("game_pk")))
        print(f"  {len(unique_pks)} games to fetch ({len(pending)} rows)...")

        sp_data = dp.fetch_season_sp_stats(season, unique_pks, request_delay=request_delay)

        if sp_data.empty:
            print("  No SP stats returned.")
            return

        ops = []
        _INT_FIELDS = ["sp_er_game", "sp_k_game", "sp_bb_game",
                       "sp_h_game", "sp_hr_game", "sp_pitch_game"]
        for _, row in sp_data.iterrows():
            update = {"sp_ip_game": float(row["sp_ip_game"])}
            for _f in _INT_FIELDS:
                if _f in row and pd.notna(row[_f]):
                    update[_f] = int(row[_f])
            ops.append(UpdateOne(
                {"sport": "mlb", "season": season,
                 "team": row["team"], "game_pk": row["game_pk"]},
                {"$set": update},
                upsert=False,
            ))
        if ops:
            col.bulk_write(ops)
            print(f"  Updated the per-game starter line "
                  f"(ip/er/k/bb/h/hr/pitches) for {len(ops)} game rows.")
    finally:
        client.close()


def backfill_game_pk(season: int) -> None:
    """
    Backfill missing game_pk values for a historical season.

    Historical seasons (2022-2025) were seeded before game_pk was stored,
    so their MongoDB docs have no game_pk field — which means the bullpen
    fatigue seeder can't find anything to process.

    This function:
      1. Fetches the full statsapi schedule for the season.
      2. Builds a (team, date) -> [game_pk, ...] mapping.
      3. Matches existing DB docs by (team, date), ordered by period.
      4. Writes game_pk to every matched doc.

    Safe to re-run — only updates docs that are missing or have empty game_pk.
    """
    from collections import defaultdict
    from pymongo import UpdateOne
    from config import SPORTS

    print(f"\n{'=' * 60}")
    print(f"Backfilling game_pk for MLB {season}")
    print(f"{'=' * 60}")

    client = db.connect()
    try:
        col = client[db._MONGO_DB]["games"]
        missing_count = col.count_documents({
            "sport": "mlb", "season": season,
            "$or": [{"game_pk": {"$exists": False}}, {"game_pk": ""}],
        })
    finally:
        client.close()

    if missing_count == 0:
        print("  All docs already have game_pk — skipping.")
        return
    print(f"  {missing_count} docs missing game_pk.")

    print(f"  Fetching {season} schedule from statsapi...")
    raw = dp.fetch_season_games_mlb(season)
    if not raw:
        print("  No games returned from statsapi.")
        return
    print(f"  {len(raw)} completed games fetched.")

    sport = SPORTS["mlb"]

    # Build (team, date) -> [(game_id_int, game_pk_str), ...]
    game_pk_map: dict[tuple, list] = defaultdict(list)
    for g in raw:
        home_raw = g.get("home_name", "")
        away_raw = g.get("away_name", "")
        home = dp._MLB_NAME_FIXES.get(home_raw, home_raw)
        away = dp._MLB_NAME_FIXES.get(away_raw, away_raw)
        game_date = g["game_date"][:10]
        game_pk = str(g.get("game_id", ""))
        if not game_pk:
            continue
        if sport.known_teams is not None:
            if home not in sport.known_teams or away not in sport.known_teams:
                continue
        game_id_int = int(game_pk)
        game_pk_map[(home, game_date)].append((game_id_int, game_pk))
        game_pk_map[(away, game_date)].append((game_id_int, game_pk))

    # Sort by game_id ascending so doubleheader game-1 < game-2
    for key in game_pk_map:
        game_pk_map[key].sort()

    # Load docs that are missing game_pk
    client = db.connect()
    try:
        col = client[db._MONGO_DB]["games"]
        docs = list(col.find(
            {"sport": "mlb", "season": season,
             "$or": [{"game_pk": {"$exists": False}}, {"game_pk": ""}]},
            {"_id": 1, "team": 1, "date": 1, "period": 1},
        ))

        # Group by (team, date), sort each group by period ascending
        doc_groups: dict[tuple, list] = defaultdict(list)
        for doc in docs:
            key = (doc["team"], str(doc.get("date", ""))[:10])
            doc_groups[key].append((doc.get("period", 0), doc["_id"]))
        for key in doc_groups:
            doc_groups[key].sort()

        ops = []
        unmatched = 0
        matched = 0
        for key, doc_list in doc_groups.items():
            pks = game_pk_map.get(key, [])
            for i, (_, doc_id) in enumerate(doc_list):
                if i < len(pks):
                    ops.append(UpdateOne(
                        {"_id": doc_id},
                        {"$set": {"game_pk": pks[i][1]}},
                    ))
                    matched += 1
                else:
                    unmatched += 1

        if ops:
            result = col.bulk_write(ops)
            print(f"  Written game_pk to {result.modified_count} docs "
                  f"({matched} matched, {unmatched} unmatched).")
        else:
            print("  No updates to write.")
        if unmatched:
            print(f"  WARNING: {unmatched} docs had no matching statsapi game.")
    finally:
        client.close()


def seed_mlb_odds_sbr_web(
    season: int,
    request_delay: float = 0.5,
) -> None:
    """
    Backfill run-line odds for a season by scraping sportsbookreview.com.

    For each date in the season that is missing spreadscore, fetches closing
    run-line odds from SBR, computes spreadscore = diff + run_line, and
    upserts only the odds fields (spread, moneyline, spreadscore).

    Never overwrites existing non-null values — safe to re-run.
    """
    print(f"\n{'=' * 60}")
    print(f"Seeding SBR web odds for MLB {season}")
    print(f"{'=' * 60}")

    client = db.connect()
    try:
        games_df = db.fetch_games(client, "mlb", season)
    finally:
        client.close()

    if games_df.empty:
        print("  No games in DB for this season — run main seed first.")
        return

    # Find specific (team, period) rows missing spreadscore
    if "spreadscore" in games_df.columns:
        missing_mask = games_df["spreadscore"].isna()
    else:
        missing_mask = pd.Series(True, index=games_df.index)

    missing_games = games_df[missing_mask].copy()
    if missing_games.empty:
        print("  All games already have spreadscore — nothing to do.")
        return

    missing_dates = sorted(missing_games["date"].dropna().astype(str).unique())
    missing_keys = set(zip(missing_games["team"], missing_games["period"].astype(int)))
    print(f"  {len(missing_games)} game(s) missing spreadscore across {len(missing_dates)} date(s).")

    # Restrict game_lookup to only the missing games
    games_df["date"] = games_df["date"].astype(str)
    games_df = missing_games.copy()

    # Build a lookup: (team, date) -> list of game dicts sorted by period
    # Lists handle doubleheaders — two games same team same date.
    games_df["date"] = games_df["date"].astype(str)
    game_lookup: dict[tuple, list[dict]] = {}
    for _, row in games_df.iterrows():
        key = (row["team"], str(row["date"]))
        game_lookup.setdefault(key, []).append({
            "diff":   int(row["diff"]),
            "period": int(row["period"]),
            "season": int(row["season"]),
            "score":  int(row["score"]),
        })
    # Sort each list by period so game 1 of a DH comes first
    for lst in game_lookup.values():
        lst.sort(key=lambda x: x["period"])

    updates = []
    for i, date in enumerate(missing_dates):
        odds_df = dp.fetch_mlb_odds_sbr_web(date)
        if odds_df.empty:
            print(f"  [{i+1}/{len(missing_dates)}] {date}: no odds returned")
            time.sleep(request_delay)
            continue

        # Track how many SBR rows we've consumed per (team, date) for DH matching
        consumed: dict[tuple, int] = {}
        matched = 0
        for _, row in odds_df.iterrows():
            key = (row["team"], date)
            games = game_lookup.get(key)
            if not games:
                continue
            idx = consumed.get(key, 0)
            if idx >= len(games):
                continue  # more SBR rows than DB rows — skip
            game = games[idx]
            consumed[key] = idx + 1

            run_line = row["run_line"]
            ss = round(float(game["diff"]) + float(run_line), 4)
            rec = {
                "sport":       "mlb",
                "team":        row["team"],
                "season":      game["season"],
                "period":      game["period"],
                "spread":      float(run_line),
                "moneyline":   float(row["moneyline"]),
                "spreadscore": ss,
            }
            if pd.notna(row.get("open_spread")):
                rec["open_spread"]    = float(row["open_spread"])
            if pd.notna(row.get("open_moneyline")):
                rec["open_moneyline"] = float(row["open_moneyline"])
            updates.append(rec)
            matched += 1

        print(f"  [{i+1}/{len(missing_dates)}] {date}: {matched}/{len(odds_df)} rows matched")
        time.sleep(request_delay)

    if not updates:
        print("  No updates to write.")
        return

    client = db.connect()
    try:
        db.upsert_games(client, updates)
        print(f"\n  Upserted odds for {len(updates)} game rows across {len(missing_dates)} dates.")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed MLB historical game data (scores + ESPN run lines + pitcher stats)."
    )
    parser.add_argument(
        "--seasons", type=int, nargs="+", required=True,
        help="Calendar years to seed. Start from 2022 (post-COVID). "
             "E.g.: --seasons 2022 2023 2024 2025"
    )
    parser.add_argument(
        "--delay", type=float, default=0.25,
        help="Seconds between API calls per game (default 0.25)."
    )
    parser.add_argument(
        "--skip-odds", action="store_true", dest="skip_odds",
        help="Skip ESPN odds fetch — seed scores and pitcher stats only."
    )
    parser.add_argument(
        "--pitcher-refresh", action="store_true", dest="pitcher_refresh",
        help="Only refresh pitcher stats on existing rows, skip game/odds seeding."
    )
    parser.add_argument(
        "--bp-fatigue", action="store_true", dest="bp_fatigue",
        help="Fetch per-game bullpen IP from statsapi boxscores (adds bp_ip_game). "
             "Run after the main seed. ~10 min per season at default delay."
    )
    parser.add_argument(
        "--sp-game-stats", action="store_true", dest="sp_game_stats",
        help="Fetch per-game starting pitcher IP and ER from statsapi boxscores "
             "(adds sp_ip_game, sp_er_game). Used to compute rolling 5-start ERA. "
             "Run after the main seed. ~10 min per season at default delay."
    )
    parser.add_argument(
        "--sbr-web", action="store_true", dest="sbr_web",
        help="Backfill missing run-line odds by scraping sportsbookreview.com. "
             "One request per date, ~0.5s delay. Safe to re-run."
    )
    parser.add_argument(
        "--backfill-game-pk", action="store_true", dest="backfill_game_pk",
        help="Backfill missing game_pk values for historical seasons (2022-2025). "
             "Required before --bp-fatigue can run on those seasons."
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only fetch games on or after this date YYYY-MM-DD. "
             "Use for fast nightly incremental updates instead of re-pulling the full season."
    )
    parser.add_argument(
        "--max-odds-dates", type=int, default=5, dest="max_odds_dates",
        help="Cap on dates fetched from the paid odds-api in one run "
             "(20 credits each). Nightly --since runs are always capped at 5. "
             "Raise this only for a deliberate backfill; a full 2026 season is "
             "~130 dates / 2,600 credits."
    )
    args = parser.parse_args()

    for season in sorted(args.seasons):
        if args.backfill_game_pk:
            backfill_game_pk(season)
        elif args.pitcher_refresh:
            seed_mlb_pitcher_stats(season, request_delay=args.delay)
        elif args.bp_fatigue:
            seed_mlb_bp_fatigue(season, request_delay=args.delay)
        elif args.sp_game_stats:
            seed_mlb_sp_game_stats(season, request_delay=args.delay)
        elif args.sbr_web:
            seed_mlb_odds_sbr_web(season, request_delay=args.delay)
        else:
            seed_mlb_season(season, skip_odds=args.skip_odds, request_delay=args.delay,
                            since=args.since, max_odds_dates=args.max_odds_dates)
        time.sleep(1.0)
