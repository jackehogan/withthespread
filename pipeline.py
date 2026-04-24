"""
Weekly prediction pipeline and historical data seeding.

Public interface
----------------
run(sport, season, next_period, lookback, client, key_type, max_evals)
    Full weekly pipeline: fetch results → store spreads → train → predict.

seed_season(client, sport, season, request_delay)
    One-time historical backfill for a single season.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from pymongo import MongoClient

import db
import data_pipeline as dp
import model as ml
from config import SportConfig


def _fetch_games_filtered(client, sport: SportConfig, season: int | None = None) -> "pd.DataFrame":
    """Fetch games from DB and apply regular season date filter if configured."""
    df = db.fetch_games(client, sport.name, season)
    if season is not None:
        return dp.filter_regular_season(df, sport, season)
    # No single season — filter each season individually then recombine
    if df.empty or "season" not in df.columns:
        return df
    import pandas as pd
    seasons = df["season"].unique()
    return pd.concat(
        [dp.filter_regular_season(df[df["season"] == s], sport, s) for s in seasons],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Weekly pipeline
# ---------------------------------------------------------------------------

def run(
    sport: SportConfig,
    season: int,
    next_period: int,
    client: MongoClient,
    key_type: str = "free",
    max_evals: int = 10,
) -> pd.DataFrame:
    """
    Execute the full weekly pipeline for one sport.

    Steps
    -----
    1. Fetch season results from api-sports.io and upsert with SpreadScores.
    2. Fetch upcoming spreads and store skeleton rows for the next period.
    3. Build feature matrices from all historical data, tune and train models.
    4. Generate predictions for the upcoming period and upsert.

    Parameters
    ----------
    sport       : SportConfig instance (e.g. NFL or NBA)
    season      : Season start year (e.g. 2024)
    next_period : Period number to predict (NFL week or NBA game number)
    client      : Open MongoClient
    key_type    : 'free' or 'paid' for the-odds-api.com
    max_evals   : Hyperopt evaluation budget per model (lookback also tuned)

    Returns
    -------
    DataFrame of predictions for the upcoming period.
    """
    # Step 1: fetch and store game results
    print(f"[1/4] Fetching {sport.name.upper()} {season} game results...")
    raw_games = dp.fetch_season_games(sport, season)
    fresh = dp.parse_game_results(raw_games, sport, season)

    if fresh.empty:
        print("  No completed games found.")
        return pd.DataFrame()

    # Merge in spread values already stored in the DB so we can compute SpreadScore
    stored = _fetch_games_filtered(client, sport, season)
    if not stored.empty:
        spread_lookup = (
            stored[["team", "period", "spread"]].dropna(subset=["spread"])
            .set_index(["team", "period"])["spread"]
        )
        fresh["spread"] = fresh.set_index(["team", "period"]).index.map(spread_lookup).values
    else:
        fresh["spread"] = np.nan

    fresh["spreadscore"] = np.where(
        fresh["spread"].notna(), fresh["diff"] + fresh["spread"], np.nan
    )
    db.upsert_games(client, fresh.to_dict("records"))
    print(f"  Upserted {len(fresh)} game rows.")

    # Step 1b: fetch and attach current-season ratings to next_period rows (NBA only)
    ratings_df = pd.DataFrame()
    if sport.name == "nba":
        last_played = str(fresh["date"].max())[:10] if "date" in fresh.columns and not fresh["date"].isna().all() else None
        ratings_df = dp.fetch_nba_ratings(season, last_played)
        if not ratings_df.empty:
            n_rated = db.upsert_game_ratings(client, sport.name, season, next_period, ratings_df)
            print(f"  Ratings upserted for {n_rated} teams (as of {last_played}).")

    # Step 2: fetch and store upcoming spreads
    print(f"[2/4] Fetching upcoming spreads (key_type={key_type})...")
    dates = dp.get_upcoming_dates(raw_games, sport, next_period)
    spreads_df = dp.fetch_upcoming_spreads(sport, dates=dates, key_type=key_type)

    if spreads_df.empty:
        print("  No upcoming spreads found.")
    else:
        _odds_fields = ["spread", "spread_juice", "total", "implied_prob"]
        db.upsert_games(client, [
            {
                "sport": sport.name, "team": team, "opponent": row["opponent"],
                "season": season, "period": next_period,
                **{f: float(row[f]) for f in _odds_fields if f in row and pd.notna(row[f])},
            }
            for team, row in spreads_df.iterrows()
        ])
        print(f"  Stored spreads for {len(spreads_df)} teams.")

    # Step 3: build features and train
    print("[3/4] Building features and training models...")
    all_games = _fetch_games_filtered(client, sport)
    reg, sigma_diff, scores, best_lookback, best_k, style_model = ml.train_models(
        all_games, next_period, sport.eval_season, sport.eval_split_period, max_evals
    )
    print(f"  Scores: {scores}")

    # Step 4: generate and store predictions
    print("[4/4] Generating predictions...")
    season_games = _fetch_games_filtered(client, sport, season)

    # Build upcoming context for the next period
    upcoming_context = None
    if not spreads_df.empty and "game_date" in spreads_df.columns:
        last_dates = season_games.groupby("team")["date"].max()
        ctx_cols = ["home", "game_date", "spread", "spread_juice", "total", "implied_prob"]
        uc = spreads_df[[c for c in ctx_cols if c in spreads_df.columns]].copy()
        rest = (
            pd.to_datetime(uc["game_date"]) -
            pd.to_datetime(uc.index.map(last_dates))
        ).dt.days
        uc["is_b2b"] = (rest == 1).astype(float)
        uc = uc.drop(columns=["game_date"])

        # Attach current ratings + opponent matchup features to upcoming context
        if sport.name == "nba" and not ratings_df.empty:
            for col in ["off_rating", "def_rating", "net_rating"]:
                if col in ratings_df.columns:
                    uc[col] = uc.index.map(ratings_df[col])

            # Opponent matchup: look up opponent's ratings from ratings_df
            if "opponent" in uc.columns:
                uc["opp_off_rating"] = uc["opponent"].map(ratings_df.get("off_rating", pd.Series(dtype=float)))
                uc["opp_def_rating"] = uc["opponent"].map(ratings_df.get("def_rating", pd.Series(dtype=float)))
                if "off_rating" in uc.columns and "opp_def_rating" in uc.columns:
                    uc["matchup_off_edge"] = uc["off_rating"] - uc["opp_def_rating"]
                    uc["matchup_def_edge"] = uc["def_rating"] - uc["opp_off_rating"]

        upcoming_context = uc

    X_pred = ml.build_prediction_features(
        season_games, next_period, best_lookback, season, upcoming_context, style_model, best_k
    )
    if X_pred.empty:
        print("  Not enough season data to predict.")
        return pd.DataFrame()

    opponent_map = (
        upcoming_context["opponent"].to_dict()
        if upcoming_context is not None and "opponent" in upcoming_context.columns
        else None
    )
    preds = ml.predict(reg, sigma_diff, X_pred, opponent_map)

    if not spreads_df.empty:
        pred_df = preds.join(spreads_df[["opponent", "spread"]], how="left")
    else:
        pred_df = preds.assign(opponent=np.nan, spread=np.nan)

    def _rel_diff(row: pd.Series, col: str) -> float | None:
        opp = row.get("opponent")
        if pd.isna(opp) or opp not in preds.index:
            return None
        return float(row[col]) - float(preds.loc[opp, col])

    pred_df["predspread_diff"] = pred_df.apply(_rel_diff, col="predspread", axis=1)
    pred_df["coverprob_diff"] = pred_df.apply(_rel_diff, col="coverprob", axis=1)

    db.upsert_predictions(client, [
        {
            "sport": sport.name, "team": team, "season": season, "period": next_period,
            "opponent": row.get("opponent") if pd.notna(row.get("opponent")) else None,
            "spread": float(row["spread"]) if pd.notna(row.get("spread")) else None,
            "predspread": float(row["predspread"]),
            "coverprob": float(row["coverprob"]),
            "predspread_diff": row["predspread_diff"],
            "coverprob_diff": row["coverprob_diff"],
        }
        for team, row in pred_df.iterrows()
    ])
    print(f"  Stored {len(pred_df)} predictions.")
    return pred_df


# ---------------------------------------------------------------------------
# Historical data seeding
# ---------------------------------------------------------------------------

def seed_season(
    client: MongoClient,
    sport: SportConfig,
    season: int,
    request_delay: float = 1.0,
) -> None:
    """
    Backfill one season of historical game data into MongoDB.

    For each period:
      1. Parses completed game results from api-sports.io (free API).
      2. Fetches spread lines from the paid historical Odds API.
      3. Computes SpreadScore = point_diff + spread.
      4. Upserts everything into the games collection.

    Re-running is safe — all writes are upserts.

    Parameters
    ----------
    client        : Open MongoClient
    sport         : SportConfig instance
    season        : Season start year
    request_delay : Seconds to wait between paid API calls (rate-limit safety)
    """
    print(f"\n{'=' * 60}")
    print(f"Seeding {sport.name.upper()} season {season}")
    print(f"{'=' * 60}")

    raw_games = dp.fetch_season_games(sport, season)
    game_df = dp.parse_game_results(raw_games, sport, season)

    if game_df.empty:
        print("  No completed games found. Skipping.")
        return

    periods = sorted(game_df["period"].dropna().unique())
    period_dates = _group_by_period(game_df)
    print(f"  {len(game_df)} game rows across {len(periods)} periods.")

    all_records: list[dict] = []
    for period in periods:
        dates = period_dates.get(int(period), [])
        period_games = game_df[game_df["period"] == period].copy()
        period_games["spread"] = None
        period_games["spreadscore"] = None

        if dates:
            try:
                spread_df = dp.fetch_historical_spreads(sport, dates)
                time.sleep(request_delay)
                if not spread_df.empty:
                    period_games["spread"] = period_games["team"].map(spread_df["spread"])
                    period_games["spreadscore"] = period_games["diff"] + period_games["spread"]
            except Exception as exc:
                print(f"  Period {int(period)}: spread fetch failed ({exc}). Storing without spread.")

        all_records.extend(period_games.to_dict("records"))
        covered = period_games["spreadscore"].notna().sum()
        print(f"  Period {int(period):3d}: {len(period_games):2d} teams, {covered:2d} with SpreadScore.")

    db.upsert_games(client, all_records)
    print(f"  Done. Upserted {len(all_records)} records.")


def _group_by_period(game_df: pd.DataFrame) -> dict[int, list[str]]:
    """Return period → sorted unique game dates (YYYY-MM-DD)."""
    groups: dict[int, list[str]] = {}
    for _, row in game_df[["period", "date"]].dropna().iterrows():
        groups.setdefault(int(row["period"]), []).append(row["date"])
    return {k: sorted(set(v)) for k, v in groups.items()}
