"""
MongoDB helpers.

Collections
-----------
games
    One document per (sport, team, season, period).
    Stores raw game data (scores, spread) and the computed spreadscore.
    Upserted on every pipeline run — safe to re-run without duplicates.

predictions
    One document per (sport, team, season, prediction_date).
    Stores model outputs for a specific game date. prediction_date is included
    in the key so each nightly run creates independent records — teams with the
    same next_period across consecutive days do not overwrite each other.
    `period` is stored as a data field (used to look up game results).

Indexes (create once via create_indexes())
------------------------------------------
    games:       unique on (sport, team, season, period)
    predictions: unique on (sport, team, season, prediction_date)
"""

import json

import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection

_MONGO_DB   = "withTheSpread"
_KEY        = ("sport", "team", "season", "period")           # games key
_PRED_KEY   = ("sport", "team", "season", "prediction_date")  # predictions key


def _read_config(path: str = "data/config.txt") -> dict:
    with open(path) as f:
        return json.load(f)


def connect(config_path: str = "data/config.txt") -> MongoClient:
    """Open and verify a MongoDB connection. Caller is responsible for closing."""
    cfg = _read_config(config_path)
    uri = (
        f"mongodb+srv://{cfg['mongo']['username']}:{cfg['mongo']['pw']}"
        "@cluster0.ml8jvfc.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp"
    )
    client = MongoClient(uri)
    client.admin.command("ping")
    return client


def create_indexes(client: MongoClient) -> None:
    """Create unique compound indexes on both collections. Idempotent."""
    client[_MONGO_DB]["games"].create_index(
        [(k, ASCENDING) for k in _KEY],
        unique=True,
        name="sport_team_season_period",
    )
    # Predictions are keyed by prediction_date (not period) so each nightly
    # run produces independent records even when next_period is the same
    # across consecutive days.
    client[_MONGO_DB]["predictions"].create_index(
        [(k, ASCENDING) for k in _PRED_KEY],
        unique=True,
        name="sport_team_season_prediction_date",
    )
    print("Indexes created (or already exist).")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def fetch_games(
    client: MongoClient,
    sport: str,
    season: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of game records for the given sport (and season)."""
    query: dict = {"sport": sport}
    if season is not None:
        query["season"] = season
    return pd.DataFrame(list(client[_MONGO_DB]["games"].find(query, {"_id": 0})))


def fetch_predictions(
    client: MongoClient,
    sport: str,
    season: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of prediction records for the given sport (and season)."""
    query: dict = {"sport": sport}
    if season is not None:
        query["season"] = season
    return pd.DataFrame(list(client[_MONGO_DB]["predictions"].find(query, {"_id": 0})))


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _upsert(collection: Collection, records: list[dict], key_fields: tuple) -> None:
    if not records:
        return
    ops = []
    for r in records:
        key = {k: r[k] for k in key_fields}
        # Only $set fields that have a real value — never overwrite existing data with None.
        payload = {k: v for k, v in r.items() if v is not None and v == v}  # v==v excludes NaN
        ops.append(UpdateOne(key, {"$set": payload}, upsert=True))
    collection.bulk_write(ops)


def upsert_games(client: MongoClient, records: list[dict]) -> None:
    """Upsert game records. Keyed on (sport, team, season, period)."""
    _upsert(client[_MONGO_DB]["games"], records, _KEY)


def upsert_predictions(client: MongoClient, records: list[dict]) -> None:
    """Upsert prediction records. Keyed on (sport, team, season, prediction_date).

    Each nightly run stores independent records — teams with the same next_period
    on consecutive days will no longer overwrite each other's predictions.
    All records must include a 'prediction_date' field (YYYY-MM-DD string).
    """
    _upsert(client[_MONGO_DB]["predictions"], records, _PRED_KEY)


def update_prediction_results(
    client: MongoClient,
    sport: str,
    season: int,
) -> int:
    """
    Match completed game results to pending predictions and compute P&L.

    For each prediction without a 'covered' field, looks up the corresponding
    game document. If a spreadscore exists (game is complete), computes:

        covered   : True if spreadscore > 0, False if < 0, None if push (= 0)
        pnl       : profit/loss for a 1-unit flat bet, using stored run-line
                    juice (moneyline field). Falls back to -110 if missing.
                      win  : 100 / abs(juice) if juice < 0
                             juice / 100      if juice > 0
                      lose : -1.0
                      push : 0.0
        result_spreadscore : the actual spreadscore from the game document

    The game is located by DATE, not by period. A prediction's `period` is the
    model's forward-looking counter and drifts from the period actually stored
    on that team's game row (measured offsets of -4..+3 over Aug 2026), so a
    period-keyed lookup silently graded ~a third of bets against a different
    game of the same team's season. `opponent` breaks the tie on doubleheaders.

    Safe to call repeatedly — only updates documents still missing 'covered'.

    Returns the number of predictions updated.
    """
    pred_col = client[_MONGO_DB]["predictions"]
    game_col = client[_MONGO_DB]["games"]

    # Only resolve predictions where EV was positive at prediction time.
    # Fetch prediction_date so we can use it as the update key.
    pending = list(pred_col.find(
        {"sport": sport, "season": season, "covered": {"$exists": False}, "ev": {"$gt": 0}},
        {"_id": 0, "team": 1, "period": 1, "prediction_date": 1, "opponent": 1,
         "moneyline": 1, "ml_odds": 1, "spread": 1, "ev": 1, "bet": 1},
    ))
    if not pending:
        return 0

    ops = []
    for pred in pending:
        # Date is the only key both collections agree on. Fall back to period
        # only for legacy predictions that predate prediction_date.
        pred_date = pred.get("prediction_date")
        _proj = {"spreadscore": 1, "diff": 1, "period": 1, "_id": 0}
        game = None
        if pred_date:
            q = {"sport": sport, "team": pred["team"],
                 "season": season, "date": pred_date}
            if pred.get("opponent"):
                # Doubleheader tiebreak; retry without it if the opponent
                # string does not match between collections.
                game = game_col.find_one({**q, "opponent": pred["opponent"]}, _proj)
            if game is None:
                game = game_col.find_one(q, _proj)
        else:
            game = game_col.find_one(
                {"sport": sport, "team": pred["team"],
                 "season": season, "period": pred["period"]},
                _proj,
            )
        if not game or game.get("spreadscore") is None:
            continue  # game not yet complete, or no game that day

        ss = float(game["spreadscore"])

        # Covered / push / loss on the run line
        if ss > 0:
            covered = True
        elif ss < 0:
            covered = False
        else:
            covered = None  # push

        # Grade the outcome that was actually BET, not always the run line.
        # A team winning 1-0 wins the moneyline but fails to cover -1.5, so
        # grading an ML bet on `covered` records a win as a loss and corrupts
        # the per-market ROI used to judge which market carries edge.
        bet = pred.get("bet", "SPREAD")
        if bet == "ML":
            raw_diff = game.get("diff")
            if raw_diff is None:
                continue  # cannot grade a moneyline bet without the run diff
            d = float(raw_diff)
            won = True if d > 0 else (False if d < 0 else None)
        else:
            won = covered

        # P&L for 1-unit flat bet — odds for whichever market was bet
        raw_juice = pred.get("ml_odds") if bet == "ML" else pred.get("moneyline")
        try:
            juice = float(raw_juice) if raw_juice is not None else -110.0
        except (TypeError, ValueError):
            juice = -110.0

        if won is None:
            pnl = 0.0
        elif won:
            pnl = (100.0 / abs(juice)) if juice < 0 else (juice / 100.0)
        else:
            pnl = -1.0

        # Match on (sport, team, season, prediction_date) — the prediction key.
        # Fall back to period-based match for legacy records without prediction_date.
        if pred_date:
            match_filter = {"sport": sport, "team": pred["team"],
                            "season": season, "prediction_date": pred_date}
        else:
            match_filter = {"sport": sport, "team": pred["team"],
                            "season": season, "period": pred["period"]}

        ops.append(UpdateOne(
            match_filter,
            {"$set": {
                # `covered` always records the run-line outcome so historical
                # analysis stays comparable; `bet_won` records whether the bet
                # actually placed won, which is what pnl is derived from.
                "covered":            covered,
                "bet_won":            won,
                "pnl":                round(pnl, 4),
                "result_spreadscore": ss,
                # period of the game actually graded, for auditing the join
                "result_period":      game.get("period"),
            }},
            upsert=False,
        ))

    if not ops:
        return 0
    result = pred_col.bulk_write(ops)
    return result.modified_count


def fetch_roi_summary(
    client: MongoClient,
    sport: str,
    season: int | None = None,
) -> dict:
    """
    Return a P&L summary for all resolved predictions.

    Keys: n_bets, wins, losses, pushes, win_rate, total_pnl, roi_pct
    """
    query: dict = {"sport": sport, "covered": {"$exists": True}, "ev": {"$gt": 0}}
    if season is not None:
        query["season"] = season

    docs = list(client[_MONGO_DB]["predictions"].find(query, {"covered": 1, "pnl": 1, "_id": 0}))
    if not docs:
        return {"n_bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                "win_rate": None, "total_pnl": 0.0, "roi_pct": None}

    wins   = sum(1 for d in docs if d.get("covered") is True)
    losses = sum(1 for d in docs if d.get("covered") is False)
    pushes = sum(1 for d in docs if d.get("covered") is None)
    pnl    = sum(d.get("pnl", 0.0) or 0.0 for d in docs)
    n_bets = wins + losses + pushes
    decisive = wins + losses  # exclude pushes from rate

    return {
        "n_bets":    n_bets,
        "wins":      wins,
        "losses":    losses,
        "pushes":    pushes,
        "win_rate":  round(wins / decisive, 4) if decisive else None,
        "total_pnl": round(pnl, 4),
        "roi_pct":   round(100 * pnl / n_bets, 2) if n_bets else None,
    }


def upsert_game_ratings(
    client: MongoClient,
    sport: str,
    season: int,
    period: int,
    ratings_df: "pd.DataFrame",
) -> int:
    """
    Merge off_rating / def_rating / net_rating into existing game documents
    for a specific (sport, season, period).

    ratings_df must be indexed by team name with columns:
        off_rating, def_rating, net_rating

    Only updates documents that already exist (does not create new ones).
    Returns the number of documents modified.
    """
    if ratings_df.empty:
        return 0

    col = client[_MONGO_DB]["games"]
    ops = []
    rating_cols = [c for c in ["off_rating", "def_rating", "net_rating"] if c in ratings_df.columns]
    for team, row in ratings_df.iterrows():
        update = {c: float(row[c]) for c in rating_cols if pd.notna(row[c])}
        if not update:
            continue
        ops.append(UpdateOne(
            {"sport": sport, "team": team, "season": season, "period": period},
            {"$set": update},
            upsert=False,   # only update existing game rows
        ))

    if not ops:
        return 0
    result = col.bulk_write(ops)
    return result.modified_count
