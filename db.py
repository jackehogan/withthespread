"""
MongoDB helpers.

Collections
-----------
games
    One document per (sport, team, season, period).
    Stores raw game data (scores, spread) and the computed spreadscore.
    Upserted on every pipeline run — safe to re-run without duplicates.

predictions
    One document per (sport, team, season, period).
    Stores model outputs for an upcoming period.
    Also upserted — re-running overwrites stale predictions cleanly.

Indexes (create once via create_indexes())
------------------------------------------
    games:       unique on (sport, team, season, period)
    predictions: unique on (sport, team, season, period)
"""

import json

import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection

_MONGO_DB = "withTheSpread"
_KEY = ("sport", "team", "season", "period")


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
    for col_name in ("games", "predictions"):
        client[_MONGO_DB][col_name].create_index(
            [(k, ASCENDING) for k in _KEY],
            unique=True,
            name="sport_team_season_period",
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

def _upsert(collection: Collection, records: list[dict]) -> None:
    if not records:
        return
    ops = [
        UpdateOne({k: r[k] for k in _KEY}, {"$set": r}, upsert=True)
        for r in records
    ]
    collection.bulk_write(ops)


def upsert_games(client: MongoClient, records: list[dict]) -> None:
    """Upsert game records. Each record must contain all four key fields."""
    _upsert(client[_MONGO_DB]["games"], records)


def upsert_predictions(client: MongoClient, records: list[dict]) -> None:
    """Upsert prediction records. Each record must contain all four key fields."""
    _upsert(client[_MONGO_DB]["predictions"], records)
