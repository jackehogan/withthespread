"""
Sport-agnostic betting spread prediction pipeline.

Supports NFL and NBA out of the box. Adding a new sport requires only a new
SportConfig entry at the top of this file.

Usage
-----
    # One-time: create MongoDB indexes
    python main.py setup

    # One-time: backfill historical data (paid Odds API key required for spreads)
    python main.py seed --sport nba --seasons 2022 2023 2024

    # Weekly: run the prediction pipeline
    python main.py run --sport nba --season 2025 --period 55 --lookback 10
    python main.py run --sport nfl --season 2024 --period 12 --lookback 5
"""

from __future__ import annotations

import argparse
import datetime
import http.client
import json
import time

import numpy as np
import pandas as pd
import requests
from hyperopt import fmin, hp, tpe
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from dataclasses import dataclass


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class SportConfig:
    """All sport-specific constants in one place. Add a new sport by adding a
    new instance below — no other code needs to change."""

    # Short identifier stored as the 'sport' field in MongoDB.
    name: str
    # api-sports.io hostname.
    api_sports_host: str
    # League ID on api-sports.io (1 = NFL, 12 = NBA).
    api_sports_league: int
    # Season string format: "{year}" for NFL, "{year}-{year+1}" for NBA.
    api_sports_season_fmt: str
    # Sport key for the-odds-api.com.
    odds_api_sport: str
    # Periods in a full regular season (weeks for NFL, games for NBA).
    season_periods: int
    # Season held out as validation set during model training.
    validation_season: int

    def format_season(self, year: int) -> str:
        """Return the season string expected by api-sports.io."""
        if "{year+1}" in self.api_sports_season_fmt:
            return self.api_sports_season_fmt.format(year=year, **{"year+1": year + 1})
        return self.api_sports_season_fmt.format(year=year)


NFL = SportConfig(
    name="nfl",
    api_sports_host="v1.american-football.api-sports.io",
    api_sports_league=1,
    api_sports_season_fmt="{year}",
    odds_api_sport="americanfootball_nfl",
    season_periods=18,
    validation_season=2022,
)

NBA = SportConfig(
    name="nba",
    api_sports_host="v1.basketball.api-sports.io",
    api_sports_league=12,
    api_sports_season_fmt="{year}-{year+1}",
    odds_api_sport="basketball_nba",
    season_periods=82,
    validation_season=2023,
)

SPORTS: dict[str, SportConfig] = {"nfl": NFL, "nba": NBA}


# =============================================================================
# DATABASE
# =============================================================================
#
# Two collections, both keyed on (sport, team, season, period):
#
#   games       — one row per team per game; stores scores, spread, spreadscore.
#   predictions — one row per team per upcoming period; stores model outputs.
#
# All writes are upserts — safe to re-run without creating duplicates.

_MONGO_DB = "withTheSpread"
_DB_KEY = ("sport", "team", "season", "period")
_CONFIG_PATH = "data/config.txt"


def _read_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def db_connect() -> MongoClient:
    """Open and ping a MongoDB connection. Caller must close."""
    cfg = _read_config()
    uri = (
        f"mongodb+srv://{cfg['mongo']['username']}:{cfg['mongo']['pw']}"
        "@cluster0.ml8jvfc.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp"
    )
    client = MongoClient(uri)
    client.admin.command("ping")
    return client


def db_create_indexes(client: MongoClient) -> None:
    """Create unique compound indexes on both collections. Idempotent."""
    for col_name in ("games", "predictions"):
        client[_MONGO_DB][col_name].create_index(
            [(k, ASCENDING) for k in _DB_KEY],
            unique=True,
            name="sport_team_season_period",
        )
    print("Indexes created (or already exist).")


def _upsert(collection: Collection, records: list[dict]) -> None:
    if not records:
        return
    ops = [
        UpdateOne({k: r[k] for k in _DB_KEY}, {"$set": r}, upsert=True)
        for r in records
    ]
    collection.bulk_write(ops)


def db_fetch_games(
    client: MongoClient, sport: str, season: int | None = None
) -> pd.DataFrame:
    query: dict = {"sport": sport}
    if season is not None:
        query["season"] = season
    return pd.DataFrame(
        list(client[_MONGO_DB]["games"].find(query, {"_id": 0}))
    )


def db_fetch_predictions(
    client: MongoClient, sport: str, season: int | None = None
) -> pd.DataFrame:
    query: dict = {"sport": sport}
    if season is not None:
        query["season"] = season
    return pd.DataFrame(
        list(client[_MONGO_DB]["predictions"].find(query, {"_id": 0}))
    )


def db_upsert_games(client: MongoClient, records: list[dict]) -> None:
    _upsert(client[_MONGO_DB]["games"], records)


def db_upsert_predictions(client: MongoClient, records: list[dict]) -> None:
    _upsert(client[_MONGO_DB]["predictions"], records)


# =============================================================================
# DATA PIPELINE
# =============================================================================
#
# Two external APIs:
#   api-sports.io     — game results (scores, dates)
#   the-odds-api.com  — betting spreads (live and historical)

_ODDS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _api_sports_get(host: str, path: str, api_key: str) -> dict:
    conn = http.client.HTTPSConnection(host)
    conn.request("GET", path, headers={
        "x-rapidapi-host": host,
        "x-rapidapi-key": api_key,
    })
    return json.loads(conn.getresponse().read().decode("utf-8"))


def fetch_season_games(sport: SportConfig, season: int) -> list[dict]:
    """Fetch all games for a season from api-sports.io. Returns raw API list."""
    cfg = _read_config()
    return _api_sports_get(
        host=sport.api_sports_host,
        path=f"/games?league={sport.api_sports_league}&season={sport.format_season(season)}",
        api_key=cfg["results"]["key"],
    )["response"]


def parse_game_results(
    raw_games: list[dict], sport: SportConfig, season: int
) -> pd.DataFrame:
    """
    Parse raw api-sports.io games into a tidy long-format DataFrame.
    One row per team per completed game with columns:
        sport, team, opponent, season, period, date, score, opp_score, diff

    period
        NFL : week number (1–18)
        NBA : sequential game number per team (1–82), assigned by date order
    """
    records = []
    for game in raw_games:
        if game["game"].get("stage") != "Regular Season":
            continue
        if game["game"]["status"]["short"] == "NS":
            continue
        home_score = game["scores"]["home"]["total"]
        away_score = game["scores"]["away"]["total"]
        if home_score is None or away_score is None:
            continue

        home_team = game["teams"]["home"]["name"]
        away_team = game["teams"]["away"]["name"]
        game_date = game["game"]["date"]["date"]
        period = _extract_period(game, sport)

        for team, opp, score, opp_score in (
            (home_team, away_team, int(home_score), int(away_score)),
            (away_team, home_team, int(away_score), int(home_score)),
        ):
            records.append({
                "sport": sport.name, "team": team, "opponent": opp,
                "season": season, "period": period, "date": game_date,
                "score": score, "opp_score": opp_score, "diff": score - opp_score,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if sport.name == "nba":
        # NBA has no week numbers — assign sequential game number per team by date
        df = df.sort_values("date")
        df["period"] = df.groupby("team").cumcount() + 1
    return df.reset_index(drop=True)


def _extract_period(game: dict, sport: SportConfig) -> int | None:
    """NFL: parse week number from 'Week N' string. NBA: assigned post-sort."""
    if sport.name == "nfl":
        try:
            return int(game["game"].get("week", "").split(" ")[1])
        except (IndexError, ValueError):
            return None
    return None


def get_upcoming_dates(
    raw_games: list[dict], sport: SportConfig, next_period: int
) -> list[str]:
    """Return ISO datetime strings for unplayed games in the upcoming period."""
    if sport.name == "nfl":
        next_week_str = f"Week {next_period}"
        dates = [
            game["game"]["date"]["date"] + "T" + game["game"]["date"]["time"] + ":00Z"
            for game in raw_games
            if game["game"].get("week") == next_week_str
            and game["game"]["status"]["short"] == "NS"
        ]
        return list(set(dates))

    # NBA: next date that has unstarted regular-season games
    unplayed = [
        g for g in raw_games
        if g["game"]["status"]["short"] == "NS"
        and g["game"].get("stage") == "Regular Season"
    ]
    if not unplayed:
        return []
    next_date = min(g["game"]["date"]["date"] for g in unplayed)
    return [
        g["game"]["date"]["date"] + "T" + g["game"]["date"]["time"] + ":00Z"
        for g in unplayed if g["game"]["date"]["date"] == next_date
    ]


def fetch_historical_spreads(
    sport: SportConfig, game_dates: list[str]
) -> pd.DataFrame:
    """
    Fetch spreads for a completed period via the paid historical Odds API.

    Snapshots 1 hour before the earliest game date so lines are open,
    then filters the response to only the actual game dates.
    """
    cfg = _read_config()
    earliest = datetime.datetime.strptime(min(game_dates), "%Y-%m-%d")
    snapshot = (earliest - datetime.timedelta(hours=1)).strftime(_ODDS_FMT)

    r = requests.get(
        f"https://api.the-odds-api.com/v4/historical/sports"
        f"/{sport.odds_api_sport}/odds"
        f"?apiKey={cfg['spreads']['key_paid']}&regions=us&markets=spreads"
        f"&oddsFormat=american&date={snapshot}"
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    print(f"  Odds API requests remaining: {r.headers.get('x-requests-remaining', '?')}")

    game_date_set = set(game_dates)
    data = [
        g for g in data
        if datetime.datetime.strptime(g["commence_time"], _ODDS_FMT).strftime("%Y-%m-%d")
        in game_date_set
    ]
    return _parse_spreads(data)


def fetch_upcoming_spreads(
    sport: SportConfig, dates: list[str] | None = None, key_type: str = "free"
) -> pd.DataFrame:
    """
    Fetch spread lines for upcoming games from the-odds-api.com.

    key_type='free' : current odds, no date filtering
    key_type='paid' : historical snapshot at min(dates)

    Returns a DataFrame indexed by team with columns: spread, opponent, order.
    """
    cfg = _read_config()
    api_key = cfg["spreads"][f"key_{key_type}"]

    if key_type == "free":
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{sport.odds_api_sport}/odds"
            f"?regions=us&markets=spreads&oddsFormat=american&apiKey={api_key}"
        )
        r.raise_for_status()
        data = r.json()
    else:
        if not dates:
            raise ValueError("dates must be provided when key_type='paid'")
        r = requests.get(
            f"https://api.the-odds-api.com/v4/historical/sports"
            f"/{sport.odds_api_sport}/odds"
            f"?apiKey={api_key}&regions=us&markets=spreads"
            f"&oddsFormat=american&date={min(dates)}"
        )
        r.raise_for_status()
        data = r.json()["data"]

    print(f"Odds API requests remaining: {r.headers.get('x-requests-remaining', '?')}")

    if key_type == "paid" and dates:
        min_date = datetime.datetime.strptime(min(dates), _ODDS_FMT).date()
        max_date = datetime.datetime.strptime(max(dates), _ODDS_FMT).date()
        data = [
            g for g in data
            if min_date
            <= datetime.datetime.strptime(g["commence_time"], _ODDS_FMT).date()
            <= max_date
        ]
    else:
        cutoff = datetime.datetime.utcnow() + datetime.timedelta(days=7)
        data = [
            g for g in data
            if datetime.datetime.strptime(g["commence_time"], _ODDS_FMT) < cutoff
        ]

    return _parse_spreads(data)


def _parse_spreads(spreaddata: list[dict]) -> pd.DataFrame:
    """Extract spread lines from an Odds API game list into a team-indexed DataFrame."""
    spreads: dict[str, float] = {}
    opponents: dict[str, str] = {}
    order: dict[str, int] = {}

    for i, game in enumerate(spreaddata):
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        for market in bookmakers[0]["markets"]:
            if market["key"] != "spreads":
                continue
            t0, t1 = market["outcomes"][0]["name"], market["outcomes"][1]["name"]
            spreads[t0] = market["outcomes"][0]["point"]
            spreads[t1] = market["outcomes"][1]["point"]
            opponents[t0], opponents[t1] = t1, t0
            order[t0] = order[t1] = i

    return pd.DataFrame({
        "spread": pd.Series(spreads),
        "opponent": pd.Series(opponents),
        "order": pd.Series(order),
    })


# =============================================================================
# MODEL
# =============================================================================
#
# Dual XGBoost:
#   Regression    — predicts SpreadScore (point_diff + spread_line)
#   Classification — predicts whether team covers (SpreadScore > 0)
#
# Features: last N SpreadScore values per team + team/season/period categoricals.
# Hyperparameters tuned via Hyperopt TPE search.
#
# Note on sign inversion: empirically the raw predictions are systematically
# backwards. Both outputs are negated until the root cause is resolved.

_XGB_FIXED = {"enable_categorical": True, "tree_method": "hist"}
_CAT_COLS = ("team", "season", "period")

_HYPEROPT_SPACE = {
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
    "max_depth": hp.quniform("max_depth", 2, 5, 1),
    "n_estimators": hp.quniform("n_estimators", 50, 200, 10),
    "reg_lambda": hp.uniform("reg_lambda", 1, 5),
    "reg_alpha": hp.uniform("reg_alpha", 1, 5),
    "min_child_weight": hp.uniform("min_child_weight", 1, 5),
}


def build_features(
    games_df: pd.DataFrame,
    next_period: int,
    lookback: int,
    validation_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build train/test/validation splits from long-format historical game data.

    Slides a window of size `lookback` across all completed periods, producing
    one training sample per (team, season, window). The validation season is
    held out entirely from the train/test split.
    """
    df = games_df[["team", "season", "period", "spreadscore"]].dropna()
    pivot = df.pivot_table(index=["team", "season"], columns="period", values="spreadscore")

    is_val = pivot.index.get_level_values("season") == validation_season
    train_pivot, val_pivot = pivot[~is_val], pivot[is_val]

    X_parts, y_parts, Xv_parts, yv_parts = [], [], [], []
    for start in range(1, next_period - lookback + 1):
        target = start + lookback
        _collect_window(train_pivot, start, lookback, target, X_parts, y_parts)
        _collect_window(val_pivot, start, lookback, target, Xv_parts, yv_parts)

    if not X_parts:
        raise ValueError(
            f"No training windows found (next_period={next_period}, lookback={lookback}). "
            f"Need at least {lookback + 1} periods of data."
        )

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if Xv_parts:
        X_val = pd.concat(Xv_parts, ignore_index=True)
        y_val = pd.concat(yv_parts, ignore_index=True).astype(float)
    else:
        X_val = pd.DataFrame(columns=X.columns)
        y_val = pd.Series(dtype=float)

    return X_train, X_test, y_train, y_test, X_val, y_val


def _collect_window(
    pivot: pd.DataFrame, start: int, lookback: int, target: int,
    X_out: list, y_out: list,
) -> None:
    if target not in pivot.columns:
        return
    feature_cols = [c for c in range(start, start + lookback) if c in pivot.columns]
    if len(feature_cols) < lookback:
        return
    y = pivot[target].dropna()
    X = pivot[feature_cols].loc[y.index].dropna()
    y = y.loc[X.index]
    if X.empty:
        return

    df = X.copy().reset_index()
    df = df.rename(columns={col: f"{lookback - i}_ago" for i, col in enumerate(feature_cols)})
    df["period"] = target
    for col in _CAT_COLS:
        df[col] = df[col].astype("category")
    for col in df.columns:
        if col not in _CAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    X_out.append(df)
    y_out.append(pd.Series(y.values, dtype=float))


def build_prediction_features(
    season_games: pd.DataFrame, next_period: int, lookback: int, season: int
) -> pd.DataFrame:
    """
    Build the prediction feature matrix using the last `lookback` completed
    periods per team. Teams with fewer periods get NaN-padded features.
    """
    completed = season_games[season_games["period"] < next_period]
    if completed.empty:
        return pd.DataFrame()

    pivot = completed.pivot_table(index="team", columns="period", values="spreadscore")
    available = min(lookback, pivot.shape[1])
    X = pivot.iloc[:, -available:].copy()

    for missing in range(lookback - available, 0, -1):
        X.insert(0, f"_pad_{missing}", np.nan)

    X.columns = [f"{lookback - i}_ago" for i in range(lookback)]
    X = X.reset_index()
    X["season"] = season
    X["period"] = next_period
    for col in _CAT_COLS:
        X[col] = X[col].astype("category")
    for col in X.columns:
        if col not in _CAT_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def _encode_cover(y: pd.Series) -> pd.Series:
    """Map sign of SpreadScore to binary: positive (covered) → 1, else → 0."""
    return np.sign(y).replace({-1: 0, 1: 1}).astype(int)


def _tune(model_class: type, X_train: pd.DataFrame, y_train: pd.Series, max_evals: int) -> dict:
    scoring = "f1" if model_class is XGBClassifier else "neg_mean_squared_error"

    def objective(params: dict) -> float:
        p = {**_XGB_FIXED, **params,
             "max_depth": int(params["max_depth"]),
             "n_estimators": int(params["n_estimators"])}
        return -cross_val_score(model_class(**p), X_train, y_train, cv=5, scoring=scoring).mean()

    best = fmin(fn=objective, space=_HYPEROPT_SPACE, algo=tpe.suggest, max_evals=max_evals)
    return {**best, "max_depth": int(best["max_depth"]), "n_estimators": int(best["n_estimators"])}


def train_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None,
    max_evals: int = 10,
) -> tuple[XGBRegressor, XGBClassifier, dict]:
    """Tune and fit regression + classification models. Returns (reg, clas, scores)."""
    y_train_enc = _encode_cover(y_train)
    y_test_enc = _encode_cover(y_test)

    print("  Tuning regression model...")
    reg = XGBRegressor(**_XGB_FIXED, **_tune(XGBRegressor, X_train, y_train, max_evals))
    reg.fit(X_train, y_train)

    print("  Tuning classification model...")
    clas = XGBClassifier(**_XGB_FIXED, **_tune(XGBClassifier, X_train, y_train_enc, max_evals))
    clas.fit(X_train, y_train_enc)

    scores = {
        "reg_train_r2": reg.score(X_train, y_train),
        "reg_test_r2": reg.score(X_test, y_test),
        "clas_train_acc": clas.score(X_train, y_train_enc),
        "clas_test_acc": clas.score(X_test, y_test_enc),
    }
    if X_val is not None and not X_val.empty:
        scores["reg_val_r2"] = reg.score(X_val, y_val)
        scores["clas_val_acc"] = clas.score(X_val, _encode_cover(y_val))

    return reg, clas, scores


def predict(reg: XGBRegressor, clas: XGBClassifier, X_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions. Returns DataFrame indexed by team with
    columns: predspread, coverprob. Both values are negated — see note above.
    """
    predspread = -reg.predict(X_pred[reg.get_booster().feature_names])
    coverprob = 1.0 - clas.predict_proba(X_pred[clas.get_booster().feature_names])[:, 1]
    return pd.DataFrame(
        {"predspread": predspread, "coverprob": coverprob},
        index=X_pred["team"].values,
    )


# =============================================================================
# PIPELINE
# =============================================================================

def run(
    sport: SportConfig,
    season: int,
    next_period: int,
    lookback: int,
    client: MongoClient,
    key_type: str = "free",
    max_evals: int = 10,
) -> pd.DataFrame:
    """
    Full weekly pipeline:
      1. Fetch season results and upsert with SpreadScores.
      2. Fetch upcoming spreads and store skeleton rows.
      3. Build features, tune, and train models.
      4. Generate predictions and upsert.

    Returns the prediction DataFrame for the upcoming period.
    """
    # Step 1: game results
    print(f"[1/4] Fetching {sport.name.upper()} {season} game results...")
    raw_games = fetch_season_games(sport, season)
    fresh = parse_game_results(raw_games, sport, season)

    if fresh.empty:
        print("  No completed games found.")
        return pd.DataFrame()

    stored = db_fetch_games(client, sport.name, season)
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
    db_upsert_games(client, fresh.to_dict("records"))
    print(f"  Upserted {len(fresh)} game rows.")

    # Step 2: upcoming spreads
    print(f"[2/4] Fetching upcoming spreads (key_type={key_type})...")
    dates = get_upcoming_dates(raw_games, sport, next_period)
    spreads_df = fetch_upcoming_spreads(sport, dates=dates, key_type=key_type)

    if spreads_df.empty:
        print("  No upcoming spreads found.")
    else:
        db_upsert_games(client, [
            {"sport": sport.name, "team": team, "opponent": row["opponent"],
             "season": season, "period": next_period, "spread": float(row["spread"])}
            for team, row in spreads_df.iterrows()
        ])
        print(f"  Stored spreads for {len(spreads_df)} teams.")

    # Step 3: train
    print("[3/4] Building features and training models...")
    all_games = db_fetch_games(client, sport.name)
    X_train, X_test, y_train, y_test, X_val, y_val = build_features(
        all_games, next_period, lookback, sport.validation_season
    )
    print(f"  Rows — train: {len(X_train)}, test: {len(X_test)}, val: {len(X_val)}")
    reg, clas, scores = train_models(X_train, X_test, y_train, y_test, X_val, y_val, max_evals)
    print(f"  Scores: {scores}")

    # Step 4: predict
    print("[4/4] Generating predictions...")
    season_games = db_fetch_games(client, sport.name, season)
    X_pred = build_prediction_features(season_games, next_period, lookback, season)
    if X_pred.empty:
        print("  Not enough data to predict.")
        return pd.DataFrame()

    preds = predict(reg, clas, X_pred)
    pred_df = preds.join(spreads_df[["opponent", "spread"]], how="left") if not spreads_df.empty \
        else preds.assign(opponent=np.nan, spread=np.nan)

    def _rel_diff(row: pd.Series, col: str) -> float | None:
        opp = row.get("opponent")
        if pd.isna(opp) or opp not in preds.index:
            return None
        return float(row[col]) - float(preds.loc[opp, col])

    pred_df["predspread_diff"] = pred_df.apply(_rel_diff, col="predspread", axis=1)
    pred_df["coverprob_diff"] = pred_df.apply(_rel_diff, col="coverprob", axis=1)

    db_upsert_predictions(client, [
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


# =============================================================================
# SEED
# =============================================================================

def _group_by_period(game_df: pd.DataFrame) -> dict[int, list[str]]:
    """Return period → sorted unique game dates (YYYY-MM-DD)."""
    groups: dict[int, list[str]] = {}
    for _, row in game_df[["period", "date"]].dropna().iterrows():
        groups.setdefault(int(row["period"]), []).append(row["date"])
    return {k: sorted(set(v)) for k, v in groups.items()}


def seed_season(
    client: MongoClient, sport: SportConfig, season: int, request_delay: float = 1.0
) -> None:
    """
    Backfill one season of historical game data into MongoDB.
    Fetches game results (free API) and spread lines (paid API) per period,
    computes SpreadScore, and upserts everything.
    """
    print(f"\n{'=' * 60}")
    print(f"Seeding {sport.name.upper()} season {season}")
    print(f"{'=' * 60}")

    raw_games = fetch_season_games(sport, season)
    game_df = parse_game_results(raw_games, sport, season)

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
                spread_df = fetch_historical_spreads(sport, dates)
                time.sleep(request_delay)
                if not spread_df.empty:
                    period_games["spread"] = period_games["team"].map(spread_df["spread"])
                    period_games["spreadscore"] = period_games["diff"] + period_games["spread"]
            except Exception as exc:
                print(f"  Period {int(period)}: spread fetch failed ({exc}). Storing without spread.")

        all_records.extend(period_games.to_dict("records"))
        covered = period_games["spreadscore"].notna().sum()
        print(f"  Period {int(period):3d}: {len(period_games):2d} teams, {covered:2d} with SpreadScore.")

    db_upsert_games(client, all_records)
    print(f"  Done. Upserted {len(all_records)} records.")


# =============================================================================
# CLI
# =============================================================================

def _cli_setup(args: argparse.Namespace) -> None:
    client = db_connect()
    try:
        db_create_indexes(client)
    finally:
        client.close()


def _cli_seed(args: argparse.Namespace) -> None:
    sport = SPORTS[args.sport]
    client = db_connect()
    try:
        for season in sorted(args.seasons):
            seed_season(client, sport, season, request_delay=args.delay)
    finally:
        client.close()
    print("\nSeeding complete.")


def _cli_run(args: argparse.Namespace) -> None:
    sport = SPORTS[args.sport]
    client = db_connect()
    try:
        result = run(
            sport=sport,
            season=args.season,
            next_period=args.period,
            lookback=args.lookback,
            client=client,
            key_type=args.key_type,
            max_evals=args.max_evals,
        )
        if not result.empty:
            print("\nPredictions:")
            print(result[["opponent", "spread", "predspread", "coverprob"]].to_string())
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sports betting spread prediction pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    sub.add_parser("setup", help="Create MongoDB indexes (run once).")

    # seed
    p_seed = sub.add_parser("seed", help="Backfill historical game data.")
    p_seed.add_argument("--sport", choices=list(SPORTS), required=True)
    p_seed.add_argument("--seasons", type=int, nargs="+", required=True,
                        help="Season start years (e.g. 2022 2023 2024).")
    p_seed.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between paid API calls (default: 1.0).")

    # run
    p_run = sub.add_parser("run", help="Run the weekly prediction pipeline.")
    p_run.add_argument("--sport", choices=list(SPORTS), required=True)
    p_run.add_argument("--season", type=int, default=datetime.datetime.now().year,
                       help="Season start year (default: current year).")
    p_run.add_argument("--period", type=int, required=True,
                       help="Next period to predict (NFL week or NBA game number).")
    p_run.add_argument("--lookback", type=int, default=5,
                       help="Prior periods used as features (default: 5).")
    p_run.add_argument("--key-type", choices=["free", "paid"], default="free",
                       dest="key_type", help="Odds API key type (default: free).")
    p_run.add_argument("--max-evals", type=int, default=10, dest="max_evals",
                       help="Hyperopt budget per model (default: 10).")

    args = parser.parse_args()
    {"setup": _cli_setup, "seed": _cli_seed, "run": _cli_run}[args.command](args)
