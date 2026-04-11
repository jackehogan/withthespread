"""
Sport-agnostic data fetching and transformation.

Two external APIs:
  - api-sports.io     : historical game results (scores, dates)
  - the-odds-api.com  : betting spreads (live and historical)

Public interface
----------------
fetch_season_games(sport, season)              -> list[dict]
parse_game_results(raw, sport, season)         -> pd.DataFrame
get_upcoming_dates(raw, sport, next_period)    -> list[str]
fetch_historical_spreads(sport, game_dates)    -> pd.DataFrame
fetch_upcoming_spreads(sport, dates, key_type) -> pd.DataFrame
"""

import datetime
import http.client
import json

import pandas as pd
import requests

from config import SportConfig

_ODDS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _read_config(path: str = "data/config.txt") -> dict:
    with open(path) as f:
        return json.load(f)


def _api_sports_get(host: str, path: str, api_key: str) -> dict:
    conn = http.client.HTTPSConnection(host)
    conn.request("GET", path, headers={
        "x-rapidapi-host": host,
        "x-rapidapi-key": api_key,
    })
    return json.loads(conn.getresponse().read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Game results — api-sports.io
# ---------------------------------------------------------------------------

def fetch_season_games(
    sport: SportConfig,
    season: int,
    config_path: str = "data/config.txt",
) -> list[dict]:
    """Fetch all games for a season from api-sports.io. Returns the raw API list."""
    cfg = _read_config(config_path)
    return _api_sports_get(
        host=sport.api_sports_host,
        path=f"/games?league={sport.api_sports_league}&season={sport.format_season(season)}",
        api_key=cfg["results"]["key"],
    )["response"]


def parse_game_results(
    raw_games: list[dict],
    sport: SportConfig,
    season: int,
) -> pd.DataFrame:
    """
    Parse raw api-sports.io game objects into a tidy long-format DataFrame.
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
        # NBA has no week numbers — assign sequential game number per team by date.
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
    raw_games: list[dict],
    sport: SportConfig,
    next_period: int,
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

    # NBA: next date with unstarted regular-season games
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


# ---------------------------------------------------------------------------
# Betting spreads — the-odds-api.com
# ---------------------------------------------------------------------------

def fetch_historical_spreads(
    sport: SportConfig,
    game_dates: list[str],
    config_path: str = "data/config.txt",
) -> pd.DataFrame:
    """
    Fetch spreads for a completed period via the paid historical Odds API.

    Snapshots 1 hour before the earliest game date so lines are open, then
    filters the response down to only the actual game dates.

    Returns a DataFrame indexed by team with columns: spread, opponent, order.
    """
    cfg = _read_config(config_path)
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

    # Filter to the actual game dates (not the snapshot date)
    game_date_set = set(game_dates)
    data = [
        g for g in data
        if datetime.datetime.strptime(g["commence_time"], _ODDS_FMT).strftime("%Y-%m-%d")
        in game_date_set
    ]
    return _parse_spreads(data)


def fetch_upcoming_spreads(
    sport: SportConfig,
    dates: list[str] | None = None,
    key_type: str = "free",
    config_path: str = "data/config.txt",
) -> pd.DataFrame:
    """
    Fetch spread lines for upcoming games from the-odds-api.com.

    key_type='free' : current live odds, no date filtering
    key_type='paid' : historical snapshot at min(dates)

    Returns a DataFrame indexed by team with columns: spread, opponent, order.
    """
    cfg = _read_config(config_path)
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
