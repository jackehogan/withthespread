"""
Sport-agnostic data fetching and transformation.

Two external APIs:
  - api-sports.io     : historical game results (scores, dates)
  - the-odds-api.com  : betting spreads (live and historical)

API structure differences (handled internally by _normalize_game)
-----------------------------------------------------------------
Football                          Basketball
--------------------------------  --------------------------------
game["game"]["stage"]             game["stage"]  (null = regular season)
game["game"]["status"]["short"]   game["status"]["short"]
game["game"]["date"]["date"]      game["date"]   (ISO string)
game["game"]["date"]["time"]      game["time"]
game["game"]["week"]              game["week"]   (always null for NBA)
game["teams"]["away"]["name"]     game["teams"]["visitors"]["name"]
game["scores"]["away"]["total"]   game["scores"]["visitors"]["livePoints"]
game["scores"]["home"]["total"]   game["scores"]["home"]["livePoints"]

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
import time

import pandas as pd
import requests

from config import SportConfig

_ODDS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Stage strings that mean a game is NOT regular season for each sport.
# Anything not in this set (including null/empty) is treated as regular season.
_NON_REGULAR_STAGES = {
    "nfl": {"Pre Season", "Post Season", "Pro Bowl"},
    "nba": {"NBA Playoffs", "Play-In Tournament", "All-Star"},
}


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
# Game normalisation — handles structural differences between sport APIs
# ---------------------------------------------------------------------------

def _normalize_game(game: dict, sport: SportConfig) -> dict | None:
    """
    Extract a consistent set of fields from a raw api-sports.io game object.

    Returns None if the game should be skipped (non-regular-season or
    scores not yet available). Otherwise returns:
        status    str   "NS" = not started, "FT" = finished, etc.
        home_team str
        away_team str
        home_score int | None
        away_score int | None
        game_date  str  "YYYY-MM-DD"
        game_time  str  "HH:MM"
        week       str  "Week N" for NFL, empty for NBA
    """
    if sport.name == "nfl":
        g = game["game"]
        stage = g.get("stage") or ""
        if stage in _NON_REGULAR_STAGES["nfl"] or stage == "":
            # NFL: must explicitly be "Regular Season"
            if stage != "Regular Season":
                return None
        return {
            "status":     g["status"]["short"],
            "home_team":  game["teams"]["home"]["name"],
            "away_team":  game["teams"]["away"]["name"],
            "home_score": game["scores"]["home"]["total"],
            "away_score": game["scores"]["away"]["total"],
            "game_date":  g["date"]["date"],
            "game_time":  g["date"].get("time", "00:00") or "00:00",
            "week":       g.get("week") or "",
        }

    # Basketball ---------------------------------------------------------------
    # Regular season games have stage=null; skip known non-regular stages.
    stage = game.get("stage") or ""
    if stage and stage in _NON_REGULAR_STAGES["nba"]:
        return None

    # All-Star / celebrity games also have stage=null — filter by team allowlist.
    if sport.known_teams is not None:
        home = game["teams"]["home"]["name"]
        away = game["teams"]["away"]["name"]
        if home not in sport.known_teams or away not in sport.known_teams:
            return None

    # game["date"] is an ISO string e.g. "2024-10-04T16:00:00+00:00"
    date_raw = game.get("date") or ""
    game_date = date_raw[:10] if date_raw else ""
    game_time = game.get("time") or "00:00"

    # Structure is the same as football apart from the missing "game" wrapper
    return {
        "status":     game["status"]["short"],
        "home_team":  game["teams"]["home"]["name"],
        "away_team":  game["teams"]["away"]["name"],
        "home_score": game["scores"]["home"]["total"],
        "away_score": game["scores"]["away"]["total"],
        "home_q4":    game["scores"]["home"].get("quarter_4"),
        "away_q4":    game["scores"]["away"].get("quarter_4"),
        "game_date":  game_date,
        "game_time":  game_time,
        "week":       "",
    }


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
        fields = _normalize_game(game, sport)
        if fields is None:
            continue
        if fields["status"] == "NS":
            continue
        if fields["home_score"] is None or fields["away_score"] is None:
            continue

        home_score = int(fields["home_score"])
        away_score = int(fields["away_score"])
        period = _extract_period(fields, sport)

        home_q4 = fields.get("home_q4")
        away_q4 = fields.get("away_q4")
        q4_available = home_q4 is not None and away_q4 is not None
        home_q4_i = int(home_q4) if q4_available else None
        away_q4_i = int(away_q4) if q4_available else None

        for team, opp, score, opp_score, is_home, q4, opp_q4 in (
            (fields["home_team"], fields["away_team"], home_score, away_score, True,  home_q4_i, away_q4_i),
            (fields["away_team"], fields["home_team"], away_score, home_score, False, away_q4_i, home_q4_i),
        ):
            records.append({
                "sport": sport.name, "team": team, "opponent": opp,
                "season": season, "period": period, "date": fields["game_date"],
                "score": score, "opp_score": opp_score, "diff": score - opp_score,
                "home": is_home,
                "q4_diff": (q4 - opp_q4) if q4 is not None else None,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = filter_regular_season(df, sport, season)

    if sport.name == "nba":
        # Assign sequential game numbers per team, ordered by date
        df = df.sort_values("date")
        df["period"] = df.groupby("team").cumcount() + 1
    return df.reset_index(drop=True)


def filter_regular_season(
    df: pd.DataFrame,
    sport: SportConfig,
    season: int,
) -> pd.DataFrame:
    """
    Remove preseason and playoff rows using the date bounds in SportConfig.

    For cross-year seasons (NBA), start is in `season` and end is in
    `season + 1`. For same-year seasons (NFL) this is a no-op since NFL
    already filters by stage string. Returns df unchanged if no bounds set.
    """
    if sport.regular_season_start is None or sport.regular_season_end is None:
        return df
    if df.empty:
        return df

    sm, sd = sport.regular_season_start
    em, ed = sport.regular_season_end
    end_year = season + 1 if sm > em else season  # cross-year if start month > end month

    start_date = f"{season}-{sm:02d}-{sd:02d}"
    end_date = f"{end_year}-{em:02d}-{ed:02d}"

    return df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()


def _extract_period(fields: dict, sport: SportConfig) -> int | None:
    """NFL: parse week number from 'Week N'. NBA: assigned post-sort, returns None."""
    if sport.name == "nfl":
        try:
            return int(fields["week"].split(" ")[1])
        except (IndexError, ValueError):
            return None
    return None


def get_upcoming_dates(
    raw_games: list[dict],
    sport: SportConfig,
    next_period: int,
) -> list[str]:
    """
    Return ISO datetime strings for unplayed games in the upcoming period.

    NFL : games matching 'Week {next_period}' that haven't started yet.
    NBA : all unstarted regular-season games on the next available date.
    """
    upcoming = []
    for game in raw_games:
        fields = _normalize_game(game, sport)
        if fields is None or fields["status"] != "NS":
            continue

        if sport.name == "nfl":
            if fields["week"] == f"Week {next_period}":
                upcoming.append(f"{fields['game_date']}T{fields['game_time']}:00Z")
        else:
            upcoming.append((fields["game_date"], f"{fields['game_date']}T{fields['game_time']}:00Z"))

    if sport.name == "nfl":
        return list(set(upcoming))

    # NBA: return only games on the earliest upcoming date
    if not upcoming:
        return []
    next_date = min(d for d, _ in upcoming)
    return list(set(dt for d, dt in upcoming if d == next_date))


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
        f"?apiKey={cfg['spreads']['key_paid']}&regions=us&markets=h2h,spreads,totals"
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
            f"?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={api_key}"
        )
        r.raise_for_status()
        data = r.json()
    else:
        if not dates:
            raise ValueError("dates must be provided when key_type='paid'")
        r = requests.get(
            f"https://api.the-odds-api.com/v4/historical/sports"
            f"/{sport.odds_api_sport}/odds"
            f"?apiKey={api_key}&regions=us&markets=h2h,spreads,totals"
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


def _american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability (0-1), ignoring vig."""
    if odds >= 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _parse_spreads(spreaddata: list[dict]) -> pd.DataFrame:
    """
    Extract all available market data from an Odds API game list.

    Returns a DataFrame indexed by team with columns:
        spread          : point spread for this team
        spread_juice    : juice (price) on the spread for this team
        total           : over/under line (same for both teams in a game)
        moneyline       : h2h American odds for this team
        implied_prob    : win probability implied by moneyline (vig-inclusive)
        opponent        : opponent team name
        home            : True if this team is the home team
        game_date       : YYYY-MM-DD of the game
        order           : game index (for deduplication)
    """
    spreads:       dict[str, float] = {}
    spread_juice:  dict[str, float] = {}
    totals:        dict[str, float] = {}
    moneylines:    dict[str, float] = {}
    implied_probs: dict[str, float] = {}
    opponents:     dict[str, str]   = {}
    order:         dict[str, int]   = {}
    home_flag:     dict[str, bool]  = {}
    game_dates:    dict[str, str]   = {}

    for i, game in enumerate(spreaddata):
        home_team = game.get("home_team", "")
        game_date = game.get("commence_time", "")[:10]
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        # Use the first bookmaker that has the most markets
        bm = max(bookmakers, key=lambda b: len(b["markets"]))
        markets_by_key = {m["key"]: m for m in bm["markets"]}

        # --- spreads ---
        if "spreads" in markets_by_key:
            for outcome in markets_by_key["spreads"]["outcomes"]:
                t = outcome["name"]
                spreads[t]      = outcome["point"]
                spread_juice[t] = outcome.get("price", None)
                opponents[t]    = ""   # filled below
                order[t]        = i
                home_flag[t]    = (t == home_team)
                game_dates[t]   = game_date
            outs = markets_by_key["spreads"]["outcomes"]
            if len(outs) == 2:
                opponents[outs[0]["name"]] = outs[1]["name"]
                opponents[outs[1]["name"]] = outs[0]["name"]

        # --- totals (game-level, same for both teams) ---
        if "totals" in markets_by_key:
            total_line = next(
                (o["point"] for o in markets_by_key["totals"]["outcomes"]
                 if o["name"] == "Over"), None
            )
            if total_line is not None:
                for t in list(spreads):
                    if game_dates.get(t) == game_date and order.get(t) == i:
                        totals[t] = total_line

        # --- h2h (moneyline) ---
        if "h2h" in markets_by_key:
            for outcome in markets_by_key["h2h"]["outcomes"]:
                t = outcome["name"]
                price = outcome.get("price")
                if price is not None:
                    moneylines[t]    = price
                    implied_probs[t] = _american_to_implied_prob(price)

    return pd.DataFrame({
        "spread":       pd.Series(spreads),
        "spread_juice": pd.Series(spread_juice),
        "total":        pd.Series(totals),
        "moneyline":    pd.Series(moneylines),
        "implied_prob": pd.Series(implied_probs),
        "opponent":     pd.Series(opponents),
        "home":         pd.Series(home_flag),
        "game_date":    pd.Series(game_dates),
        "order":        pd.Series(order),
    })


# ---------------------------------------------------------------------------
# NBA advanced ratings — nba_api (free, no credits)
# ---------------------------------------------------------------------------

# Season format mapping: DB season year (start year) -> nba_api season string
def _nba_season_str(season: int) -> str:
    """Convert DB season year (e.g. 2024) to nba_api format (e.g. '2024-25')."""
    return f"{season}-{str(season + 1)[-2:]}"


def fetch_nba_ratings(season: int, date_to: str | None = None) -> pd.DataFrame:
    """
    Fetch cumulative advanced team ratings from stats.nba.com.

    Uses LeagueDashTeamStats with measure_type='Advanced'. If date_to is
    provided (YYYY-MM-DD), returns ratings accumulated only through that date,
    giving a pre-period snapshot with no leakage.

    Parameters
    ----------
    season    : Season start year (e.g. 2024 for 2024-25)
    date_to   : Optional cutoff date string 'YYYY-MM-DD'. If None, uses
                all available games for the season.

    Returns
    -------
    DataFrame indexed by team name with columns:
        off_rating, def_rating, net_rating
    Returns empty DataFrame on failure.
    """
    try:
        from nba_api.stats.endpoints import LeagueDashTeamStats
    except ImportError:
        print("  nba_api not installed. Run: pip install nba_api")
        return pd.DataFrame()

    kwargs: dict = {
        "season":                      _nba_season_str(season),
        "measure_type_detailed_defense": "Advanced",
        "per_mode_detailed":           "PerGame",
    }
    if date_to is not None:
        # nba_api expects MM/DD/YYYY
        try:
            dt = datetime.datetime.strptime(date_to, "%Y-%m-%d")
            kwargs["date_to_nullable"] = dt.strftime("%m/%d/%Y")
        except ValueError:
            pass

    try:
        df = LeagueDashTeamStats(**kwargs).get_data_frames()[0]
    except Exception as exc:
        print(f"  fetch_nba_ratings failed ({exc})")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    keep = {"TEAM_NAME": "team", "E_OFF_RATING": "off_rating",
            "E_DEF_RATING": "def_rating", "E_NET_RATING": "net_rating"}
    df = df[list(keep)].rename(columns=keep).set_index("team")

    # Drop rows with no data (teams with 0 games played in range)
    df = df.dropna(subset=["off_rating", "def_rating"])
    return df
