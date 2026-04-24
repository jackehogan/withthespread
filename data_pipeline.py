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


# ---------------------------------------------------------------------------
# MLB — scores via statsapi (free, no key required)
# ---------------------------------------------------------------------------

# Regular-season game_type codes from statsapi
_MLB_REGULAR_GAME_TYPES = {"R"}

# Canonical team name corrections across seasons
# (statsapi uses the current name; historical name -> canonical)
_MLB_NAME_FIXES: dict[str, str] = {
    "Cleveland Indians": "Cleveland Guardians",  # renamed after 2021
}


def fetch_season_games_mlb(season: int) -> list[dict]:
    """
    Fetch all regular-season MLB game results for a calendar year via statsapi.

    Returns a list of raw statsapi game dicts (only completed regular-season
    games with scores). Spring training and playoffs are excluded.

    Parameters
    ----------
    season : Calendar year (e.g. 2023 for the 2023 MLB season).
    """
    try:
        import statsapi
    except ImportError:
        raise ImportError("MLB-StatsAPI not installed. Run: pip install MLB-StatsAPI")

    # Regular season spans late March / early April through late September.
    start = f"03/15/{season}"
    end   = f"10/10/{season}"

    raw = statsapi.schedule(start_date=start, end_date=end, sportId=1)
    return [
        g for g in raw
        if g.get("game_type") in _MLB_REGULAR_GAME_TYPES
        and g.get("status") == "Final"
        and g.get("away_score") is not None
        and g.get("home_score") is not None
    ]


def parse_game_results_mlb(
    raw_games: list[dict],
    season: int,
) -> pd.DataFrame:
    """
    Convert raw statsapi game dicts to the standard long-format DataFrame.

    One row per team per game with columns:
        sport, team, opponent, season, period, date,
        score, opp_score, diff, home,
        sp_name  (probable starter name, may be empty string)

    period is assigned as sequential game number per team ordered by date,
    matching the NBA convention (1–162).
    """
    from config import SPORTS
    sport = SPORTS["mlb"]

    records = []
    for g in raw_games:
        home  = _MLB_NAME_FIXES.get(g["home_name"], g["home_name"])
        away  = _MLB_NAME_FIXES.get(g["away_name"], g["away_name"])

        # Skip if either team is not in the known-teams allowlist
        if sport.known_teams is not None:
            if home not in sport.known_teams or away not in sport.known_teams:
                continue

        home_score = int(g["home_score"])
        away_score = int(g["away_score"])
        game_date  = g["game_date"][:10]   # YYYY-MM-DD

        home_sp = g.get("home_probable_pitcher") or ""
        away_sp = g.get("away_probable_pitcher") or ""

        for team, opp, score, opp_sc, is_home, sp_name in (
            (home, away, home_score, away_score, True,  home_sp),
            (away, home, away_score, home_score, False, away_sp),
        ):
            records.append({
                "sport":     "mlb",
                "team":      team,
                "opponent":  opp,
                "season":    season,
                "period":    None,          # assigned below
                "date":      game_date,
                "score":     score,
                "opp_score": opp_sc,
                "diff":      score - opp_sc,
                "home":      int(is_home),
                "sp_name":   sp_name,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Apply regular-season date filter
    df = filter_regular_season(df, sport, season)

    # Assign sequential game number per team by date
    df = df.sort_values("date")
    df["period"] = df.groupby("team").cumcount() + 1

    return df.reset_index(drop=True)


def fetch_mlb_pitcher_stats(season: int) -> pd.DataFrame:
    """
    Fetch season pitcher stats for all qualified starters via statsapi.

    Uses the official MLB Stats API (free, no key). Returns prior-season stats
    so they can be joined to upcoming game rows without leakage.

    Parameters
    ----------
    season : The season to pull stats from (e.g. pass season-1 to get prior-year
             stats for a prediction in `season`).

    Returns
    -------
    DataFrame indexed by pitcher full name with columns:
        era, whip, k9, bb9, gs  (games started, for minimum qualifier filter)

    Returns empty DataFrame on failure.
    """
    try:
        import statsapi
    except ImportError:
        print("  MLB-StatsAPI not installed.")
        return pd.DataFrame()

    try:
        result = statsapi.get("stats", {
            "stats":       "season",
            "group":       "pitching",
            "sportId":     1,
            "season":      season,
            "gameType":    "R",
            "limit":       1000,
            "playerPool":  "ALL",    # include non-qualified starters
        })
    except Exception as exc:
        print(f"  fetch_mlb_pitcher_stats failed ({exc})")
        return pd.DataFrame()

    splits = result.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return pd.DataFrame()

    rows = []
    for s in splits:
        stat = s.get("stat", {})
        player = s.get("player", {})
        name = player.get("fullName", "")
        if not name:
            continue
        try:
            rows.append({
                "name":  name,
                "era":   float(stat.get("era", "nan") or "nan"),
                "whip":  float(stat.get("whip", "nan") or "nan"),
                "k9":    float(stat.get("strikeoutsPer9Inn", "nan") or "nan"),
                "bb9":   float(stat.get("walksPer9Inn", "nan") or "nan"),
                "gs":    int(stat.get("gamesStarted", 0) or 0),
            })
        except (ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("name")
    # Keep only pitchers with at least 1 start (filters out pure relievers)
    df = df[df["gs"] >= 1].drop(columns=["gs"])
    return df


# ---------------------------------------------------------------------------
# MLB — run lines from SBR CSV / Excel files
# ---------------------------------------------------------------------------
# Download historical MLB odds archives from:
#   https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlboddsarchives.htm
#
# Save files to a local directory (default: data/sbr/).
# Expected filename pattern: mlb_{season}.xlsx  (e.g. mlb_2023.xlsx)
# or any Excel/CSV file containing "mlb" and the year in the name.
#
# SBR MLB file format (one pair of rows per game: visitor first, then home):
#   Date | Rot | VH | Team | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | F | Open | Close | ML | 2H
#
# VH   : V = visitor (away), H = home
# F    : final runs scored
# ML   : moneyline in American odds (e.g. -150 means -150 to win 100)
# Open : opening run total (over/under); sometimes run line
# Close: closing run total
# Run line is always ±1.5 — favorite identified by who has the more negative ML.

def parse_sbr_mlb(filepath: str) -> pd.DataFrame:
    """
    Parse a single SBR MLB odds archive file (Excel or CSV).

    Expects the standard SBR format with VH, Team, F (final), and ML columns.
    Pairs visitor/home rows to reconstruct matchups and assigns run lines
    (always ±1.5; favorite identified by more negative ML).

    Parameters
    ----------
    filepath : Path to the .xlsx or .csv file.

    Returns
    -------
    DataFrame with columns:
        date, team, opponent, home, run_line, moneyline
    where run_line is -1.5 for the favourite and +1.5 for the underdog.
    Returns empty DataFrame if the file cannot be parsed.
    """
    fp = str(filepath)
    try:
        if fp.endswith(".csv"):
            raw = pd.read_csv(fp, header=0)
        else:
            raw = pd.read_excel(fp, header=0)
    except Exception as exc:
        print(f"  parse_sbr_mlb: could not read {fp} ({exc})")
        return pd.DataFrame()

    raw.columns = [str(c).strip().upper() for c in raw.columns]

    # Flexible column name mapping
    col_map = {}
    for c in raw.columns:
        cl = c.upper()
        if cl in ("VH", "V/H"):
            col_map["VH"] = c
        elif cl in ("TEAM",):
            col_map["TEAM"] = c
        elif cl in ("F", "FINAL", "SCORE"):
            col_map["F"] = c
        elif cl in ("ML", "MONEYLINE", "MONEY LINE"):
            col_map["ML"] = c
        elif cl in ("DATE",):
            col_map["DATE"] = c

    required = ["VH", "TEAM", "F", "ML", "DATE"]
    missing = [k for k in required if k not in col_map]
    if missing:
        print(f"  parse_sbr_mlb: missing columns {missing} in {fp}. "
              f"Found: {list(raw.columns)}")
        return pd.DataFrame()

    df = raw[[col_map[k] for k in required]].copy()
    df.columns = required

    # Parse date: SBR uses YYYYMMDD integers or MM/DD/YYYY strings
    def _parse_date(val) -> str:
        s = str(val).strip().replace("/", "-")
        try:
            if len(s) == 8 and s.isdigit():
                return f"{s[:4]}-{s[4:6]}-{s[6:]}"
            return pd.to_datetime(s).strftime("%Y-%m-%d")
        except Exception:
            return ""

    df["DATE"] = df["DATE"].apply(_parse_date)
    df = df[df["DATE"] != ""]

    # Normalise VH
    df["VH"] = df["VH"].astype(str).str.strip().str.upper()
    df = df[df["VH"].isin(["V", "H"])]

    # Normalise ML (may contain 'pk', 'PK', or NL for no line)
    def _parse_ml(val) -> float:
        try:
            return float(str(val).replace("pk", "100").replace("PK", "100"))
        except (ValueError, TypeError):
            return float("nan")

    df["ML"] = df["ML"].apply(_parse_ml)
    df["F"]  = pd.to_numeric(df["F"], errors="coerce")
    df = df.dropna(subset=["F", "ML"])

    # SBR team abbreviations -> full names
    _SBR_TEAM_MAP = {
        "ARI": "Arizona Diamondbacks",  "ATL": "Atlanta Braves",
        "BAL": "Baltimore Orioles",     "BOS": "Boston Red Sox",
        "CHC": "Chicago Cubs",          "CWS": "Chicago White Sox",
        "CIN": "Cincinnati Reds",       "CLE": "Cleveland Guardians",
        "COL": "Colorado Rockies",      "DET": "Detroit Tigers",
        "HOU": "Houston Astros",        "KC":  "Kansas City Royals",
        "LAA": "Los Angeles Angels",    "LAD": "Los Angeles Dodgers",
        "MIA": "Miami Marlins",         "MIL": "Milwaukee Brewers",
        "MIN": "Minnesota Twins",       "NYM": "New York Mets",
        "NYY": "New York Yankees",      "OAK": "Oakland Athletics",
        "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
        "SD":  "San Diego Padres",      "SF":  "San Francisco Giants",
        "SEA": "Seattle Mariners",      "STL": "St. Louis Cardinals",
        "TB":  "Tampa Bay Rays",        "TEX": "Texas Rangers",
        "TOR": "Toronto Blue Jays",     "WSH": "Washington Nationals",
        "WAS": "Washington Nationals",
    }
    df["TEAM"] = df["TEAM"].astype(str).str.strip().str.upper()
    df["TEAM"] = df["TEAM"].map(_SBR_TEAM_MAP).fillna(df["TEAM"])

    # Pair rows: each game is visitor (V) row immediately followed by home (H) row
    records = []
    rows_list = df.reset_index(drop=True)
    i = 0
    while i < len(rows_list) - 1:
        v_row = rows_list.iloc[i]
        h_row = rows_list.iloc[i + 1]
        if v_row["VH"] == "V" and h_row["VH"] == "H" and v_row["DATE"] == h_row["DATE"]:
            away_team = v_row["TEAM"]
            home_team = h_row["TEAM"]
            away_ml   = v_row["ML"]
            home_ml   = h_row["ML"]
            date      = v_row["DATE"]

            # Assign run lines: favourite (lower/more negative ML) gets -1.5
            if pd.isna(away_ml) or pd.isna(home_ml):
                away_rl, home_rl = float("nan"), float("nan")
            elif away_ml <= home_ml:
                away_rl, home_rl = -1.5, +1.5   # away is favourite
            else:
                away_rl, home_rl = +1.5, -1.5   # home is favourite

            for team, opp, ml, rl, is_home in (
                (away_team, home_team, away_ml, away_rl, 0),
                (home_team, away_team, home_ml, home_rl, 1),
            ):
                records.append({
                    "date":       date,
                    "team":       team,
                    "opponent":   opp,
                    "home":       is_home,
                    "run_line":   rl,
                    "moneyline":  ml,
                })
            i += 2
        else:
            i += 1   # row mismatch — skip one and try to re-sync

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def load_sbr_mlb(season: int, sbr_dir: str = "data/sbr") -> pd.DataFrame:
    """
    Find and parse the SBR CSV/Excel file for a given MLB season.

    Searches `sbr_dir` for files matching `*mlb*{season}*` (case-insensitive).
    Returns empty DataFrame if no matching file is found.

    Parameters
    ----------
    season  : Calendar year (e.g. 2023).
    sbr_dir : Directory containing downloaded SBR files.
    """
    import glob, os

    patterns = [
        os.path.join(sbr_dir, f"*mlb*{season}*.xlsx"),
        os.path.join(sbr_dir, f"*mlb*{season}*.xls"),
        os.path.join(sbr_dir, f"*mlb*{season}*.csv"),
        os.path.join(sbr_dir, f"*{season}*mlb*.xlsx"),
        os.path.join(sbr_dir, f"*{season}*mlb*.csv"),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=False)
        if matches:
            print(f"  Loading SBR file: {matches[0]}")
            return parse_sbr_mlb(matches[0])

    print(f"  No SBR file found for MLB {season} in {sbr_dir}/")
    print(f"  Download from: https://www.sportsbookreviewsonline.com/"
          f"scoresoddsarchives/mlb/mlboddsarchives.htm")
    print(f"  Save as: {sbr_dir}/mlb_{season}.xlsx")
    return pd.DataFrame()
