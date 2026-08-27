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
import re
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import SportConfig

_ODDS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# All game dates in the DB are keyed to US/Eastern — see _et_date.
_ET = ZoneInfo("America/New_York")


def _et_date(ts: str) -> str:
    """
    Convert a UTC timestamp to its US/Eastern calendar date ('YYYY-MM-DD').

    MLB's official game date is the local date at the ballpark, and statsapi
    reports it that way.  Every odds feed (ESPN, the-odds-api, ProphetX)
    timestamps in UTC instead, so any first pitch from 8pm ET onward lands on
    the *next* UTC day.  Slicing the raw string files those games under
    tomorrow, and the (team, date) merge against scores silently misses — which
    cost ~43% of West-Coast home games their run line before this was fixed.

    Eastern is a safe stand-in for ballpark-local time: the latest MLB first
    pitch is ~7pm PT (10pm ET), so the ET date always equals the official date.

    Falls back to the leading 10 characters if the timestamp cannot be parsed.
    """
    if not ts:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts[:10]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(_ET).strftime("%Y-%m-%d")

# Stage strings that mean a game is NOT regular season for each sport.
# Anything not in this set (including null/empty) is treated as regular season.
_NON_REGULAR_STAGES = {
    "nfl": {"Pre Season", "Post Season", "Pro Bowl"},
    "nba": {"NBA Playoffs", "Play-In Tournament", "All-Star"},
}


def _normalize_pitcher_name(name: str) -> str:
    """
    Normalize a pitcher name for reliable lookup across data sources.

    statsapi schedule returns accented characters (e.g. "Eury Pérez") but the
    encoding is sometimes mangled in transit.  Stripping all diacritics to ASCII
    equivalents and lowercasing ensures the schedule name matches the stats name
    regardless of source encoding.
    """
    import unicodedata
    if not name:
        return ""
    # Decompose to base + combining chars, then drop the combining chars
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_name.strip().lower()


def data_quality_report(games_df: pd.DataFrame, label: str = "") -> dict:
    """
    Report fill rates for all seeded columns and flag anything below threshold.

    Returns a dict of {col: fill_rate} for programmatic use.
    Prints a formatted report with warnings for low-fill columns.

    Thresholds
    ----------
    < 0.60  ERROR  — feature is essentially absent, model quality severely impacted
    < 0.80  WARN   — significant data gap, check seeding pipeline
    >= 0.80 OK
    """
    ERROR_THRESH = 0.60
    WARN_THRESH  = 0.80

    SEED_COLS = [
        ("spreadscore",    "Training target — games without this are dropped"),
        ("moneyline",      "Run-line juice — needed for EV computation"),
        ("sp_era",         "Starter ERA — prior season"),
        ("sp_whip",        "Starter WHIP — prior season"),
        ("bp_era",         "Bullpen ERA — prior season"),
        ("bp_ip_game",     "Bullpen IP per game — needed for bp_ip_14d fatigue feature"),
    ]

    title = f"Data Quality Report{' — ' + label if label else ''}"
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"  Total rows: {len(games_df)}")

    rates = {}
    errors = []
    warns  = []

    for col, desc in SEED_COLS:
        if col not in games_df.columns:
            print(f"  {'NOT IN DB':6s}  {col:22s}  {desc}")
            rates[col] = 0.0
            errors.append(col)
            continue
        n    = int(games_df[col].notna().sum())
        rate = n / max(len(games_df), 1)
        rates[col] = rate
        if rate < ERROR_THRESH:
            tag = "ERROR"
            errors.append(col)
        elif rate < WARN_THRESH:
            tag = "WARN "
            warns.append(col)
        else:
            tag = "OK   "
        print(f"  {tag}  {col:22s}: {n:5d}/{len(games_df):5d} ({rate:5.1%})  {desc}")

    # sp_era name-mismatch breakdown
    if "sp_era" in games_df.columns and "sp_name" in games_df.columns:
        missing_era = games_df[games_df["sp_era"].isna()]
        has_name    = missing_era["sp_name"].notna() & (missing_era["sp_name"] != "")
        if has_name.sum() > 0:
            print(f"\n  sp_era name-mismatch detail ({has_name.sum()} rows have sp_name but no ERA match):")
            top = missing_era[has_name]["sp_name"].value_counts().head(5)
            for name, cnt in top.items():
                print(f"    {cnt:4d}x  {name}")
            if len(top) == 5:
                print(f"    ... (run --pitcher-refresh to attempt a fix)")

    if errors:
        print(f"\n  ERRORS ({len(errors)} columns below {ERROR_THRESH:.0%}): {errors}")
        print("  These features will be mostly NaN in the model — check seeding.")
    if warns:
        print(f"  WARNS  ({len(warns)} columns below {WARN_THRESH:.0%}): {warns}")
    if not errors and not warns:
        print(f"\n  All columns OK.")

    print(f"{'=' * 60}\n")
    return rates


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
        if _et_date(g["commence_time"]) in game_date_set
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
            <= datetime.date.fromisoformat(_et_date(g["commence_time"]))
            <= max_date
        ]
    else:
        cutoff = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
        data = [
            g for g in data
            if datetime.datetime.strptime(g["commence_time"], _ODDS_FMT)
            .replace(tzinfo=datetime.timezone.utc) < cutoff
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
        game_date = _et_date(game.get("commence_time", ""))
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        # Prefer major US books in order; fall back to most-markets
        _PREFERRED_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "pointsbet"]
        bm_by_key = {b["key"]: b for b in bookmakers}
        bm = next(
            (bm_by_key[k] for k in _PREFERRED_BOOKS if k in bm_by_key),
            max(bookmakers, key=lambda b: len(b["markets"])),
        )
        markets_by_key = {m["key"]: m for m in bm["markets"]}

        # --- spreads ---
        if "spreads" in markets_by_key:
            for outcome in markets_by_key["spreads"]["outcomes"]:
                t = outcome["name"]
                if t in spreads:
                    continue  # keep first occurrence — avoids doubleheader overwrite
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
# MLB — run lines / moneylines via ESPN API (free, no key required)
# ---------------------------------------------------------------------------
# ESPN's internal API returns pickcenter odds for completed games going back
# to at least 2022.  No authentication or key is required.
#
# Endpoints used:
#   GET sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/seasons/{year}
#       /types/2/events?limit=900&page={n}
#     -> returns event IDs for the full regular season
#
#   GET site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={id}
#     -> returns pickcenter[0]: spread, moneyLine per team, overUnder, etc.
#
# Run lines in MLB are always ±1.5 — the favourite gets -1.5 (must win by 2+).
# We identify the favourite from the team whose moneyLine is more negative.

_ESPN_CORE  = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
_ESPN_SITE  = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
_ESPN_HDRS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ESPN team displayName -> canonical name (edge cases only; most match statsapi)
_ESPN_TEAM_FIX: dict[str, str] = {
    "Cleveland Indians": "Cleveland Guardians",   # renamed 2022
}

# The franchise dropped "Oakland" after the 2024 season.  statsapi — which is
# what the DB `team` field is keyed on — reports "Oakland Athletics" through
# 2024 and plain "Athletics" from 2025, while ESPN says "Athletics" in both
# eras.  Mapping unconditionally to "Oakland Athletics" was correct until 2024
# and silently broke every A's merge from 2025 on (0% run-line fill).
_ATHLETICS_RENAMED_FROM = 2025


def _espn_canonical(name: str, season: int | None = None) -> str:
    """Map an ESPN display name onto the statsapi spelling for that season."""
    if name == "Athletics":
        if season is not None and season < _ATHLETICS_RENAMED_FROM:
            return "Oakland Athletics"
        return "Athletics"
    return _ESPN_TEAM_FIX.get(name, name)


def fetch_espn_event_ids_for_date(date: str) -> list[str]:
    """
    Return ESPN event IDs for a specific date using the scoreboard endpoint.

    Much faster than fetching the full season list — use for nightly incremental
    updates where only 1-2 dates are needed.

    Parameters
    ----------
    date : 'YYYY-MM-DD' string.

    Returns
    -------
    List of event ID strings for that date.
    """
    date_compact = date.replace("-", "")
    url = (f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
           f"?dates={date_compact}&limit=50")
    try:
        r = requests.get(url, headers=_ESPN_HDRS, timeout=15)
        r.raise_for_status()
    except Exception as exc:
        print(f"  fetch_espn_event_ids_for_date {date} failed: {exc}")
        return []

    data = r.json()
    ids = []
    for event in data.get("events", []):
        eid = event.get("id")
        if eid:
            ids.append(str(eid))
    return ids


def fetch_espn_event_ids(season: int) -> list[str]:
    """
    Return all regular-season event IDs for a given MLB season from ESPN.

    Parameters
    ----------
    season : Calendar year (e.g. 2023).

    Returns
    -------
    List of event ID strings (e.g. ['401471020', ...]).
    """
    ids: list[str] = []
    page = 1
    while True:
        url = (f"{_ESPN_CORE}/seasons/{season}/types/2/events"
               f"?limit=900&page={page}&lang=en&region=us")
        try:
            r = requests.get(url, headers=_ESPN_HDRS, timeout=15)
            r.raise_for_status()
        except Exception as exc:
            print(f"  fetch_espn_event_ids page {page} failed: {exc}")
            break

        data = r.json()
        items = data.get("items", [])
        for item in items:
            ref = item.get("$ref", "")
            m = re.search(r"/events/(\d+)", ref)
            if m:
                ids.append(m.group(1))

        if page >= data.get("pageCount", 1):
            break
        page += 1

    return ids


def fetch_espn_game_odds(event_id: str, retries: int = 3) -> dict | None:
    """
    Fetch run line, moneyline, and over/under for one MLB game from ESPN.

    Returns a dict with keys:
        game_date    YYYY-MM-DD
        home_team    canonical team name
        away_team    canonical team name
        home_score   int | None
        away_score   int | None
        run_line_home  float  (-1.5 for home favourite, +1.5 for underdog)
        run_line_away  float
        ml_home      float  (moneyline American odds)
        ml_away      float
        over_under   float | None

    Returns None if no pickcenter data is available.
    Retries with exponential backoff on transient failures (rate limiting etc).
    """
    url = f"{_ESPN_SITE}/summary?event={event_id}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_ESPN_HDRS, timeout=8)
            if r.status_code == 429:
                wait = 10 * (2 ** attempt)   # 10s, 20s, 40s
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return None
    else:
        return None

    # --- Game metadata from header ---
    header = data.get("header", {})
    comp   = header.get("competitions", [{}])[0]
    game_date = _et_date(comp.get("date") or "")

    # Season drives the Athletics naming era — see _espn_canonical.
    _season = int(game_date[:4]) if game_date else None

    home_team = away_team = None
    home_score = away_score = None
    for c in comp.get("competitors", []):
        name = _espn_canonical(c.get("team", {}).get("displayName", ""), _season)
        score_str = c.get("score")
        score = int(score_str) if score_str and str(score_str).isdigit() else None
        if c.get("homeAway") == "home":
            home_team, home_score = name, score
        else:
            away_team, away_score = name, score

    if not home_team or not away_team or not game_date:
        return None

    # --- Odds from pickcenter ---
    pc = data.get("pickcenter", [])
    if not pc:
        # Return date so the caller can still break at the cutoff date.
        return {"game_date": game_date, "no_odds": True}

    p = pc[0]
    home_odds = p.get("homeTeamOdds", {})
    away_odds = p.get("awayTeamOdds", {})

    ml_home = home_odds.get("moneyLine")
    ml_away = away_odds.get("moneyLine")
    home_fav = bool(home_odds.get("favorite", False))

    if ml_home is None or ml_away is None:
        return None

    # Run line assignment: favourite gets -1.5, underdog gets +1.5
    run_line_home = -1.5 if home_fav else +1.5
    run_line_away = +1.5 if home_fav else -1.5

    return {
        "game_date":      game_date,
        "home_team":      home_team,
        "away_team":      away_team,
        "home_score":     home_score,
        "away_score":     away_score,
        "run_line_home":  run_line_home,
        "run_line_away":  run_line_away,
        "ml_home":        float(ml_home),
        "ml_away":        float(ml_away),
        "over_under":     p.get("overUnder"),
    }


def fetch_mlb_odds_espn(
    season: int,
    request_delay: float = 0.25,
    verbose: bool = True,
    completed_dates: set[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch run lines and moneylines for completed regular-season MLB games via ESPN.

    Only requests odds for games that have already been played.  Pass
    `completed_dates` (set of 'YYYY-MM-DD' strings from the statsapi game
    results) to skip future/unplayed games entirely — this cuts API calls by
    ~60% mid-season and avoids wasting time on events with no pickcenter data.

    Parameters
    ----------
    season           : Calendar year (e.g. 2023).
    request_delay    : Seconds between requests (default 0.25 = ~4 req/s).
    verbose          : Print progress every 250 games.
    completed_dates  : Optional set of date strings ('YYYY-MM-DD') for which
                       we have completed game results.  Events outside this set
                       are skipped.  If None, all events are fetched (legacy
                       behaviour for historical back-fills).

    Returns
    -------
    DataFrame with columns: date, team, opponent, home, run_line, moneyline, over_under.
    Returns empty DataFrame on failure.
    """
    import datetime as _dt
    today_str = _dt.date.today().isoformat()

    # Fast path: if only a few specific dates are needed, use the scoreboard
    # endpoint to fetch event IDs per-date rather than pulling the full season list.
    if completed_dates is not None and len(completed_dates) <= 7:
        if verbose:
            print(f"  Fetching ESPN event IDs for {len(completed_dates)} date(s) via scoreboard...")
        event_ids = []
        for d in sorted(completed_dates):
            event_ids.extend(fetch_espn_event_ids_for_date(d))
        if not event_ids:
            print(f"  No event IDs found for the requested dates.")
            return pd.DataFrame()
    else:
        if verbose:
            print(f"  Fetching ESPN event IDs for {season}...")
        event_ids = fetch_espn_event_ids(season)
        if not event_ids:
            print(f"  No event IDs found for {season}.")
            return pd.DataFrame()

    latest_completed = max(completed_dates) if completed_dates else today_str
    filtered_ids = event_ids

    n_total = len(filtered_ids)
    if verbose:
        scope = f"{sorted(completed_dates)}" if completed_dates and len(completed_dates) <= 7 else (f"up to {max(completed_dates)}" if completed_dates else "full season")
        print(f"  {n_total} events — fetching odds ({scope})...")

    records: list[dict] = []
    errors  = 0
    skipped = 0
    latest_completed = max(completed_dates) if completed_dates else today_str
    for i, eid in enumerate(filtered_ids):
        game = fetch_espn_game_odds(eid)
        if game is None:
            errors += 1
            continue

        date = game["game_date"]
        # Stop once we pass the last date we need — ESPN events are chronological.
        if completed_dates is not None and date > latest_completed:
            break
        # No pickcenter odds (future game or no data) — date checked, skip record.
        if game.get("no_odds"):
            continue
        if completed_dates is not None and date not in completed_dates:
            continue
        for team, opp, rl, ml, is_home in (
            (game["home_team"], game["away_team"], game["run_line_home"], game["ml_home"], 1),
            (game["away_team"], game["home_team"], game["run_line_away"], game["ml_away"], 0),
        ):
            records.append({
                "date":       date,
                "team":       team,
                "opponent":   opp,
                "home":       is_home,
                "run_line":   rl,
                "moneyline":  ml,
                "over_under": game["over_under"],
            })

        if verbose and (i + 1) % 250 == 0:
            print(f"    {i+1}/{n_total} events processed  ({errors} no-odds, {skipped} future skipped)")

        time.sleep(request_delay)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if verbose:
        pct = df["run_line"].notna().mean()
        print(f"  Done. {len(df)//2} games, {pct:.1%} with run line data.")
    return df


# ---------------------------------------------------------------------------
# MLB — run-line odds via SBR web scrape (historical, no API key required)
# ---------------------------------------------------------------------------

_SBR_URL = "https://www.sportsbookreview.com/betting-odds/mlb-baseball/pointspread/full-game/?date={date}"
_SBR_HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

_SBR_TEAM_FIX: dict[str, str] = {
    "Cleveland Indians":    "Cleveland Guardians",
    "Oakland Athletics":    "Oakland Athletics",
    "Athletics":            "Oakland Athletics",
    "Tampa Bay Devil Rays": "Tampa Bay Rays",
    "Anaheim Angels":       "Los Angeles Angels",
    "Florida Marlins":      "Miami Marlins",
    "Montreal Expos":       "Washington Nationals",
}

_SBR_BOOK_PREFERENCE = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet", "betonlineag"]


def _sbr_canonical(name: str) -> str:
    return _SBR_TEAM_FIX.get(name, name)


def _sbr_best_line(odds_views: list) -> dict | None:
    """Pick closing line from best available sportsbook. Returns None if no valid line."""
    if not odds_views:
        return None
    views_by_book = {ov["sportsbook"]: ov for ov in odds_views if ov}
    for book in _SBR_BOOK_PREFERENCE:
        if book in views_by_book:
            cl = views_by_book[book].get("currentLine") or {}
            if cl.get("homeSpread") is not None and cl.get("homeOdds") is not None:
                return cl
    for ov in odds_views:
        if not ov:
            continue
        cl = ov.get("currentLine") or {}
        if cl.get("homeSpread") is not None and cl.get("homeOdds") is not None:
            return cl
    return None


def _sbr_best_opening_line(odds_views: list) -> dict | None:
    """Pick opening line from best available sportsbook. Returns None if no valid line."""
    if not odds_views:
        return None
    views_by_book = {ov["sportsbook"]: ov for ov in odds_views if ov}
    for book in _SBR_BOOK_PREFERENCE:
        if book in views_by_book:
            ol = views_by_book[book].get("openingLine") or {}
            if ol.get("homeSpread") is not None and ol.get("homeOdds") is not None:
                return ol
    for ov in odds_views:
        if not ov:
            continue
        ol = ov.get("openingLine") or {}
        if ol.get("homeSpread") is not None and ol.get("homeOdds") is not None:
            return ol
    return None


_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
# Snapshot hour must match when the prediction job runs (cron '0 9 * * *').
# EV has to be priced against odds obtainable at decision time, and a later
# snapshot would encode lineups and scratches that did not exist at 5am.
_ODDS_API_SNAPSHOT_HOUR = "09:00:00Z"
_ODDS_API_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "pointsbet"]


def fetch_mlb_odds_api(
    dates: set[str] | list[str],
    config_path: str = "data/config.txt",
    request_delay: float = 0.3,
    verbose: bool = True,
    max_dates: int | None = 5,
) -> pd.DataFrame:
    """
    Fetch observed run lines and both market prices from the paid odds-api.

    Unlike ESPN — which exposes no run-line price at all, forcing the line to be
    synthesised from a `favorite` boolean and the H2H price to be stored in its
    place — this returns the real line plus both markets for the same game.

    Returns a DataFrame with columns:
        date, team, run_line, spread_juice, ml_odds, snapshot_ts

    Costs 20 credits per date (h2h + spreads, us region).

    max_dates caps a single call and keeps the NEWEST dates, so a nightly run
    can never sweep a whole season. A full 2026 season is ~130 dates = 2,600
    credits, enough to exhaust a month's allowance in one command. Pass
    max_dates=None deliberately for a backfill.
    """
    cfg = _read_config(config_path)
    key = cfg["spreads"]["key_paid"]

    dates = sorted(dates)
    if max_dates is not None and len(dates) > max_dates:
        print(f"  odds-api: {len(dates)} dates requested but max_dates={max_dates}; "
              f"keeping the {max_dates} most recent "
              f"({dates[-max_dates]}..{dates[-1]}). "
              f"Pass max_dates=None for an intentional backfill.")
        dates = dates[-max_dates:]

    def _has_spreads(bm: dict) -> bool:
        return any(m.get("key") == "spreads" for m in bm.get("markets", []))

    records: list[dict] = []
    for i, date in enumerate(dates, 1):
        try:
            r = requests.get(
                f"{_ODDS_API_BASE}/historical/sports/baseball_mlb/odds",
                params={"apiKey": key, "regions": "us", "markets": "h2h,spreads",
                        "oddsFormat": "american",
                        "date": f"{date}T{_ODDS_API_SNAPSHOT_HOUR}"},
                timeout=30,
            )
        except Exception as exc:
            print(f"  odds-api {date} request failed: {exc}")
            continue
        if r.status_code != 200:
            print(f"  odds-api {date}: HTTP {r.status_code} {r.text[:120]}")
            time.sleep(request_delay)
            continue
        if verbose and i == 1:
            print(f"  odds-api credits remaining: "
                  f"{r.headers.get('x-requests-remaining', '?')}")

        payload = r.json()
        snap_ts = payload.get("timestamp")
        # Sort by start time so game 1 of a doubleheader precedes game 2.
        board = sorted(payload.get("data", []),
                       key=lambda g: g.get("commence_time") or "")

        for g in board:
            # Only keep games whose Eastern date matches the date requested;
            # a 5am board also lists upcoming days.
            if _et_date(g.get("commence_time", "")) != date:
                continue
            bms = g.get("bookmakers", [])
            bm = next((b for want in _ODDS_API_BOOKS for b in bms
                       if b.get("key") == want and _has_spreads(b)), None)
            if bm is None:
                bm = next((b for b in bms if _has_spreads(b)), None)
            if bm is None:
                continue

            markets = {m["key"]: m for m in bm.get("markets", [])}
            spreads = markets.get("spreads")
            if not spreads:
                continue
            h2h = {o["name"]: o.get("price")
                   for o in markets.get("h2h", {}).get("outcomes", [])}

            for o in spreads.get("outcomes", []):
                if o.get("point") is None or o.get("price") is None:
                    continue
                records.append({
                    "date":         date,
                    "team":         o["name"],
                    "run_line":     float(o["point"]),
                    "spread_juice": float(o["price"]),
                    "ml_odds":      (float(h2h[o["name"]])
                                     if h2h.get(o["name"]) is not None else None),
                    "snapshot_ts":  snap_ts,
                })
        time.sleep(request_delay)

    df = pd.DataFrame(records)
    if verbose:
        print(f"  odds-api returned {len(df)} team-rows across "
              f"{df['date'].nunique() if not df.empty else 0} date(s).")
    return df


def fetch_mlb_odds_sbr_web(
    date: str,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Scrape MLB run-line closing odds for one date from sportsbookreview.com.

    Parameters
    ----------
    date : 'YYYY-MM-DD'

    Returns
    -------
    DataFrame with columns: date, team, opponent, home, run_line, moneyline.
    Only rows where run_line is not None are returned.
    """
    import re as _re
    url = _SBR_URL.format(date=date)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_SBR_HDRS, timeout=15)
            r.raise_for_status()
            break
        except Exception:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return pd.DataFrame()

    m = _re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', r.text, _re.DOTALL)
    if not m:
        return pd.DataFrame()

    try:
        data = json.loads(m.group(1))
        game_rows = (
            data["props"]["pageProps"]["oddsTables"][0]["oddsTableModel"]["gameRows"]
        )
    except (KeyError, IndexError, json.JSONDecodeError):
        return pd.DataFrame()

    records = []
    for g in game_rows:
        gv = g.get("gameView", {})
        home_name = _sbr_canonical(gv.get("homeTeam", {}).get("fullName", ""))
        away_name = _sbr_canonical(gv.get("awayTeam", {}).get("fullName", ""))
        if not home_name or not away_name:
            continue

        odds_views = g.get("oddsViews") or []
        cl = _sbr_best_line(odds_views)
        if cl is None:
            continue

        home_spread = cl.get("homeSpread")
        home_odds   = cl.get("homeOdds")
        away_odds   = cl.get("awayOdds")

        if home_spread is None or home_odds is None or away_odds is None:
            continue

        # Opening line from same sportsbook preference
        ol = _sbr_best_opening_line(odds_views)
        open_home_spread = ol.get("homeSpread") if ol else None
        open_home_odds   = ol.get("homeOdds")   if ol else None
        open_away_odds   = ol.get("awayOdds")   if ol else None

        home_rec = {
            "date":      date,
            "team":      home_name,
            "opponent":  away_name,
            "home":      1,
            "run_line":  float(home_spread),
            "moneyline": float(home_odds),
        }
        away_rec = {
            "date":      date,
            "team":      away_name,
            "opponent":  home_name,
            "home":      0,
            "run_line":  float(-home_spread),
            "moneyline": float(away_odds),
        }
        if open_home_spread is not None and open_home_odds is not None and open_away_odds is not None:
            home_rec["open_spread"]    = float(open_home_spread)
            home_rec["open_moneyline"] = float(open_home_odds)
            away_rec["open_spread"]    = float(-open_home_spread)
            away_rec["open_moneyline"] = float(open_away_odds)

        records.extend([home_rec, away_rec])

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


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


def fetch_season_games_mlb(
    season: int,
    since: str | None = None,
) -> list[dict]:
    """
    Fetch regular-season MLB game results for a calendar year via statsapi.

    Returns a list of raw statsapi game dicts (only completed regular-season
    games with scores). Spring training and playoffs are excluded.

    Parameters
    ----------
    season : Calendar year (e.g. 2023 for the 2023 MLB season).
    since  : Optional 'YYYY-MM-DD' lower bound. When provided only games on or
             after this date are fetched — use for nightly incremental updates
             instead of re-pulling the full season each run.
    """
    try:
        import statsapi
    except ImportError:
        raise ImportError("MLB-StatsAPI not installed. Run: pip install MLB-StatsAPI")

    # Regular season spans late March / early April through late September.
    season_start = f"03/15/{season}"
    season_end   = f"10/10/{season}"

    if since is not None:
        import datetime as _dt
        since_dt = _dt.date.fromisoformat(since)
        start = since_dt.strftime("%m/%d/%Y")
    else:
        start = season_start

    import concurrent.futures as _cf
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(statsapi.schedule,
                              start_date=start, end_date=season_end, sportId=1)
            raw = _fut.result(timeout=60)
    except _cf.TimeoutError:
        raise RuntimeError(
            "statsapi.schedule timed out after 60s — MLB Stats API may be down."
        )
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
    period_offsets: dict[str, int] | None = None,
    known_periods: dict[tuple[str, str], int] | None = None,
) -> pd.DataFrame:
    """
    Convert raw statsapi game dicts to the standard long-format DataFrame.

    One row per team per game with columns:
        sport, team, opponent, season, period, date,
        score, opp_score, diff, home,
        sp_name  (probable starter name, may be empty string)

    period is assigned as sequential game number per team ordered by date,
    matching the NBA convention (1–162).

    Parameters
    ----------
    period_offsets : optional dict mapping team name -> current max period in DB.
        Sets the starting point for games not already in `known_periods`.
    known_periods : optional dict mapping (team, game_pk) -> period already
        stored in the DB.  Pass this on every incremental (--since) seed: it
        makes period assignment idempotent, so re-fetching a game that is
        already seeded reuses its period instead of minting a duplicate row
        under a new one.  Omit for a full-season reseed, which numbers every
        game from 1.
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

        game_pk = str(g.get("game_id", ""))   # statsapi game ID for boxscore lookup

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
                "game_pk":   game_pk,       # for boxscore / bullpen IP lookup
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

    df = df.sort_values("date")

    if known_periods:
        # Identity-keyed assignment.  A --since run always re-fetches games
        # that are already seeded (the since-date itself overlaps), and giving
        # those a fresh cumcount+offset filed the same game_pk under a second
        # period — 926 duplicate rows accumulated this way before it was
        # caught.  Reuse the stored period for any game already known and only
        # allocate new numbers for genuinely new games.
        next_by_team = dict(period_offsets or {})
        periods = []
        for team, pk in zip(df["team"], df["game_pk"]):
            existing = known_periods.get((team, str(pk)))
            if existing is not None:
                periods.append(int(existing))
            else:
                nxt = int(next_by_team.get(team, 0)) + 1
                next_by_team[team] = nxt
                periods.append(nxt)
        df["period"] = periods
    else:
        # Full-season seed: number every game from 1 in date order.
        df["period"] = df.groupby("team").cumcount() + 1
        if period_offsets:
            df["period"] = df["period"] + (
                df["team"].map(period_offsets).fillna(0).astype(int)
            )

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
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(statsapi.get, "stats", {
                "stats":       "season",
                "group":       "pitching",
                "sportId":     1,
                "season":      season,
                "gameType":    "R",
                "limit":       1000,
                "playerPool":  "ALL",
            })
            result = _fut.result(timeout=30)
    except Exception as exc:
        print(f"  fetch_mlb_pitcher_stats failed ({exc})")
        return pd.DataFrame()

    splits = result.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return pd.DataFrame()

    def _parse_ip(ip_str: str) -> float:
        """Convert '180.2' (180 and 2/3 innings) to a float."""
        s = str(ip_str or "0")
        try:
            if "." in s:
                whole, frac = s.split(".", 1)
                return int(whole) + int(frac) / 3.0
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    rows = []
    for s in splits:
        stat = s.get("stat", {})
        player = s.get("player", {})
        name = player.get("fullName", "")
        if not name:
            continue
        try:
            gs = int(stat.get("gamesStarted", 0) or 0)
            ip = _parse_ip(stat.get("inningsPitched", "0"))
            rows.append({
                "name":  name,
                "era":   float(stat.get("era", "nan") or "nan"),
                "whip":  float(stat.get("whip", "nan") or "nan"),
                "k9":    float(stat.get("strikeoutsPer9Inn", "nan") or "nan"),
                "bb9":   float(stat.get("walksPer9Inn", "nan") or "nan"),
                "gs":    gs,
                "ip":    ip,
            })
        except (ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Normalize names to handle accent/encoding mismatches between statsapi
    # schedule (which returns "Eury Pérez") and stats endpoint.
    # Keep original name for display; use normalized key as index.
    df["name_key"] = df["name"].apply(_normalize_pitcher_name)
    df = df.set_index("name_key")

    # Minimum IP filter — pitchers with very few innings have unreliable ERAs
    # (e.g. 4 ER in 1.1 IP = 27.00 ERA). Require at least 10 IP for a
    # meaningful sample. This keeps high-usage relievers who transition to
    # starting roles while dropping one-game cup-of-coffee appearances.
    df = df[df["ip"] >= 10].copy()

    # ip_per_start is NaN for pure relievers (gs == 0) which is fine.
    df["ip_per_start"] = (df["ip"] / df["gs"]).where(df["gs"] > 0)
    df = df.drop(columns=["gs", "ip", "name"])
    return df


def fetch_mlb_bullpen_stats(season: int) -> pd.DataFrame:
    """
    Fetch team-level bullpen stats for a season via statsapi.

    Aggregates individual reliever stats (gamesStarted == 0) to the team
    level, weighted by innings pitched.  Use season-1 when building features
    to avoid leakage.

    Parameters
    ----------
    season : The season to pull stats from.

    Returns
    -------
    DataFrame indexed by team name with columns:
        bp_era, bp_whip, bp_k9, bp_hr9

    Returns empty DataFrame on failure.
    """
    try:
        import statsapi
    except ImportError:
        print("  MLB-StatsAPI not installed.")
        return pd.DataFrame()

    try:
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(statsapi.get, "stats", {
                "stats":      "season",
                "group":      "pitching",
                "sportId":    1,
                "season":     season,
                "gameType":   "R",
                "limit":      2000,
                "playerPool": "ALL",
            })
            raw = _fut.result(timeout=30)
    except Exception as exc:
        print(f"  fetch_mlb_bullpen_stats failed ({exc})")
        return pd.DataFrame()

    splits = raw.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return pd.DataFrame()

    def _parse_ip(ip_str: str) -> float:
        """Convert '45.2' (45 and 2/3 innings) to a float."""
        ip_str = str(ip_str or "0")
        try:
            if "." in ip_str:
                whole, frac = ip_str.split(".", 1)
                return int(whole) + int(frac) / 3.0
            return float(ip_str)
        except (ValueError, TypeError):
            return 0.0

    rows = []
    for s in splits:
        stat = s.get("stat", {})
        team = s.get("team", {}).get("name", "")
        if not team:
            continue
        gs = int(stat.get("gamesStarted", 0) or 0)
        if gs > 0:
            continue  # starters handled separately
        ip = _parse_ip(stat.get("inningsPitched", "0"))
        if ip < 5:
            continue  # ignore cup-of-coffee appearances
        try:
            rows.append({
                "team": team,
                "ip":   ip,
                "era":  float(stat.get("era",  "nan") or "nan"),
                "whip": float(stat.get("whip", "nan") or "nan"),
                "k9":   float(stat.get("strikeoutsPer9Inn", "nan") or "nan"),
                "hr9":  float(stat.get("homeRunsPer9",      "nan") or "nan"),
            })
        except (ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Aggregate to team level, innings-pitched weighted
    def _wmean(grp: pd.DataFrame, col: str) -> float:
        w = grp["ip"]
        v = grp[col]
        valid = v.notna() & w.notna()
        denom = w[valid].sum()
        if denom == 0:
            return float("nan")
        return float((v[valid] * w[valid]).sum() / denom)

    # MLB teams play 162 games — use this to normalise total bullpen IP
    _GAMES_PER_SEASON = 162.0

    team_rows = []
    for team, grp in df.groupby("team"):
        total_bp_ip = grp["ip"].sum()
        team_rows.append({
            "team":           team,
            "bp_era":         _wmean(grp, "era"),
            "bp_whip":        _wmean(grp, "whip"),
            "bp_k9":          _wmean(grp, "k9"),
            "bp_hr9":         _wmean(grp, "hr9"),
            # Avg innings per game the bullpen pitches — high = heavily used pen
            "bp_ip_per_game": round(total_bp_ip / _GAMES_PER_SEASON, 3),
        })

    return pd.DataFrame(team_rows).set_index("team")


def fetch_game_bp_ip(game_pk: str) -> dict[str, float]:
    """
    Fetch per-team bullpen innings pitched for a single completed game.

    Uses statsapi.boxscore_data(game_pk) — free, no key required.

    Parameters
    ----------
    game_pk : statsapi game ID (stored as 'game_pk' in MongoDB game docs).

    Returns
    -------
    Dict mapping team name → bullpen IP for that game.
    e.g. {'New York Yankees': 3.667, 'Boston Red Sox': 2.0}
    Returns empty dict on failure.
    """
    try:
        import statsapi
    except ImportError:
        return {}

    try:
        box = statsapi.boxscore_data(int(game_pk))
    except Exception:
        return {}

    def _parse_ip(ip_str) -> float:
        """Convert statsapi IP string (e.g. '4.1' = 4⅓) to decimal innings."""
        s = str(ip_str or "0").strip()
        if not s or s in ("IP", "-", ""):
            return 0.0
        try:
            if "." in s:
                whole, frac = s.split(".", 1)
                return int(whole) + int(frac) / 3.0
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    # Abbreviation → full team name for the current statsapi response format.
    # The old format used homeTeamStats.teamInfo.team.name (full name).
    # The new format uses teamInfo.home.abbreviation + teamName.
    _ABBREV_TO_FULL: dict[str, str] = {
        "AZ":  "Arizona Diamondbacks",   "ATL": "Atlanta Braves",
        "BAL": "Baltimore Orioles",       "BOS": "Boston Red Sox",
        "CHC": "Chicago Cubs",            "CWS": "Chicago White Sox",
        "CIN": "Cincinnati Reds",         "CLE": "Cleveland Guardians",
        "COL": "Colorado Rockies",        "DET": "Detroit Tigers",
        "HOU": "Houston Astros",          "KC":  "Kansas City Royals",
        "LAA": "Los Angeles Angels",      "LAD": "Los Angeles Dodgers",
        "MIA": "Miami Marlins",           "MIL": "Milwaukee Brewers",
        "MIN": "Minnesota Twins",         "NYM": "New York Mets",
        "NYY": "New York Yankees",        "ATH": "Athletics",
        "OAK": "Athletics",               "PHI": "Philadelphia Phillies",
        "PIT": "Pittsburgh Pirates",      "SD":  "San Diego Padres",
        "SF":  "San Francisco Giants",    "SEA": "Seattle Mariners",
        "STL": "St. Louis Cardinals",     "TB":  "Tampa Bay Rays",
        "TEX": "Texas Rangers",           "TOR": "Toronto Blue Jays",
        "WSH": "Washington Nationals",    "WAS": "Washington Nationals",
    }

    def _resolve_team_name(side: str) -> str:
        """Try new teamInfo format first, fall back to old homeTeamStats format."""
        # New format: box['teamInfo']['home']['abbreviation']
        abbrev = box.get("teamInfo", {}).get(side, {}).get("abbreviation", "")
        if abbrev and abbrev in _ABBREV_TO_FULL:
            return _ABBREV_TO_FULL[abbrev]
        # Old format: box['homeTeamStats']['teamInfo']['team']['name']
        old_name = (
            box.get(f"{side}TeamStats", {})
               .get("teamInfo", {})
               .get("team", {})
               .get("name", "")
        )
        return _MLB_NAME_FIXES.get(old_name, old_name)

    result: dict[str, float] = {}
    for side in ("home", "away"):
        team_name = _resolve_team_name(side)
        pitchers  = box.get(f"{side}Pitchers", [])

        # Filter to real pitcher rows (skip the header row where personId=0)
        real_pitchers = [p for p in pitchers if p.get("personId", 0) != 0]

        bp_ip = 0.0
        for i, p in enumerate(real_pitchers):
            # IP key changed from 'inningsPitched' to 'ip' in newer statsapi versions
            ip_val  = p.get("ip") or p.get("inningsPitched") or "0"
            ip      = _parse_ip(ip_val)

            # Identify starter: first pitcher listed who threw > 2 IP.
            # Opener strategy: if first pitcher threw ≤ 2 IP, they are a relief
            # opener — do NOT skip them; the true starter follows.
            if i == 0 and ip > 2.0:
                continue   # skip starter

            bp_ip += ip

        if team_name:
            result[team_name] = round(bp_ip, 3)

    return result


def fetch_game_sp_stats(game_pk: str) -> dict[str, dict]:
    """
    Fetch starting pitcher stats for a single completed game.

    Uses the same boxscore_data call as fetch_game_bp_ip — no extra API cost
    if called together.  Identifies the starter using the same heuristic:
    first pitcher who threw > 2 IP (opener rule: ≤ 2 IP = relief opener,
    true starter follows).

    Parameters
    ----------
    game_pk : statsapi game ID.

    Returns
    -------
    Dict mapping team name → {'sp_name', 'sp_ip_game', 'sp_er_game',
    'sp_k_game', 'sp_bb_game', 'sp_h_game', 'sp_hr_game', 'sp_pitch_game'}
    Returns empty dict on failure.
    """
    try:
        import statsapi
    except ImportError:
        return {}

    try:
        box = statsapi.boxscore_data(int(game_pk))
    except Exception:
        return {}

    def _parse_ip(ip_str) -> float:
        s = str(ip_str or "0").strip()
        if not s or s in ("IP", "-", ""):
            return 0.0
        try:
            if "." in s:
                whole, frac = s.split(".", 1)
                return int(whole) + int(frac) / 3.0
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    _ABBREV_TO_FULL: dict[str, str] = {
        "AZ":  "Arizona Diamondbacks",   "ATL": "Atlanta Braves",
        "BAL": "Baltimore Orioles",       "BOS": "Boston Red Sox",
        "CHC": "Chicago Cubs",            "CWS": "Chicago White Sox",
        "CIN": "Cincinnati Reds",         "CLE": "Cleveland Guardians",
        "COL": "Colorado Rockies",        "DET": "Detroit Tigers",
        "HOU": "Houston Astros",          "KC":  "Kansas City Royals",
        "LAA": "Los Angeles Angels",      "LAD": "Los Angeles Dodgers",
        "MIA": "Miami Marlins",           "MIL": "Milwaukee Brewers",
        "MIN": "Minnesota Twins",         "NYM": "New York Mets",
        "NYY": "New York Yankees",        "ATH": "Athletics",
        "OAK": "Athletics",               "PHI": "Philadelphia Phillies",
        "PIT": "Pittsburgh Pirates",      "SD":  "San Diego Padres",
        "SF":  "San Francisco Giants",    "SEA": "Seattle Mariners",
        "STL": "St. Louis Cardinals",     "TB":  "Tampa Bay Rays",
        "TEX": "Texas Rangers",           "TOR": "Toronto Blue Jays",
        "WSH": "Washington Nationals",    "WAS": "Washington Nationals",
    }

    def _resolve_team(side: str) -> str:
        abbrev = box.get("teamInfo", {}).get(side, {}).get("abbreviation", "")
        if abbrev and abbrev in _ABBREV_TO_FULL:
            return _ABBREV_TO_FULL[abbrev]
        old_name = (
            box.get(f"{side}TeamStats", {})
               .get("teamInfo", {}).get("team", {}).get("name", "")
        )
        return _MLB_NAME_FIXES.get(old_name, old_name)

    result: dict[str, dict] = {}
    for side in ("home", "away"):
        team_name = _resolve_team(side)
        pitchers  = box.get(f"{side}Pitchers", [])
        real      = [p for p in pitchers if p.get("personId", 0) != 0]

        # Find the starter: first pitcher who threw > 2 IP.
        # If first threw ≤ 2 IP (opener), skip them and take the next.
        starter = None
        for i, p in enumerate(real):
            ip_val = p.get("ip") or p.get("inningsPitched") or "0"
            ip = _parse_ip(ip_val)
            if i == 0 and ip <= 2.0:
                continue  # opener — skip, true starter follows
            starter = p
            break

        if starter and team_name:
            ip_val  = starter.get("ip") or starter.get("inningsPitched") or "0"
            er_val  = starter.get("er") or starter.get("earnedRuns") or 0

            # Always resolve full name from the players dict — the pitcher
            # row's 'name' field is abbreviated (last name only).
            pid = starter.get("personId", 0)
            sp_name = (
                box.get(f"{side}", {})
                   .get("players", {})
                   .get(f"ID{pid}", {})
                   .get("person", {})
                   .get("fullName", "")
                or starter.get("name", "")   # fallback to abbreviated name
            )

            def _int(v) -> int:
                try:
                    return int(str(v).strip())
                except (TypeError, ValueError):
                    return 0

            # k / bb / h / hr sit on the same summary row already being read, so
            # these cost nothing beyond four dictionary lookups. They are what
            # lets K/9, BB/9, WHIP and FIP be blended in-season instead of being
            # frozen at their prior-season values.
            result[team_name] = {
                "sp_name":       sp_name,
                "sp_ip_game":    round(_parse_ip(ip_val), 3),
                "sp_er_game":    int(er_val) if er_val is not None else 0,
                "sp_k_game":     _int(starter.get("k")),
                "sp_bb_game":    _int(starter.get("bb")),
                "sp_h_game":     _int(starter.get("h")),
                "sp_hr_game":    _int(starter.get("hr")),
                "sp_pitch_game": _int(starter.get("p")),
            }

    return result


def fetch_season_sp_stats(
    season: int,
    game_pks: list[str],
    request_delay: float = 0.25,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch per-game starting pitcher stats for all games in a season.

    Returns
    -------
    DataFrame with columns: game_pk, team, sp_name, sp_ip_game, sp_er_game
    """
    rows = []
    errors = 0
    for i, pk in enumerate(game_pks):
        result = fetch_game_sp_stats(pk)
        if result:
            for team, stats in result.items():
                rows.append({"game_pk": pk, "team": team, **stats})
        else:
            errors += 1
        if verbose and (i + 1) % 250 == 0:
            print(f"    {i+1}/{len(game_pks)} games processed ({errors} errors)")
        time.sleep(request_delay)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_season_bp_ip(
    season: int,
    game_pks: list[str],
    request_delay: float = 0.25,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch per-game bullpen IP for all games in a season.

    Parameters
    ----------
    season       : Season year (for display only).
    game_pks     : List of game_pk strings (from MongoDB game docs).
    request_delay: Seconds between statsapi calls (default 0.25).
    verbose      : Print progress every 250 games.

    Returns
    -------
    DataFrame with columns: game_pk, team, bp_ip_game
    """
    rows = []
    errors = 0
    for i, pk in enumerate(game_pks):
        result = fetch_game_bp_ip(pk)
        if result:
            for team, ip in result.items():
                rows.append({"game_pk": pk, "team": team, "bp_ip_game": ip})
        else:
            errors += 1
        if verbose and (i + 1) % 250 == 0:
            print(f"    {i+1}/{len(game_pks)} games processed ({errors} errors)")
        time.sleep(request_delay)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


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
