"""
Sport configurations for the betting prediction pipeline.

To add a new sport:
  1. Add a SportConfig instance below.
  2. Verify api_sports_host and api_sports_league against api-sports.io docs.
     Set both to None for sports that use an alternative scores source (e.g.
     MLB uses statsapi instead of api-sports.io).
  3. Verify odds_api_sport against the-odds-api.com /v4/sports endpoint.
  4. Choose an eval_season (held-out year) and eval_split_period (period that
     divides the eval season into test [before] and val [from]).
  5. Register it in the SPORTS dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SportConfig:
    """All sport-specific constants in one place."""

    # Short identifier stored as the 'sport' field in MongoDB.
    name: str
    # api-sports.io hostname. None for sports using an alternative scores source.
    api_sports_host: str | None
    # League ID on api-sports.io. None for sports using an alternative scores source.
    api_sports_league: int | None
    # Season string format: "{year}" for NFL/MLB, "{year}-{year+1}" for NBA.
    api_sports_season_fmt: str
    # Sport key for the-odds-api.com.
    odds_api_sport: str
    # Periods in a full regular season (weeks for NFL, games for NBA).
    season_periods: int
    # Season held out entirely from training (split into test + val).
    eval_season: int
    # Period boundary within eval_season: target periods < this → test set,
    # target periods >= this → val set. Roughly the season midpoint.
    eval_split_period: int

    # Regular season date bounds as (month, day), used to exclude preseason
    # and playoff games when the API provides no stage field to filter on.
    # For cross-year seasons (NBA), end is relative to season_year + 1.
    # Set to None for sports where stage-based filtering is sufficient (NFL).
    regular_season_start: tuple[int, int] | None = None
    regular_season_end: tuple[int, int] | None = None

    # If set, only games where BOTH teams are in this set are kept.
    # Used to exclude All-Star / celebrity games that share stage=None with
    # regular-season games and cannot be filtered by stage or date alone.
    known_teams: frozenset | None = None

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
    eval_season=2022,
    eval_split_period=10,  # weeks 1-9 → test, weeks 10-18 → val
)

NBA = SportConfig(
    name="nba",
    api_sports_host="v1.basketball.api-sports.io",
    api_sports_league=12,
    api_sports_season_fmt="{year}-{year+1}",
    odds_api_sport="basketball_nba",
    season_periods=82,
    eval_season=2023,
    eval_split_period=42,  # games 1-41 → test, games 42-82 → val
    # Regular season historically runs mid-Oct through mid-Apr.
    # Oct 16 safely excludes preseason; Apr 15 safely excludes playoffs.
    regular_season_start=(10, 16),
    regular_season_end=(4, 15),
    # All 30 franchises. Games involving All-Star/celebrity teams (stage=None,
    # same as regular season) are filtered out by this allowlist.
    known_teams=frozenset({
        "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets",
        "Charlotte Hornets", "Chicago Bulls", "Cleveland Cavaliers",
        "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons",
        "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
        "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
        "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves",
        "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder",
        "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
        "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs",
        "Toronto Raptors", "Utah Jazz", "Washington Wizards",
    }),
)

# ---------------------------------------------------------------------------
# MLB
# ---------------------------------------------------------------------------
# Run lines in baseball are fixed at ±1.5 (juice varies).
# SpreadScore = run_diff + run_line (same convention as NBA/NFL).
# Scores fetched from the free statsapi (stats.mlb.com) — no api-sports.io key
# needed. Historical run lines loaded from local SBR CSV files.
#
# Season year = calendar year (e.g. 2023 = the 2023 regular season).
# Period = sequential game number per team (1-162).
# Training starts from 2021 (post-COVID). 2020 season was 60 games with
# temporary rule changes (universal DH, 7-inning doubleheaders) that make it
# a weak and potentially misleading training signal.
#
# Key features not in NBA:
#   sp_era, sp_whip, sp_k9  — starting pitcher quality (prior-season stats)
#   opp_sp_era, opp_sp_whip — opponent pitcher
#   sp_era_edge             — self.sp_era - opp.sp_era  (matchup)
MLB = SportConfig(
    name="mlb",
    api_sports_host=None,         # uses statsapi, not api-sports.io
    api_sports_league=None,
    api_sports_season_fmt="{year}",
    odds_api_sport="baseball_mlb",
    season_periods=162,
    eval_season=2023,
    eval_split_period=81,          # games 1-80 -> test, 81-162 -> val
    # Opening Day has been as early as March 20 in recent years; World Series
    # can extend to early November but regular season ends late September.
    regular_season_start=(3, 20),
    regular_season_end=(10, 1),
    known_teams=frozenset({
        "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles",
        "Boston Red Sox", "Chicago Cubs", "Chicago White Sox",
        "Cincinnati Reds", "Cleveland Guardians", "Colorado Rockies",
        "Detroit Tigers", "Houston Astros", "Kansas City Royals",
        "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins",
        "Milwaukee Brewers", "Minnesota Twins", "New York Mets",
        "New York Yankees", "Oakland Athletics", "Philadelphia Phillies",
        "Pittsburgh Pirates", "San Diego Padres", "San Francisco Giants",
        "Seattle Mariners", "St. Louis Cardinals", "Tampa Bay Rays",
        "Texas Rangers", "Toronto Blue Jays", "Washington Nationals",
        # 2022 rename: Cleveland Indians -> Cleveland Guardians (both kept)
        "Cleveland Indians",
        # Athletics played in Oakland through 2024 before relocating
        "Athletics",
    }),
)

SPORTS: dict[str, SportConfig] = {
    "nfl": NFL,
    "nba": NBA,
    "mlb": MLB,
}
