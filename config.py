"""
Sport configurations for the betting prediction pipeline.

To add a new sport:
  1. Add a SportConfig instance below.
  2. Verify api_sports_host and api_sports_league against api-sports.io docs.
  3. Verify odds_api_sport against the-odds-api.com /v4/sports endpoint.
  4. Choose a validation_season (held-out year for model evaluation).
  5. Register it in the SPORTS dict.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SportConfig:
    """All sport-specific constants in one place."""

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

    # Regular season date bounds as (month, day), used to exclude preseason
    # and playoff games when the API provides no stage field to filter on.
    # For cross-year seasons (NBA), end is relative to season_year + 1.
    # Set to None for sports where stage-based filtering is sufficient (NFL).
    regular_season_start: tuple[int, int] | None = None
    regular_season_end: tuple[int, int] | None = None

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
    # Regular season historically runs mid-Oct through mid-Apr.
    # Oct 16 safely excludes preseason; Apr 15 safely excludes playoffs.
    regular_season_start=(10, 16),
    regular_season_end=(4, 15),
)

SPORTS: dict[str, SportConfig] = {
    "nfl": NFL,
    "nba": NBA,
}
