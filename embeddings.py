"""
Playstyle embedding sub-model.

Captures team identity through *how* they play rather than *who* they are.

Approach
--------
Build a (n_teams x n_teams) matchup matrix where cell [i, j] is the average
point differential when team i plays team j. Factorise via Truncated SVD into
two (n_teams x k) matrices:

    Offense[i] : how team i imposes its style on opponents
    Defense[j] : how team j absorbs opponents' styles

For any matchup (Team A vs Team B):

    style_edge = (Offense[A] @ Defense[B]) - (Offense[B] @ Defense[A])

Positive style_edge means Team A's style systematically exploits Team B's
weaknesses based on historical matchup data. This is fed as a single scalar
feature into the main XGBoost model.

Leakage prevention
------------------
Embeddings are always fit on training data only (non-eval seasons). The eval
season teams still appear in the embedding because they played in training
seasons — we just don't use their eval-season outcomes to fit the vectors.

Public interface
----------------
fit(games_df)                          -> StyleModel
StyleModel.transform(games_df)         -> pd.Series  (style_edge per row)
StyleModel.predict_edge(team, opponent)-> float
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.decomposition import TruncatedSVD


@dataclass
class StyleModel:
    """Fitted playstyle embedding model."""
    offense: np.ndarray       # (n_teams, k)
    defense: np.ndarray       # (n_teams, k)
    team_index: dict[str, int]
    edge_clip: float = np.inf  # winsorization bound (3σ from training data)

    def predict_edge(self, team: str, opponent: str) -> float:
        """
        Stylistic edge of `team` over `opponent`.
        Positive = team's style exploits opponent's weaknesses.
        Returns 0.0 for unknown teams.
        Clipped to ±edge_clip so prediction-time values match training scale.
        """
        ti = self.team_index.get(team, -1)
        oi = self.team_index.get(opponent, -1)
        if ti == -1 or oi == -1:
            return 0.0
        raw = float(
            self.offense[ti] @ self.defense[oi] -
            self.offense[oi] @ self.defense[ti]
        )
        return float(np.clip(raw, -self.edge_clip, self.edge_clip))

    def transform(self, games_df: pd.DataFrame) -> pd.Series:
        """
        Compute style_edge for every row in games_df.
        Requires columns: team, opponent.
        Returns a Series aligned to games_df.index.
        """
        return games_df.apply(
            lambda r: self.predict_edge(r["team"], r["opponent"]), axis=1
        )


def fit(games_df: pd.DataFrame, k: int = 3, verbose: bool = False) -> StyleModel | None:
    """
    Fit playstyle embeddings from game data.

    Parameters
    ----------
    games_df : DataFrame with columns team, opponent, diff.
               Should contain only games that have already occurred
               (period < target) for the season being modelled.
    k        : number of latent dimensions (2–4 recommended)
    verbose  : print variance explained

    Returns
    -------
    Fitted StyleModel, or None if insufficient data.
    """
    completed = games_df.dropna(subset=["diff", "opponent"]).copy()
    if len(completed) < 10:
        return None

    # Average point differential per (team, opponent) matchup
    matchup = (
        completed.groupby(["team", "opponent"])["diff"]
        .mean()
        .unstack(fill_value=0.0)
    )

    # Square matrix — ensure all teams appear in both rows and columns
    all_teams = sorted(set(matchup.index) | set(matchup.columns))
    matchup = matchup.reindex(index=all_teams, columns=all_teams, fill_value=0.0)

    n_teams = len(all_teams)
    k = min(k, n_teams - 1)

    M = matchup.values.astype(float)

    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(M)
    V = svd.components_.T
    S = np.sqrt(svd.singular_values_)

    offense = U * S
    defense = V * S

    team_index = {team: i for i, team in enumerate(all_teams)}

    if verbose:
        print(f"  StyleModel: {n_teams} teams, k={k}, "
              f"variance explained={svd.explained_variance_ratio_.sum():.1%}")

    return StyleModel(offense=offense, defense=defense, team_index=team_index)


def build_period_models(
    games_df: pd.DataFrame,
    seasons: list[int],
    max_period: int,
    k: int = 3,
) -> dict[tuple[int, int], "StyleModel | None"]:
    """
    Precompute one StyleModel per (season, target_period), using only
    completed games from that season with period < target_period.

    This ensures no future information leaks into the embedding — early
    periods have weak signal, later periods have richer matchup data.

    Returns
    -------
    dict mapping (season, target_period) -> StyleModel or None
    """
    models: dict[tuple[int, int], StyleModel | None] = {}
    for season in seasons:
        season_games = games_df[games_df["season"] == season]
        for target in range(1, max_period + 1):
            prior = season_games[season_games["period"] < target]
            models[(season, target)] = fit(prior, k=k, verbose=False)
    return models
