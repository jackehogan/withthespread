"""
Elo rating sub-model based on raw point differential.

Why point differential (not SpreadScore)
-----------------------------------------
SpreadScore incorporates the betting spread, which moves based on public money
flow. A popular team may have an inflated spread regardless of true quality.
Point differential is a pure performance signal — market-independent — which
complements the SpreadScore rolling features already in the main model.

Rating update
-------------
Instead of binary win/loss, we use a continuous outcome score derived from
the point differential via a sigmoid:

    S_team = 1 / (1 + exp(-diff / scale))

A blowout win (+25) → S ≈ 0.85
A narrow win  (+3)  → S ≈ 0.55
A blowout loss(-25) → S ≈ 0.15

Then the standard Elo update:

    E_team = 1 / (1 + 10^((opp_elo - team_elo) / 400))
    new_elo = old_elo + K * (S_team - E_team)

Season reset
------------
Ratings reset to `initial_rating` (1500) at the start of each season.
This avoids stale ratings carrying over when rosters change significantly.

Features produced
-----------------
    elo_diff    : team_elo - opponent_elo  (relative strength)
    opponent_elo: opponent's absolute rating (quality of opposition)

Pre-game ratings are stored — the rating reflects what we knew *before*
the game, so there is no lookahead leakage.

K tuning
--------
K controls how fast ratings move. Candidates [16, 32, 48, 64] are searched
via cross-validation in model._select_k() alongside lookback selection.

Public interface
----------------
compute(games_df, k, initial_rating, scale) -> pd.DataFrame
    Returns DataFrame indexed by (team, season, period) with columns:
    elo, opp_elo, elo_diff
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

_INITIAL_RATING = 1500
_SCALE          = 15    # point diff at which sigmoid ≈ 0.73 (roughly one possession)


def compute(
    games_df: pd.DataFrame,
    k: float = 32,
    window: int = 20,
    initial_rating: float = _INITIAL_RATING,
    scale: float = _SCALE,
) -> pd.DataFrame:
    """
    Compute pre-game Elo ratings for every (team, season, period).

    Parameters
    ----------
    games_df       : DataFrame with columns team, opponent, season, period, diff.
    k              : rating sensitivity — how much ratings move per game.
    window         : number of most recent games used to compute the rating.
                     EDA showed a 20-game window has ~8x the predictive signal
                     of full-season cumulative Elo (cover gap +0.49 vs +0.03).
                     Full-season cumulative is equivalent to window=None.
    initial_rating : starting rating each season.
    scale          : point diff divisor in sigmoid (higher = less sensitive to margins).

    Returns
    -------
    DataFrame indexed by (team, season, period) with columns:
        elo         : team's Elo BEFORE this game
        opp_elo     : opponent's Elo BEFORE this game
        elo_diff    : elo - opp_elo
    """
    needed = {"team", "opponent", "season", "period", "diff"}
    if not needed.issubset(games_df.columns):
        raise ValueError(f"games_df missing columns: {needed - set(games_df.columns)}")

    df = (
        games_df[list(needed)]
        .dropna(subset=["opponent", "diff"])
        .sort_values(["season", "period"])
        .copy()
    )

    records: list[dict] = []

    for season, season_df in df.groupby("season"):
        # game_log stores unique games in order: (period, team, opp, diff)
        # Used for windowed recompute; team perspective only (opp diff = -diff).
        game_log: list[tuple] = []
        seen_games: set[tuple] = set()

        # Cumulative ratings used only when window is None
        cum_ratings: dict[str, float] = {}

        for period, period_df in season_df.groupby("period"):

            # NOTE: game_log is populated AFTER recording pre-game ratings so
            # that the current period's outcomes are never included in the Elo
            # features for that same period (no lookahead into the label).

            if window is not None:
                # Recompute from scratch over the last `window` unique games
                # (all from *prior* periods — current period not yet appended).
                # This gives each team a rating reflecting only recent form.
                recent = game_log[-window:]
                fresh: dict[str, float] = {}
                for (_, t, o, d) in recent:
                    r_t = fresh.get(t, initial_rating)
                    r_o = fresh.get(o, initial_rating)
                    s_t = 1.0 / (1.0 + math.exp(-d / scale))
                    e_t = 1.0 / (1.0 + 10 ** ((r_o - r_t) / 400))
                    fresh[t] = r_t + k * (s_t - e_t)
                    fresh[o] = r_o + k * ((1 - s_t) - (1 - e_t))
                pre = {t: fresh.get(t, initial_rating) for t in
                       set(r["team"] for _, r in period_df.iterrows()) |
                       set(r["opponent"] for _, r in period_df.iterrows())}
            else:
                # Standard cumulative: carry ratings forward from last period
                pre = {}
                for _, row in period_df.iterrows():
                    pre[row["team"]]     = cum_ratings.get(row["team"],     initial_rating)
                    pre[row["opponent"]] = cum_ratings.get(row["opponent"], initial_rating)

            # Record pre-game ratings (current period NOT yet in game_log)
            for _, row in period_df.iterrows():
                team = row["team"]
                opp  = row["opponent"]
                records.append({
                    "team":     team,
                    "season":   season,
                    "period":   period,
                    "elo":      pre.get(team, initial_rating),
                    "opp_elo":  pre.get(opp,  initial_rating),
                    "elo_diff": pre.get(team, initial_rating) - pre.get(opp, initial_rating),
                })

            # NOW append current period's games to game_log so they are
            # available as prior history for future periods only.
            for _, row in period_df.iterrows():
                team = row["team"]
                opp  = row["opponent"]
                key  = tuple(sorted([team, opp]))
                game_key = (period, key)
                if game_key not in seen_games:
                    seen_games.add(game_key)
                    game_log.append((period, team, opp, float(row["diff"])))

            # Update cumulative ratings (only used when window=None)
            if window is None:
                processed: set[tuple] = set()
                for _, row in period_df.iterrows():
                    team = row["team"]
                    opp  = row["opponent"]
                    key  = tuple(sorted([team, opp]))
                    if key in processed:
                        continue
                    processed.add(key)
                    r_t  = pre[team];  r_o = pre[opp]
                    d    = float(row["diff"])
                    s_t  = 1.0 / (1.0 + math.exp(-d / scale))
                    e_t  = 1.0 / (1.0 + 10 ** ((r_o - r_t) / 400))
                    cum_ratings[team] = r_t + k * (s_t - e_t)
                    cum_ratings[opp]  = r_o + k * ((1 - s_t) - (1 - e_t))

    return (
        pd.DataFrame(records)
        .set_index(["team", "season", "period"])
        [["elo", "opp_elo", "elo_diff"]]
    )
