"""
Elo rating sub-model based on raw run differential.

Design
------
Ratings are sequential with no window. K alone sets how fast the past is
forgotten -- a hard window is a second, redundant memory that fights K, and
measurement showed every finite window loses to unlimited history at every K
tested. See the parameter notes below.

Update, per game, from the home side's perspective:

    E_home  = 1 / (1 + 10^((R_away - (R_home + HFA)) / 400))
    S_home  = 1 if the home team won, else 0
    delta   = K * (S_home - E_home)
    R_home += delta
    R_away -= delta

The update is zero-sum, so the league total never drifts from 30 * 1500.

Home-field advantage enters the EXPECTATION only. It is never stored in a
rating, so ratings stay pure team strength, and the reported `elo_diff` is the
raw rating gap with no home term folded in -- the model already has a `home`
feature and would otherwise double-count it.

Season boundaries
-----------------
Ratings regress toward the mean between seasons rather than resetting flat:

    R_start = 1500 + carry * (R_end - 1500)

carry=0 is a full reset, carry=1 keeps everything. Teams absent from the prior
season start at 1500.

Parameter choices
-----------------
K=6, carry=0.5, HFA=20 were selected on 2022-2025 and confirmed by
walk-forward validation across 2023-2026 (~8,000 out-of-sample games), where
every fold independently selected no window and three of four selected
carry=0.5. Two independent criteria agreed on K=6: best log loss, and a
realised-versus-claimed slope ratio of 1.000 (the effect of a rating gap
matches what the 400-scale asserts).

HFA=20 is not fitted so much as implied -- home teams win 52.99% of games, and
400 * log10(.5299/.4701) = 20.8 rating points.

Expect out-of-sample AUC near 0.57 against the win label, not the 0.583 the
tuning seasons suggest; walk-forward showed the tuned configuration performs no
better than this fixed one, so the constants are deliberately not searched.

Features produced
-----------------
    elo        : team's rating BEFORE this game
    opp_elo    : opponent's rating BEFORE this game
    elo_diff   : elo - opp_elo

Pre-game ratings are recorded before the result is applied, so a game never
informs its own features.

Public interface
----------------
compute(games_df, k, carry, hfa, initial_rating) -> pd.DataFrame
    Indexed by (team, season, period), columns: elo, opp_elo, elo_diff
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_INITIAL_RATING = 1500.0
_K = 6.0
_CARRY = 0.5
_HFA = 20.0

# Franchises that changed name mid-history. Without canonicalising, carryover
# treats the renamed club as brand new and resets it to the initial rating.
_CANON = {"Oakland Athletics": "Athletics"}


def _game_table(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse team-perspective rows into one row per game, home side's view.

    Each game normally appears twice, once per team. A handful have only one
    perspective stored; those are recovered from whichever row exists.
    """
    df = games_df.dropna(subset=["opponent", "diff"]).copy()
    # Canonical names drive the ratings so a renamed franchise keeps its history.
    # The ORIGINAL names are carried alongside and used for the output index, or
    # the caller's (team, season, period) keys would stop matching.
    df["_team_c"] = df["team"].replace(_CANON)
    df["_opp_c"] = df["opponent"].replace(_CANON)

    has_home = "home" in df.columns and df["home"].notna().any()
    if not has_home:
        # No home flag: treat every row as its own perspective and de-duplicate
        # on the unordered pair so each game is applied once.
        df["_pair"] = [tuple(sorted(p)) for p in zip(df["_team_c"], df["_opp_c"])]
        df = df.drop_duplicates(subset=["season", "_pair", "period"], keep="first")
        df["home_team"] = df["_team_c"]
        df["away_team"] = df["_opp_c"]
        df["home_name"] = df["team"]
        df["away_name"] = df["opponent"]
        df["home_margin"] = pd.to_numeric(df["diff"], errors="coerce")
        df["home_period"] = df["period"]
        df["away_period"] = np.nan
        return df

    rows = []
    key = "game_pk" if "game_pk" in df.columns else None
    if key is None:
        df["game_pk"] = [f"{s}-{'-'.join(sorted(p))}-{pd_}" for s, p, pd_ in
                         zip(df["season"],
                             zip(df["team"], df["opponent"]),
                             df["period"])]
        key = "game_pk"

    for _, blk in df.groupby(key, sort=False):
        h = blk[blk["home"] == 1]
        a = blk[blk["home"] != 1]
        if len(h):
            r = h.iloc[0]
            home_team, away_team = r["_team_c"], r["_opp_c"]
            home_name, away_name = r["team"], r["opponent"]
            margin = float(r["diff"])
            hp = r["period"]
            ap = a.iloc[0]["period"] if len(a) else np.nan
            date = r.get("date")
            season = r["season"]
        else:                                   # orphan: only the away row exists
            r = a.iloc[0]
            home_team, away_team = r["_opp_c"], r["_team_c"]
            home_name, away_name = r["opponent"], r["team"]
            margin = -float(r["diff"])
            hp = np.nan
            ap = r["period"]
            date = r.get("date")
            season = r["season"]
        rows.append((season, date, r.get("game_pk"), home_team, away_team,
                     home_name, away_name, margin, hp, ap))

    out = pd.DataFrame(rows, columns=["season", "date", "game_pk", "home_team",
                                      "away_team", "home_name", "away_name",
                                      "home_margin", "home_period",
                                      "away_period"])
    return out


def compute(
    games_df: pd.DataFrame,
    k: float = _K,
    carry: float = _CARRY,
    hfa: float = _HFA,
    initial_rating: float = _INITIAL_RATING,
) -> pd.DataFrame:
    """
    Pre-game Elo ratings for every (team, season, period).

    Parameters
    ----------
    games_df       : columns team, opponent, season, period, diff; `home`,
                     `date` and `game_pk` are used when present.
    k              : rating sensitivity. Also the memory: a game's influence
                     half-lives in roughly 481/k games.
    carry          : fraction of end-of-season rating carried into the next.
    hfa            : home-field advantage in rating points, applied to the
                     expectation only and never stored.
    initial_rating : rating for a team with no history.
    """
    needed = {"team", "opponent", "season", "period", "diff"}
    missing = needed - set(games_df.columns)
    if missing:
        raise ValueError(f"games_df missing columns: {missing}")

    g = _game_table(games_df)
    if g.empty:
        return pd.DataFrame(
            columns=["elo", "opp_elo", "elo_diff"],
            index=pd.MultiIndex.from_arrays([[], [], []],
                                            names=["team", "season", "period"]),
        )

    # Chronological order. `period` is a per-team game counter, so it orders a
    # single team correctly but is not comparable across teams; date is the
    # league-wide clock, with game_pk breaking doubleheader ties.
    sort_cols = ["season"]
    if "date" in g.columns and g["date"].notna().any():
        sort_cols.append("date")
    if "game_pk" in g.columns:
        g["_pk"] = pd.to_numeric(g["game_pk"], errors="coerce")
        sort_cols.append("_pk")
    else:
        sort_cols.append("home_period")
    g = g.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    ratings: dict[str, float] = {}
    prev_season = None
    records: list[dict] = []

    seasons = g["season"].to_numpy()
    homes = g["home_team"].to_numpy()
    aways = g["away_team"].to_numpy()
    hname = g["home_name"].to_numpy()
    aname = g["away_name"].to_numpy()
    margins = g["home_margin"].to_numpy(dtype=float)
    hper = g["home_period"].to_numpy()
    aper = g["away_period"].to_numpy()

    for i in range(len(g)):
        season = seasons[i]
        if season != prev_season:
            ratings = {t: initial_rating + carry * (r - initial_rating)
                       for t, r in ratings.items()}
            prev_season = season

        h, a = homes[i], aways[i]
        rh = ratings.get(h, initial_rating)
        ra = ratings.get(a, initial_rating)

        # Record pre-game state for both sides before the result is applied.
        if hper[i] == hper[i]:                              # not NaN
            records.append({"team": hname[i], "season": season, "period": hper[i],
                            "elo": rh, "opp_elo": ra, "elo_diff": rh - ra})
        if aper[i] == aper[i]:
            records.append({"team": aname[i], "season": season, "period": aper[i],
                            "elo": ra, "opp_elo": rh, "elo_diff": ra - rh})

        e_home = 1.0 / (1.0 + 10 ** ((ra - (rh + hfa)) / 400.0))
        s_home = 1.0 if margins[i] > 0 else 0.0
        delta = k * (s_home - e_home)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    out = pd.DataFrame.from_records(records)
    out["period"] = pd.to_numeric(out["period"], errors="coerce").astype("int64")
    return out.set_index(["team", "season", "period"])[["elo", "opp_elo", "elo_diff"]]
