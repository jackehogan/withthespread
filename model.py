"""
Machine-learning layer: feature engineering, hyperparameter tuning, training,
and prediction.

Feature design
--------------
Rolling SpreadScore (diff + spread_line) is used as the feature matrix.
SpreadScore captures performance-vs-expectations, which is the signal most
relevant to future spread coverage. ~31% of feature cells are NaN (games
without a spread line); XGBoost handles these natively.

Context features added for the target period:
  home   : 1 if the team is playing at home, 0 if away
  is_b2b : 1 if the team played the previous day (back-to-back), else 0
  spread : the game's spread line (encodes the market's prior expectation)

The `team` identifier is intentionally excluded from the feature matrix to
prevent the model memorising team identities instead of learning form patterns.
`season` and `period` are retained as categorical context features.

Lookback (number of prior periods used as features) is tuned automatically
via cross-validation on the training set over candidates [3, 5, 7, 10, 15].

Split strategy (no random shuffling — strict temporal ordering)
---------------------------------------------------------------
  Train : all seasons except eval_season
  Test  : eval_season, target period <  eval_split_period  (first half)
  Val   : eval_season, target period >= eval_split_period  (second half)

Public interface
----------------
build_features(games_df, next_period, lookback, eval_season, eval_split_period)
build_prediction_features(season_games, next_period, lookback, season)
train_models(games_df, next_period, eval_season, eval_split_period, max_evals)
    -> reg, clas, scores, best_lookback
predict(reg, clas, X_pred)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from sklearn.model_selection import cross_val_score
from scipy.stats import norm
from xgboost import XGBClassifier, XGBRegressor

import embeddings as emb
import elo as elo_mod

_XGB_FIXED = {"enable_categorical": True, "tree_method": "hist"}

# team and season excluded — see module docstring
_CAT_COLS = ("period",)

_LOOKBACK_CANDIDATES = [3, 5, 7, 10, 15]
_K_CANDIDATES        = [16, 32, 48, 64]

_HYPEROPT_SPACE = {
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
    "max_depth": hp.quniform("max_depth", 2, 5, 1),
    "n_estimators": hp.quniform("n_estimators", 50, 200, 10),
    "reg_lambda": hp.uniform("reg_lambda", 1, 5),
    "reg_alpha": hp.uniform("reg_alpha", 1, 5),
    "min_child_weight": hp.uniform("min_child_weight", 1, 5),
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _precompute(
    games_df: pd.DataFrame,
    next_period: int,
    eval_season: int,
    k_values: list[float],
) -> dict:
    """
    Compute everything that doesn't depend on lookback or a specific K value.

    Called once per (next_period, eval_season) and shared across all
    (lookback, K) candidates in _select_hyperparams and the final
    build_features call, eliminating ~20× redundant recomputation.

    Returns a dict with keys:
      elo_by_k     : dict  K -> elo_ratings DataFrame (team, season, period)
      train_ss     : SS pivot rows for non-eval seasons
      eval_ss      : SS pivot rows for eval season
      train_ctx    : context (home, is_b2b, spread) for non-eval seasons
      eval_ctx     : context for eval season
      train_style  : style_edge Series for non-eval seasons (winsorised)
      eval_style   : style_edge Series for eval season (winsorised)
      style_model  : StyleModel fit on current season up to next_period
    """
    # --- SS pivot + context (shared across all lookback/K combos) ---
    ss_pivot = games_df.pivot_table(
        index=["team", "season"], columns="period", values="spreadscore"
    )
    context = _compute_context(games_df)

    is_eval = ss_pivot.index.get_level_values("season") == eval_season
    train_ss  = ss_pivot[~is_eval]
    eval_ss   = ss_pivot[is_eval]
    train_ctx = context[~context.index.get_level_values("season").isin([eval_season])]
    eval_ctx  = context[context.index.get_level_values("season") == eval_season]

    # --- Elo: one pass per K value, split into train/eval immediately ---
    elo_by_k = {}
    train_elo_by_k = {}
    eval_elo_by_k  = {}
    for k in k_values:
        er = elo_mod.compute(games_df, k=k)
        elo_by_k[k]       = er
        _eval_mask = er.index.get_level_values("season") == eval_season
        train_elo_by_k[k] = er[~_eval_mask]
        eval_elo_by_k[k]  = er[_eval_mask]

    # --- Style embeddings: one pass total ---
    all_seasons   = games_df["season"].unique().tolist()
    period_models = emb.build_period_models(games_df, all_seasons, next_period, k=3)

    records = []
    for (season, target), model in period_models.items():
        period_games = games_df[
            (games_df["season"] == season) & (games_df["period"] == target)
        ]
        for _, row in period_games.iterrows():
            edge = model.predict_edge(row["team"], row["opponent"]) if model else np.nan
            records.append({"team": row["team"], "season": season,
                            "period": target, "style_edge": edge})

    style_edges = (
        pd.DataFrame(records)
        .set_index(["team", "season", "period"])["style_edge"]
        if records else pd.Series(dtype=float, name="style_edge")
    )

    # Winsorise at ±3σ computed on training seasons only
    _train_mask = ~style_edges.index.get_level_values("season").isin([eval_season])
    _se_std = style_edges[_train_mask].std()
    _clip = float(3.0 * _se_std) if _se_std > 0 else np.inf
    if _se_std > 0:
        style_edges = style_edges.clip(lower=-_clip, upper=_clip)

    train_style = style_edges[
        ~style_edges.index.get_level_values("season").isin([eval_season])
    ]
    eval_style = style_edges[
        style_edges.index.get_level_values("season") == eval_season
    ]

    # Style model for prediction time: fit on current season up to next_period
    current_season_games = games_df[games_df["season"] == max(all_seasons)]
    style_model = emb.fit(
        current_season_games[current_season_games["period"] < next_period],
        k=3, verbose=True,
    )
    if style_model is not None and _se_std > 0:
        style_model.edge_clip = _clip

    # Pre-group context, style, and Elo by period for O(1) lookup inside the
    # feature precomputation loop (replaces O(n) .xs() calls).
    def _group_by_period(df_or_series):
        """Return dict mapping period -> sub-DataFrame/Series (level dropped)."""
        out = {}
        level = df_or_series.index.names.index("period")
        for period, grp in df_or_series.groupby(level="period"):
            out[period] = grp.droplevel("period")
        return out

    train_ctx_by_p   = _group_by_period(train_ctx)
    eval_ctx_by_p    = _group_by_period(eval_ctx)
    train_style_by_p = _group_by_period(train_style)
    eval_style_by_p  = _group_by_period(eval_style)
    train_elo_by_k_p = {k: _group_by_period(train_elo_by_k[k]) for k in k_values}
    eval_elo_by_k_p  = {k: _group_by_period(eval_elo_by_k[k])  for k in k_values}

    # Precompute complete feature rows per (target, K) so _select_hyperparams
    # can iterate over lookback candidates with only dict lookups — no repeated
    # pivot/xs/streak computation.  The window-collection output for a given
    # target is identical across all lookback values; only the filtering differs.
    train_feats_by_k_target: dict[float, dict[int, tuple]] = {k: {} for k in k_values}
    eval_feats_by_k_target:  dict[float, dict[int, tuple]] = {k: {} for k in k_values}
    train_n_prior_by_target: dict[int, int] = {}
    eval_n_prior_by_target:  dict[int, int] = {}

    all_targets = sorted(set(train_ss.columns) | set(eval_ss.columns))
    _CTX = ["home", "is_b2b", "spread"]

    def _add_elo(df_base, common, elo_by_p, target):
        elo_grp = elo_by_p.get(target)
        if elo_grp is not None:
            ea = elo_grp.reindex(common)
            return ea["elo_diff"].values, ea["opp_elo"].values
        return np.full(len(common), np.nan), np.full(len(common), np.nan)

    for target in all_targets:
        # --- train split ---
        if target in train_ss.columns:
            n_prior = int(sum(1 for c in train_ss.columns if c < target))
            train_n_prior_by_target[target] = n_prior
            y_tr = train_ss[target].dropna()
            y_tr = y_tr[y_tr != 0]
            if not y_tr.empty:
                common = train_ss.index.intersection(y_tr.index)
                y_tr   = y_tr.loc[common]
                base   = _compute_ss_features(train_ss, common, target).reset_index(drop=True)
                base["period"] = target
                ctx_grp = train_ctx_by_p.get(target)
                for col in _CTX:
                    base[col] = ctx_grp.reindex(common)[col].values if (ctx_grp is not None and col in ctx_grp.columns) else np.nan
                sty_grp = train_style_by_p.get(target)
                base["style_edge"] = sty_grp.reindex(common).values if sty_grp is not None else np.nan
                for k in k_values:
                    df_k = base.copy()
                    df_k["elo_diff"], df_k["opponent_elo"] = _add_elo(base, common, train_elo_by_k_p[k], target)
                    _recast_categoricals(df_k)
                    for col in df_k.columns:
                        if col not in _CAT_COLS:
                            df_k[col] = pd.to_numeric(df_k[col], errors="coerce")
                    train_feats_by_k_target[k][target] = (df_k, pd.Series(y_tr.values, dtype=float))

        # --- eval split ---
        if target in eval_ss.columns:
            n_prior = int(sum(1 for c in eval_ss.columns if c < target))
            eval_n_prior_by_target[target] = n_prior
            y_ev = eval_ss[target].dropna()
            y_ev = y_ev[y_ev != 0]
            if not y_ev.empty:
                common = eval_ss.index.intersection(y_ev.index)
                y_ev   = y_ev.loc[common]
                base   = _compute_ss_features(eval_ss, common, target).reset_index(drop=True)
                base["period"] = target
                ctx_grp = eval_ctx_by_p.get(target)
                for col in _CTX:
                    base[col] = ctx_grp.reindex(common)[col].values if (ctx_grp is not None and col in ctx_grp.columns) else np.nan
                sty_grp = eval_style_by_p.get(target)
                base["style_edge"] = sty_grp.reindex(common).values if sty_grp is not None else np.nan
                for k in k_values:
                    df_k = base.copy()
                    df_k["elo_diff"], df_k["opponent_elo"] = _add_elo(base, common, eval_elo_by_k_p[k], target)
                    _recast_categoricals(df_k)
                    for col in df_k.columns:
                        if col not in _CAT_COLS:
                            df_k[col] = pd.to_numeric(df_k[col], errors="coerce")
                    eval_feats_by_k_target[k][target] = (df_k, pd.Series(y_ev.values, dtype=float))

    return {
        "elo_by_k":                elo_by_k,
        "train_elo_by_k":          train_elo_by_k,
        "eval_elo_by_k":           eval_elo_by_k,
        "train_ss":                train_ss,
        "eval_ss":                 eval_ss,
        "train_ctx":               train_ctx,
        "eval_ctx":                eval_ctx,
        "train_style":             train_style,
        "eval_style":              eval_style,
        "style_model":             style_model,
        "train_feats_by_k_target": train_feats_by_k_target,
        "eval_feats_by_k_target":  eval_feats_by_k_target,
        "train_n_prior_by_target": train_n_prior_by_target,
        "eval_n_prior_by_target":  eval_n_prior_by_target,
    }


def build_features(
    games_df: pd.DataFrame,
    next_period: int,
    lookback: int,
    eval_season: int,
    eval_split_period: int,
    best_k: float = 32,
    _cache: "dict | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build train/test/val splits with strict temporal separation.

    Parameters
    ----------
    games_df          : columns team, season, period, spreadscore
    next_period       : the period to predict; windows end before it
    lookback          : number of prior spreadscore values used as features
    eval_season       : season held out entirely from training
    eval_split_period : period boundary dividing test (before) and val (from)
    _cache            : precomputed dict from _precompute() containing train_ss,
                        eval_ss, train_ctx, eval_ctx, train_style, eval_style,
                        elo_by_k, and style_model. When supplied, all expensive
                        recomputations are skipped — only the window-collection
                        loop runs, which varies with lookback.

    Returns
    -------
    X_train, X_test, y_train, y_test, X_val, y_val, style_model
    """
    if _cache is not None and "train_feats_by_k_target" in _cache:
        # Fastest path: feature rows precomputed per target — just filter by lookback.
        train_feats   = _cache["train_feats_by_k_target"][best_k]
        eval_feats    = _cache["eval_feats_by_k_target"][best_k]
        train_n_prior = _cache["train_n_prior_by_target"]
        eval_n_prior  = _cache["eval_n_prior_by_target"]
        style_model   = _cache["style_model"]

        X_train_parts, y_train_parts = [], []
        X_test_parts,  y_test_parts  = [], []
        X_val_parts,   y_val_parts   = [], []

        for start in range(1, next_period - lookback + 1):
            target = start + lookback
            if target in train_feats and train_n_prior.get(target, 0) >= lookback:
                df, y = train_feats[target]
                X_train_parts.append(df)
                y_train_parts.append(y)
            if target in eval_feats and eval_n_prior.get(target, 0) >= lookback:
                df, y = eval_feats[target]
                if target < eval_split_period:
                    X_test_parts.append(df)
                    y_test_parts.append(y)
                else:
                    X_val_parts.append(df)
                    y_val_parts.append(y)

        if not X_train_parts:
            raise ValueError(
                f"No training windows found (next_period={next_period}, lookback={lookback})."
            )

        def _concat(parts, y_parts):
            X = pd.concat(parts, ignore_index=True)
            _recast_categoricals(X)
            y = pd.concat(y_parts, ignore_index=True).astype(float)
            return X, y

        X_train, y_train = _concat(X_train_parts, y_train_parts)
        X_test,  y_test  = _concat(X_test_parts,  y_test_parts)  if X_test_parts  else (pd.DataFrame(columns=X_train.columns), pd.Series(dtype=float))
        X_val,   y_val   = _concat(X_val_parts,   y_val_parts)   if X_val_parts   else (pd.DataFrame(columns=X_train.columns), pd.Series(dtype=float))
        return X_train, X_test, y_train, y_test, X_val, y_val, style_model

    if _cache is not None:
        # Fallback cache path (no precomputed features — unpack splits, run window loop).
        train_ss    = _cache["train_ss"]
        eval_ss     = _cache["eval_ss"]
        train_ctx   = _cache["train_ctx"]
        eval_ctx    = _cache["eval_ctx"]
        train_style = _cache["train_style"]
        eval_style  = _cache["eval_style"]
        style_model = _cache["style_model"]
        train_elo   = _cache["train_elo_by_k"][best_k]
        eval_elo    = _cache["eval_elo_by_k"][best_k]
    else:
        # Slow path: compute everything from scratch.
        ss_pivot = games_df.pivot_table(
            index=["team", "season"], columns="period", values="spreadscore"
        )
        context = _compute_context(games_df)

        is_eval   = ss_pivot.index.get_level_values("season") == eval_season
        train_ss  = ss_pivot[~is_eval]
        eval_ss   = ss_pivot[is_eval]
        train_ctx = context[~context.index.get_level_values("season").isin([eval_season])]
        eval_ctx  = context[context.index.get_level_values("season") == eval_season]

        elo_ratings = elo_mod.compute(games_df, k=best_k)
        train_elo   = elo_ratings[
            ~elo_ratings.index.get_level_values("season").isin([eval_season])
        ]
        eval_elo    = elo_ratings[
            elo_ratings.index.get_level_values("season") == eval_season
        ]

        all_seasons   = games_df["season"].unique().tolist()
        period_models = emb.build_period_models(games_df, all_seasons, next_period, k=3)
        records = []
        for (season, target), model in period_models.items():
            period_games = games_df[
                (games_df["season"] == season) & (games_df["period"] == target)
            ]
            for _, row in period_games.iterrows():
                edge = model.predict_edge(row["team"], row["opponent"]) if model else np.nan
                records.append({"team": row["team"], "season": season,
                                "period": target, "style_edge": edge})
        style_edges = (
            pd.DataFrame(records)
            .set_index(["team", "season", "period"])["style_edge"]
            if records else pd.Series(dtype=float, name="style_edge")
        )
        _train_mask = ~style_edges.index.get_level_values("season").isin([eval_season])
        _se_std = style_edges[_train_mask].std()
        _clip = float(3.0 * _se_std) if _se_std > 0 else np.inf
        if _se_std > 0:
            style_edges = style_edges.clip(lower=-_clip, upper=_clip)
        train_style = style_edges[
            ~style_edges.index.get_level_values("season").isin([eval_season])
        ]
        eval_style  = style_edges[
            style_edges.index.get_level_values("season") == eval_season
        ]
        current_season_games = games_df[games_df["season"] == max(all_seasons)]
        style_model = emb.fit(
            current_season_games[current_season_games["period"] < next_period],
            k=3, verbose=True,
        )
        if style_model is not None and _se_std > 0:
            style_model.edge_clip = _clip

    X_train_parts, y_train_parts = [], []
    X_test_parts,  y_test_parts  = [], []
    X_val_parts,   y_val_parts   = [], []

    for start in range(1, next_period - lookback + 1):
        target = start + lookback
        _collect_window(train_ss, train_ctx, train_style, train_elo,
                        start, lookback, target, X_train_parts, y_train_parts)
        if target < eval_split_period:
            _collect_window(eval_ss, eval_ctx, eval_style, eval_elo,
                            start, lookback, target, X_test_parts, y_test_parts)
        else:
            _collect_window(eval_ss, eval_ctx, eval_style, eval_elo,
                            start, lookback, target, X_val_parts, y_val_parts)

    if not X_train_parts:
        raise ValueError(
            f"No training windows found (next_period={next_period}, lookback={lookback}). "
            f"Need at least {lookback + 1} periods of data."
        )

    def _concat(parts, y_parts):
        X = pd.concat(parts, ignore_index=True)
        _recast_categoricals(X)
        y = pd.concat(y_parts, ignore_index=True).astype(float)
        return X, y

    X_train, y_train = _concat(X_train_parts, y_train_parts)

    if X_test_parts:
        X_test, y_test = _concat(X_test_parts, y_test_parts)
    else:
        X_test = pd.DataFrame(columns=X_train.columns)
        y_test = pd.Series(dtype=float)

    if X_val_parts:
        X_val, y_val = _concat(X_val_parts, y_val_parts)
    else:
        X_val = pd.DataFrame(columns=X_train.columns)
        y_val = pd.Series(dtype=float)

    return X_train, X_test, y_train, y_test, X_val, y_val, style_model


def _recast_categoricals(df: pd.DataFrame) -> None:
    """Re-cast _CAT_COLS to category dtype in-place (lost after pd.concat)."""
    for col in _CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")


def _compute_context(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(team, season, period) context features:
      home              : 1 if playing at home, 0 if away
      is_b2b            : 1 if the team played yesterday, else 0
      spread            : game spread line (NaN if unavailable)
      off_rating        : team's cumulative offensive rating through prior period
      def_rating        : team's cumulative defensive rating through prior period
      net_rating        : team's cumulative net rating through prior period
      opp_off_rating    : opponent's off_rating (same period)
      opp_def_rating    : opponent's def_rating (same period)
      matchup_off_edge  : team off_rating - opp def_rating (offensive advantage)
      matchup_def_edge  : team def_rating - opp off_rating (defensive advantage, lower = better)

    Returns a DataFrame indexed by (team, season, period).
    """
    # spread_juice, total, implied_prob excluded until historical odds are backfilled
    # off_rating / def_rating / net_rating populated once nba_api ratings are seeded
    _RATING_COLS = ["off_rating", "def_rating", "net_rating"]
    _CTX_ODDS    = ["spread"] + _RATING_COLS

    needed = {"team", "season", "period", "date", "home"}
    _matchup_cols = ["opp_off_rating", "opp_def_rating", "matchup_off_edge", "matchup_def_edge"]
    _all_out = ["home", "is_b2b"] + _CTX_ODDS + _matchup_cols

    if not needed.issubset(games_df.columns):
        idx = pd.MultiIndex.from_frame(games_df[["team", "season", "period"]])
        return pd.DataFrame({c: np.nan for c in _all_out}, index=idx)

    odds_present = [c for c in _CTX_ODDS if c in games_df.columns]
    opp_present  = "opponent" in games_df.columns

    cols = ["team", "season", "period", "date", "home"] + odds_present
    if opp_present:
        cols.append("opponent")

    df = games_df[cols].copy()
    df = df.sort_values(["team", "season", "period"])
    df["prev_date"] = df.groupby(["team", "season"])["date"].shift(1)
    rest = (pd.to_datetime(df["date"]) - pd.to_datetime(df["prev_date"])).dt.days
    df["is_b2b"] = (rest == 1).astype(float)
    df["home"]   = df["home"].astype(float)

    for c in _CTX_ODDS:
        if c not in df.columns:
            df[c] = np.nan

    # --- Opponent matchup features ---
    # Build a lookup: (team, season, period) -> off_rating, def_rating
    ratings_have = [c for c in ["off_rating", "def_rating"] if c in df.columns]
    if opp_present and ratings_have:
        rating_lookup = (
            df.set_index(["team", "season", "period"])[ratings_have]
        )
        # Join opponent ratings by resolving (opponent, season, period)
        opp_keys = list(zip(df["opponent"], df["season"], df["period"]))
        opp_ratings = rating_lookup.reindex(opp_keys)
        opp_ratings.index = df.index

        if "off_rating" in ratings_have:
            df["opp_off_rating"] = opp_ratings["off_rating"].values
        if "def_rating" in ratings_have:
            df["opp_def_rating"] = opp_ratings["def_rating"].values

        # Matchup edges
        if "off_rating" in ratings_have and "def_rating" in ratings_have:
            df["matchup_off_edge"] = df["off_rating"] - df["opp_def_rating"]   # positive = offence advantage
            df["matchup_def_edge"] = df["def_rating"] - df["opp_off_rating"]   # negative = defensive advantage
    else:
        for c in _matchup_cols:
            df[c] = np.nan

    out_cols = [c for c in _all_out if c in df.columns]
    return df.set_index(["team", "season", "period"])[out_cols]


_SS_MEAN_WINDOW  = 5   # fixed rolling window for ss_mean (EDA: 5-8 games optimal)
_STREAK_CAP      = 5   # cap consecutive cover/fade streak at this value


def _compute_ss_features(ss_pivot: pd.DataFrame, common_idx, target: int) -> pd.DataFrame:
    """
    Compute SpreadScore-derived features for a set of team-season rows:
      1_ago_ss    : most recent game's SpreadScore (XGBoost finds ±15 threshold)
      ss_mean_5   : mean over last 5 games (fixed window, EDA-validated)
      cover_streak: consecutive covers going into this game (0-_STREAK_CAP)
                    momentum signal — monotone in EDA (50% -> 52.4% at N=5)
      fade_streak : consecutive fades going into this game (0-_STREAK_CAP)
                    mean-reversion signal — peaks at N=3 (52.5%), non-monotone,
                    kept separate so XGBoost learns different response function
    """
    all_prior = sorted(c for c in ss_pivot.columns if c < target)

    # 1_ago_ss
    lag1_col = target - 1
    lag1 = (
        ss_pivot[lag1_col].reindex(common_idx)
        if lag1_col in ss_pivot.columns
        else pd.Series(np.nan, index=common_idx)
    )

    # ss_mean_5 — use up to _SS_MEAN_WINDOW periods; allow fewer (NaN if none)
    mean_cols = all_prior[-_SS_MEAN_WINDOW:]
    ss_mean = (
        ss_pivot[mean_cols].reindex(common_idx).mean(axis=1)
        if mean_cols
        else pd.Series(np.nan, index=common_idx)
    )

    # cover_streak and fade_streak — vectorised over all team-season rows.
    #
    # Algorithm (numpy, no Python loop over teams):
    #   1. M : (n_teams, n_prior) matrix, reversed so column 0 = most recent game.
    #   2. valid[i,j] : True when M[i,j] is non-NaN and non-zero (not a push).
    #   3. direction[i] : sign of the first valid value in row i (0 if none).
    #   4. same_dir[i,j] : True when M[i,j] is valid AND has the same sign as
    #      direction[i] — i.e. this game continues the streak.
    #   5. cumprod of same_dir along axis=1: 1 while the streak holds, 0 after
    #      the first break.  Summing gives streak length.
    #   6. Clip at _STREAK_CAP and split into cover/fade by direction.
    n = len(common_idx)
    if not all_prior:
        cover_streak_vals = np.zeros(n)
        fade_streak_vals  = np.zeros(n)
    else:
        M = ss_pivot[all_prior].reindex(common_idx).values[:, ::-1]   # (n, n_prior)
        M_sign = np.sign(M)                                            # -1 / 0 / +1 / nan

        # valid: non-NaN AND non-zero (zero = push, which breaks the streak)
        valid = (~np.isnan(M)) & (M_sign != 0)                        # (n, n_prior)

        # direction = sign of first valid column per row (0 if no valid column)
        has_valid    = valid.any(axis=1)                               # (n,)
        first_col    = valid.argmax(axis=1)                            # (n,)
        direction    = np.where(has_valid, M_sign[np.arange(n), first_col], 0.0)  # (n,)

        # same_dir: column matches direction AND is a valid (non-NaN, non-zero) result
        same_dir = valid & (M_sign == direction[:, np.newaxis])        # (n, n_prior)

        # cumprod collapses to 0 after the first False — sum = streak length
        run      = np.cumprod(same_dir.astype(np.int8), axis=1)
        streaks  = run.sum(axis=1).clip(0, _STREAK_CAP).astype(float)

        cover_streak_vals = np.where(direction ==  1, streaks, 0.0)
        fade_streak_vals  = np.where(direction == -1, streaks, 0.0)

    out = pd.DataFrame({
        "1_ago_ss":     lag1.values,
        "ss_mean_5":    ss_mean.values,
        "cover_streak": cover_streak_vals,
        "fade_streak":  fade_streak_vals,
    }, index=common_idx)
    return out


def _collect_window(
    ss_pivot: pd.DataFrame,
    context: pd.DataFrame,
    style_edges: pd.Series,
    elo_ratings: pd.DataFrame,
    start: int,
    lookback: int,
    target: int,
    X_out: list,
    y_out: list,
) -> None:
    """
    Append one sliding-window sample to X_out / y_out.

    Features:
      SS-derived  : 1_ago_ss, ss_mean_5, cover_streak  (fixed windows, EDA-validated)
      Context     : home, is_b2b, spread  (for the target period)
      Sub-models  : style_edge, elo_diff, opponent_elo
    Target        : spreadscore at `target`, non-NaN and non-zero (no pushes).
    `team`        : dropped — identity memorisation risk.
    `lookback`    : controls training window density, not feature window.
    """
    if target not in ss_pivot.columns:
        return
    # Require at least `lookback` completed periods before target for training density
    prior_cols = [c for c in ss_pivot.columns if c < target]
    if len(prior_cols) < lookback:
        return

    y = ss_pivot[target].dropna()
    y = y[y != 0]
    if y.empty:
        return

    common_idx = ss_pivot.index.intersection(y.index)
    y = y.loc[common_idx]

    # --- SpreadScore features (fixed windows) ---
    ss_feats = _compute_ss_features(ss_pivot, common_idx, target)
    df = ss_feats.reset_index(drop=True)
    df["period"] = target

    # --- Context features for the target period ---
    _CTX_COLS = ["home", "is_b2b", "spread", "off_rating", "def_rating", "net_rating",
                 "opp_off_rating", "opp_def_rating", "matchup_off_edge", "matchup_def_edge"]
    try:
        ctx_slice   = context.xs(target, level="period")
        ctx_aligned = ctx_slice.reindex(common_idx)
        for col in _CTX_COLS:
            df[col] = ctx_aligned[col].values if col in ctx_aligned.columns else np.nan
    except KeyError:
        for col in _CTX_COLS:
            df[col] = np.nan

    # --- Playstyle edge ---
    try:
        style_slice  = style_edges.xs(target, level="period")
        df["style_edge"] = style_slice.reindex(common_idx).values
    except KeyError:
        df["style_edge"] = np.nan

    # --- Elo ratings ---
    try:
        elo_slice    = elo_ratings.xs(target, level="period")
        elo_aligned  = elo_slice.reindex(common_idx)
        df["elo_diff"]     = elo_aligned["elo_diff"].values
        df["opponent_elo"] = elo_aligned["opp_elo"].values
    except KeyError:
        df["elo_diff"]     = np.nan
        df["opponent_elo"] = np.nan

    _recast_categoricals(df)
    for col in df.columns:
        if col not in _CAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    X_out.append(df)
    y_out.append(pd.Series(y.values, dtype=float))


def build_prediction_features(
    season_games: pd.DataFrame,
    next_period: int,
    lookback: int,
    season: int,
    upcoming_context: pd.DataFrame | None = None,
    style_model: emb.StyleModel | None = None,
    best_k: float = 32,
) -> pd.DataFrame:
    """
    Build the prediction feature matrix for upcoming games.

    `team` is kept in the returned DataFrame (used as the output index in
    predict()) but is not included in the trained feature names, so it is
    automatically excluded when the model selects its input columns.

    Parameters
    ----------
    upcoming_context : optional DataFrame indexed by team with columns
                       `home` (0/1), `is_b2b` (0/1), and `spread` for the
                       upcoming game. Missing columns default to NaN.
    style_model      : fitted StyleModel from embeddings.fit(); if provided,
                       computes style_edge for each upcoming matchup.
    best_k           : Elo K value selected during training — must match so
                       elo_diff and opponent_elo have the same scale as training.
    """
    completed = season_games[season_games["period"] < next_period]
    if completed.empty:
        return pd.DataFrame()

    # Build SS pivot for this season (team x period)
    ss_piv = completed.pivot_table(
        index="team", columns="period", values="spreadscore"
    )
    # Add a fake season level to match the (team, season) index used in training
    ss_piv.index = pd.MultiIndex.from_tuples(
        [(t, season) for t in ss_piv.index], names=["team", "season"]
    )

    teams = ss_piv.index
    ss_feats = _compute_ss_features(ss_piv, teams, next_period)
    X = ss_feats.reset_index()   # brings team + season back
    X = X.drop(columns=["season"], errors="ignore")
    X["period"] = next_period

    # Join context features for the upcoming game
    # spread_juice, total, implied_prob re-added once historical odds are backfilled
    _CTX_COLS = ["home", "is_b2b", "spread", "off_rating", "def_rating", "net_rating",
                 "opp_off_rating", "opp_def_rating", "matchup_off_edge", "matchup_def_edge"]
    if upcoming_context is not None and not upcoming_context.empty:
        for col in _CTX_COLS:
            X[col] = X["team"].map(upcoming_context[col]) if col in upcoming_context.columns else np.nan
    else:
        for col in _CTX_COLS:
            X[col] = np.nan

    # Elo ratings for upcoming game (pre-game, using all completed games this season)
    elo_df = elo_mod.compute(completed, k=best_k)
    # Get most recent Elo per team (last period before next_period)
    latest_elo = (
        elo_df.reset_index()
        .sort_values("period")
        .groupby("team")
        .last()
        [["elo", "opp_elo", "elo_diff"]]
    )
    X["elo_diff"]     = X["team"].map(latest_elo["elo_diff"])
    X["opponent_elo"] = X["team"].map(latest_elo["opp_elo"])

    # Playstyle edge — fit on completed games this season (period < next_period)
    # so we only use information available at prediction time
    if style_model is None:
        completed_games = completed  # already filtered to period < next_period
        style_model = emb.fit(completed_games, k=3)

    if style_model is not None and upcoming_context is not None and "opponent" in upcoming_context.columns:
        X["style_edge"] = X.apply(
            lambda r: style_model.predict_edge(
                r["team"],
                upcoming_context.loc[r["team"], "opponent"]
            ) if r["team"] in upcoming_context.index else np.nan,
            axis=1,
        )
    else:
        X["style_edge"] = np.nan

    _recast_categoricals(X)
    for col in X.columns:
        if col not in _CAT_COLS and col != "team":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _encode_cover(y: pd.Series) -> pd.Series:
    """Map sign of SpreadScore to binary cover label: positive → 1, else → 0."""
    return np.sign(y).replace({-1: 0, 1: 1}).astype(int)


def _select_hyperparams(
    games_df: pd.DataFrame,
    next_period: int,
    eval_season: int,
    eval_split_period: int,
) -> tuple[int, float, dict]:
    """
    Choose the best (lookback, K) combination via 5-fold CV on training data.

    Searches the Cartesian product of _LOOKBACK_CANDIDATES × _K_CANDIDATES.
    Scores the regressor (neg MSE) — regression is now the sole model.
    Falls back to (smallest lookback, K=32) if no candidate yields enough data.

    Elo and style embeddings are precomputed once and shared across all
    candidates — they don't depend on lookback or K (Elo varies with K but
    not lookback, so one Elo pass per K value suffices).

    Returns
    -------
    best_lookback, best_k, best_elo_ratings, precomp_style
    """
    print("  Precomputing Elo, style embeddings, and SS pivot...")
    cache = _precompute(games_df, next_period, eval_season, _K_CANDIDATES)

    best_lb, best_k, best_score = _LOOKBACK_CANDIDATES[0], _K_CANDIDATES[0], -np.inf
    for lb in _LOOKBACK_CANDIDATES:
        for k in _K_CANDIDATES:
            try:
                X_train, _, y_train, _, _, _, _ = build_features(
                    games_df, next_period, lb, eval_season, eval_split_period,
                    best_k=k, _cache=cache,
                )
            except ValueError:
                continue
            if len(X_train) < 30:
                continue
            score = cross_val_score(
                XGBRegressor(**_XGB_FIXED, max_depth=3, n_estimators=100,
                             learning_rate=0.1),
                X_train, y_train, cv=5, scoring="neg_mean_squared_error",
            ).mean()
            if score > best_score:
                best_score, best_lb, best_k = score, lb, k

    return best_lb, best_k, cache


def _tune(
    model_class: type,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_evals: int,
) -> dict:
    """Run Hyperopt TPE search; return best params with integer types corrected."""
    scoring = "f1" if model_class is XGBClassifier else "neg_mean_squared_error"

    def objective(params: dict) -> float:
        p = {
            **_XGB_FIXED, **params,
            "max_depth": int(params["max_depth"]),
            "n_estimators": int(params["n_estimators"]),
        }
        return -cross_val_score(
            model_class(**p), X_train, y_train, cv=5, scoring=scoring
        ).mean()

    best = fmin(fn=objective, space=_HYPEROPT_SPACE, algo=tpe.suggest, max_evals=max_evals)
    return {**best, "max_depth": int(best["max_depth"]),
            "n_estimators": int(best["n_estimators"])}


def train_models(
    games_df: pd.DataFrame,
    next_period: int,
    eval_season: int,
    eval_split_period: int,
    max_evals: int = 10,
) -> tuple[XGBRegressor, float, dict, int, int, emb.StyleModel]:
    """
    Select lookback, tune hyperparameters, and fit the regression model.

    Cover probability is derived from the regression output at inference time
    via a normal CDF: coverprob = Φ(spread_diff / σ_diff), where spread_diff
    is the difference between the two teams' predicted SpreadScores and σ_diff
    is the std of pairwise prediction errors estimated on training residuals.

    Returns
    -------
    reg          : fitted XGBRegressor (predicts SpreadScore per team)
    sigma_diff   : std of (predspread_A - predspread_B) out-of-sample residuals —
                   used to convert spread_diff to coverprob via normal CDF
    scores       : dict of train/test/val evaluation metrics
    best_lookback: lookback value selected by CV
    best_k       : Elo K value selected by CV — must be forwarded to
                   build_prediction_features so Elo features match training scale
    style_model  : fitted StyleModel for prediction-time style_edge
    """
    print("  Selecting lookback and K...")
    best_lookback, best_k, cache = _select_hyperparams(
        games_df, next_period, eval_season, eval_split_period
    )
    print(f"  Best lookback: {best_lookback}  Best K: {best_k}")

    X_train, X_test, y_train, y_test, X_val, y_val, style_model = build_features(
        games_df, next_period, best_lookback, eval_season, eval_split_period,
        best_k=best_k, _cache=cache,
    )
    print(f"  Rows — train: {len(X_train)}, test: {len(X_test)}, val: {len(X_val)}")

    print("  Tuning regression model...")
    reg = XGBRegressor(**_XGB_FIXED, **_tune(XGBRegressor, X_train, y_train, max_evals))
    reg.fit(X_train, y_train)

    # Estimate σ_diff from held-out residuals (val > test > train as fallback).
    # Using out-of-sample residuals avoids the overfitting bias that makes training
    # residuals ~38% smaller than true generalisation error (R² train≈0.52 vs val≈0.12).
    # σ_diff = σ_residual · √2  (Var(A-B) = Var(A) + Var(B) for independent predictions).
    train_resid = y_train.values - reg.predict(X_train)
    if not X_val.empty:
        oos_resid = y_val.values - reg.predict(X_val)
    elif not X_test.empty:
        oos_resid = y_test.values - reg.predict(X_test)
    else:
        oos_resid = train_resid   # last resort — early periods with no eval data
    sigma_resid = float(np.std(oos_resid))
    sigma_diff  = sigma_resid * np.sqrt(2)

    # Evaluation metrics
    train_sign_acc = float((np.sign(reg.predict(X_train)) == np.sign(y_train.values)).mean())

    sigma_source = "val" if not X_val.empty else ("test" if not X_test.empty else "train")
    scores = {
        "lookback":       best_lookback,
        "elo_k":          best_k,
        "sigma_source":   sigma_source,
        "sigma_resid":    round(sigma_resid, 3),
        "sigma_diff":     round(sigma_diff, 3),
        "reg_train_r2":   round(reg.score(X_train, y_train), 3),
        "reg_train_sign": round(train_sign_acc, 3),
        "reg_test_r2":    round(reg.score(X_test, y_test), 3) if not X_test.empty else None,
    }
    if not X_test.empty:
        test_sign_acc = float(
            (np.sign(reg.predict(X_test)) == np.sign(y_test.values)).mean()
        )
        scores["reg_test_sign"] = round(test_sign_acc, 3)
    if not X_val.empty:
        val_sign_acc = float(
            (np.sign(reg.predict(X_val)) == np.sign(y_val.values)).mean()
        )
        scores["reg_val_r2"]   = round(reg.score(X_val, y_val), 3)
        scores["reg_val_sign"] = round(val_sign_acc, 3)

    return reg, sigma_diff, scores, best_lookback, best_k, style_model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    reg: XGBRegressor,
    sigma_diff: float,
    X_pred: pd.DataFrame,
    opponent_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Generate predictions indexed by team.

    Steps
    -----
    1. Regression model predicts `predspread` per team (kept for inspection).
    2. For each game pair (A vs B): spread_diff = predspread_A - predspread_B.
       This cancels systematic bias and enforces P(A) + P(B) = 1.
    3. coverprob = Φ(spread_diff / sigma_diff) via normal CDF.

    Parameters
    ----------
    reg          : fitted XGBRegressor
    sigma_diff   : std of pairwise prediction errors from training (from train_models)
    X_pred       : feature matrix with a `team` column
    opponent_map : dict mapping team -> opponent for the upcoming game.
                   If None, coverprob falls back to Φ(predspread / (sigma_diff/√2)).

    Returns
    -------
    DataFrame indexed by team with columns:
        predspread  : raw regression output (positive = expected to cover)
        spread_diff : predspread_team - predspread_opponent  (decision signal)
        coverprob   : Φ(spread_diff / sigma_diff)
    """
    feat_cols  = reg.get_booster().feature_names
    predspread = reg.predict(X_pred[feat_cols])
    teams      = X_pred["team"].values

    pred_series = pd.Series(predspread, index=teams)

    spread_diff = np.full(len(teams), np.nan)
    # Per-team sigma: paired games use σ_diff; unmatched teams fall back to
    # σ_diff/√2 (the single-prediction uncertainty, since no opponent subtraction).
    sigmas = np.full(len(teams), sigma_diff / np.sqrt(2))

    for i, team in enumerate(teams):
        opp = (opponent_map or {}).get(team)
        if opp and opp in pred_series.index:
            spread_diff[i] = pred_series[team] - pred_series[opp]
            sigmas[i] = sigma_diff          # paired: spread_diff variance = 2·σ²
        else:
            spread_diff[i] = pred_series[team]  # single: use raw predspread
            # sigmas[i] already set to sigma_diff / √2

    coverprob = norm.cdf(spread_diff / sigmas)

    return pd.DataFrame(
        {
            "predspread":  predspread,
            "spread_diff": spread_diff,
            "coverprob":   coverprob,
        },
        index=teams,
    )
