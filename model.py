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
    -> clf, scores, best_lookback
predict(clf, X_pred)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBClassifier

import embeddings as emb
import elo as elo_mod

_XGB_FIXED = {"enable_categorical": True, "tree_method": "hist"}

# team and season excluded — see module docstring
_CAT_COLS = ("period",)

_LOOKBACK_CANDIDATES     = [7, 10, 15]
_K_CANDIDATES            = [16, 32, 48, 64]
_ELO_WINDOW              = 20    # EDA showed 20-game window has ~8x signal vs full-season cumulative
_BP_FATIGUE_DAYS         = 14    # fixed rolling window for bp_ip_14d — not a hyperparameter

# style_edge is not fed to the model. Measured on the 2025 hold-out it scored
# exactly 0.00 on both SHAP and XGBoost gain — the trees never split on it once
# — and removing the whole block cost 0.0000 test ROC.
#
# Out-of-sample testing showed why. The SVD was factorising raw run
# differential, which is team strength, so it only ever duplicated Elo with
# more noise. Residualising strength out first (the correct fix) left nothing:
# correlation with out-of-sample matchup residuals was 0.002 per game and
# -0.000 aggregated per team pair, across k=1..3, several shrinkage levels and
# multi-season pooling. At ~2.7 games per team pair there is no recoverable
# low-rank matchup effect above roughly 0.1 runs.
#
# The StyleModel plumbing is left intact so the saved bundle format still
# loads; only the feature injection is removed.
_USE_STYLE_EDGE          = False

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

_ATS_ELO_WINDOW = 40   # wider than standard Elo: spreadscore is the noisier signal


def _attach_ats_elo(elo_df: pd.DataFrame, games_df: pd.DataFrame,
                    k: float) -> pd.DataFrame:
    """
    Add ats_elo_diff / ats_opp_elo alongside the standard Elo columns.

    Same rating machinery, driven by spreadscore instead of run differential,
    so it rates COVERING rather than winning. Returns the frame unchanged if
    spreadscore is unavailable.
    """
    if "spreadscore" not in games_df.columns:
        elo_df["ats_elo_diff"] = np.nan
        elo_df["ats_opp_elo"] = np.nan
        return elo_df
    sub = games_df.dropna(subset=["spreadscore"])
    if sub.empty:
        elo_df["ats_elo_diff"] = np.nan
        elo_df["ats_opp_elo"] = np.nan
        return elo_df
    ats = elo_mod.compute(sub, k=k, window=_ATS_ELO_WINDOW,
                          value_col="spreadscore")
    elo_df["ats_elo_diff"] = ats["elo_diff"].reindex(elo_df.index)
    elo_df["ats_opp_elo"]  = ats["opp_elo"].reindex(elo_df.index)
    return elo_df


def _cover_and_win_labels(ss_vals, ctx_aligned) -> tuple[pd.Series, pd.Series]:
    """
    Derive both prediction targets from the same rows.

        cover = spreadscore > 0
        win   = diff > 0, and diff = spreadscore - spread

    No extra pivot is needed: raw spreadscore is still in hand at this point
    and `spread` comes from the context already fetched for the feature matrix.

    Win is NaN wherever spread is missing, since diff cannot be recovered
    there. Those rows are dropped when fitting the win model but still train
    the cover model, so neither target loses rows unnecessarily.
    """
    ss = np.asarray(ss_vals, dtype=float)
    cover = pd.Series((ss > 0).astype(float), dtype=float)

    if ctx_aligned is not None and "spread" in ctx_aligned.columns:
        spread = pd.to_numeric(ctx_aligned["spread"], errors="coerce").values
        diff = ss - spread
        win = pd.Series(np.where(np.isnan(diff), np.nan, (diff > 0).astype(float)),
                        dtype=float)
    else:
        win = pd.Series(np.full(len(ss), np.nan), dtype=float)
    return cover, win


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
    # Two ratings are computed. Standard Elo rates who WINS; ATS Elo runs the
    # same machinery on spreadscore and rates who COVERS. Measured on 2025,
    # each is better at its own target (standard .5766 vs win, .5483 vs cover;
    # ATS .5544 vs win, .5645 vs cover) and they carry independent information.
    # Both models receive both, so each can lean on whichever suits its label.
    elo_by_k = {}
    train_elo_by_k = {}
    eval_elo_by_k  = {}
    for k in k_values:
        er = elo_mod.compute(games_df, k=k, window=_ELO_WINDOW)
        er = _attach_ats_elo(er, games_df, k)
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
    # Must match _CTX_COLS in _collect_window — all context columns that the
    # training feature matrix should contain.  Expanding this beyond ["home","is_b2b"]
    # is the fix for pitcher/bullpen/matchup stats being absent from training data.
    # bp_ip_14d is included here as a NaN placeholder; _apply_bp overwrites it.
    _CTX = [
        "home", "is_b2b",
        "sp_era", "sp_whip", "sp_k9", "sp_ip_per_start",
        "opp_sp_era", "opp_sp_whip", "sp_era_edge",
        "sp_era_rolling", "opp_sp_era_rolling", "sp_era_rolling_edge",
        "bp_era", "bp_whip", "bp_k9", "bp_hr9", "bp_ip_per_game",
        "opp_bp_era", "opp_bp_whip", "bp_era_edge",
        "ml_implied_prob",
        "bp_ip_14d",
    ]

    def _add_elo(df_base, common, elo_by_p, target):
        """Return (elo_diff, opponent_elo, ats_elo_diff, ats_opp_elo)."""
        _nan = np.full(len(common), np.nan)
        elo_grp = elo_by_p.get(target)
        if elo_grp is None:
            return _nan, _nan, _nan, _nan
        ea = elo_grp.reindex(common)
        return (
            ea["elo_diff"].values,
            ea["opp_elo"].values,
            ea["ats_elo_diff"].values if "ats_elo_diff" in ea.columns else _nan,
            ea["ats_opp_elo"].values  if "ats_opp_elo"  in ea.columns else _nan,
        )

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
                _ctx_al = ctx_grp.reindex(common) if ctx_grp is not None else None
                for col in _CTX:
                    base[col] = _ctx_al[col].values if (_ctx_al is not None and col in _ctx_al.columns) else np.nan
                # style_edge deliberately omitted — see _USE_STYLE_EDGE.
                y_cov_tr, y_win_tr = _cover_and_win_labels(y_tr, _ctx_al)
                for k in k_values:
                    df_k = base.copy()
                    (df_k["elo_diff"], df_k["opponent_elo"],
                     df_k["ats_elo_diff"], df_k["ats_opp_elo"]) = _add_elo(
                        base, common, train_elo_by_k_p[k], target)
                    _recast_categoricals(df_k)
                    for col in df_k.columns:
                        if col not in _CAT_COLS:
                            df_k[col] = pd.to_numeric(df_k[col], errors="coerce")
                    train_feats_by_k_target[k][target] = (df_k, y_cov_tr, y_win_tr, common)

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
                _ctx_al = ctx_grp.reindex(common) if ctx_grp is not None else None
                for col in _CTX:
                    base[col] = _ctx_al[col].values if (_ctx_al is not None and col in _ctx_al.columns) else np.nan
                # style_edge deliberately omitted — see _USE_STYLE_EDGE.
                y_cov_ev, y_win_ev = _cover_and_win_labels(y_ev, _ctx_al)
                for k in k_values:
                    df_k = base.copy()
                    (df_k["elo_diff"], df_k["opponent_elo"],
                     df_k["ats_elo_diff"], df_k["ats_opp_elo"]) = _add_elo(
                        base, common, eval_elo_by_k_p[k], target)
                    _recast_categoricals(df_k)
                    for col in df_k.columns:
                        if col not in _CAT_COLS:
                            df_k[col] = pd.to_numeric(df_k[col], errors="coerce")
                    eval_feats_by_k_target[k][target] = (df_k, y_cov_ev, y_win_ev, common)

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

        X_train_parts, y_train_parts, w_train_parts = [], [], []
        X_test_parts,  y_test_parts,  w_test_parts  = [], [], []
        X_val_parts,   y_val_parts,   w_val_parts   = [], [], []

        for start in range(1, next_period - lookback + 1):
            target = start + lookback
            if target in train_feats and train_n_prior.get(target, 0) >= lookback:
                df, y, w, _ = train_feats[target]
                X_train_parts.append(df)
                y_train_parts.append(y)
                w_train_parts.append(w)
            if target in eval_feats and eval_n_prior.get(target, 0) >= lookback:
                df, y, w, _ = eval_feats[target]
                if target < eval_split_period:
                    X_test_parts.append(df)
                    y_test_parts.append(y)
                    w_test_parts.append(w)
                else:
                    X_val_parts.append(df)
                    y_val_parts.append(y)
                    w_val_parts.append(w)

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

        # Win labels ride alongside as an 8th return value so existing callers
        # that unpack seven items keep working.
        def _cat_y(parts):
            return (pd.concat(parts, ignore_index=True).astype(float)
                    if parts else pd.Series(dtype=float))
        win_labels = {
            "train": _cat_y(w_train_parts),
            "test":  _cat_y(w_test_parts),
            "val":   _cat_y(w_val_parts),
        }
        return X_train, X_test, y_train, y_test, X_val, y_val, style_model, win_labels

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

        elo_ratings = _attach_ats_elo(
            elo_mod.compute(games_df, k=best_k, window=_ELO_WINDOW), games_df, best_k)
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

    # Slow path: _collect_window does not derive win labels, so the win model
    # is unavailable here. Returned empty for a consistent signature; callers
    # skip win training when these are empty. In practice _precompute always
    # populates the fast path above.
    _empty_win = {"train": pd.Series(dtype=float),
                  "test":  pd.Series(dtype=float),
                  "val":   pd.Series(dtype=float)}
    return X_train, X_test, y_train, y_test, X_val, y_val, style_model, _empty_win


def _recast_categoricals(df: pd.DataFrame) -> None:
    """Re-cast _CAT_COLS to category dtype in-place (lost after pd.concat)."""
    for col in _CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")


def _american_to_raw_prob(ml) -> float:
    """
    Convert American moneyline odds to raw implied probability (includes vig).

    Examples
    --------
    -150  →  150 / 250  = 0.600
    +130  →  100 / 230  = 0.435
    """
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(ml):
        return float("nan")
    if ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    else:
        return 100.0 / (ml + 100.0)


def _compute_context(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(team, season, period) context features:
      home              : 1 if playing at home, 0 if away
      is_b2b            : 1 if the team played yesterday, else 0
      spread            : game spread line (NaN if unavailable)
      sp_era            : starting pitcher ERA (prior season)
      sp_whip           : starting pitcher WHIP (prior season)
      sp_k9             : starting pitcher K/9 (prior season)
      opp_sp_era        : opponent's starting pitcher ERA (same period)
      opp_sp_whip       : opponent's starting pitcher WHIP (same period)
      sp_era_edge       : opp_sp_era - sp_era (positive = our starter is better)
      bp_era            : team bullpen ERA (prior season)
      bp_whip           : team bullpen WHIP (prior season)
      bp_k9             : team bullpen K/9 (prior season)
      bp_hr9            : team bullpen HR/9 (prior season)
      opp_bp_era        : opponent bullpen ERA (same period)
      opp_bp_whip       : opponent bullpen WHIP (same period)
      bp_era_edge       : opp_bp_era - bp_era (positive = our bullpen is better)
      ml_implied_prob   : no-vig implied probability from run-line moneyline
      bp_ip_14d         : rolling N-day bullpen fatigue (filled separately if seeded)

    Returns a DataFrame indexed by (team, season, period).
    """
    # sp_era / sp_whip / sp_k9 populated for MLB once pitcher stats are seeded
    # bp_era / bp_whip / bp_k9 / bp_hr9 populated for MLB bullpen stats
    # ml_implied_prob computed from stored moneyline (no-vig, both sides normalised)
    _PITCHER_COLS = ["sp_era", "sp_whip", "sp_k9", "sp_ip_per_start"]
    _BULLPEN_COLS = ["bp_era", "bp_whip", "bp_k9", "bp_hr9", "bp_ip_per_game"]
    # Prices are loaded for ml_implied_prob but not output as raw features.
    # spread_juice is the correctly-labelled run-line price; moneyline is the
    # legacy overloaded column kept only as a fallback for un-migrated rows.
    _CTX_ODDS     = ["spread", "spread_juice", "moneyline"] + _PITCHER_COLS + _BULLPEN_COLS
    _CTX_FEAT     = ["spread"] + _PITCHER_COLS + _BULLPEN_COLS  # features only

    needed = {"team", "season", "period", "date", "home"}
    _matchup_cols = [
        "opp_sp_era", "opp_sp_whip", "sp_era_edge",
        "sp_era_rolling", "opp_sp_era_rolling", "sp_era_rolling_edge",
        "opp_bp_era", "opp_bp_whip", "bp_era_edge",
        "ml_implied_prob",
        "bp_ip_14d",   # rolling N-day bullpen fatigue (seeded separately)
    ]
    _all_out = ["home", "is_b2b"] + _CTX_FEAT + _matchup_cols

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
    # Build a lookup: (team, season, period) -> sp_era, sp_whip, bp_era, bp_whip
    pitchers_have = [c for c in ["sp_era", "sp_whip"] if c in df.columns]
    bullpen_have  = [c for c in ["bp_era", "bp_whip"] if c in df.columns]
    lookup_cols   = pitchers_have + bullpen_have

    if opp_present and lookup_cols:
        matchup_lookup = df.set_index(["team", "season", "period"])[lookup_cols]
        opp_keys = list(zip(df["opponent"], df["season"], df["period"]))
        opp_vals = matchup_lookup.reindex(opp_keys)
        opp_vals.index = df.index

        # MLB starter matchup edges (lower ERA = better pitcher)
        if "sp_era" in pitchers_have:
            df["opp_sp_era"]  = opp_vals["sp_era"].values
            df["sp_era_edge"] = df["opp_sp_era"] - df["sp_era"]   # positive = our starter is better
        if "sp_whip" in pitchers_have:
            df["opp_sp_whip"] = opp_vals["sp_whip"].values

        # MLB bullpen matchup edges (lower ERA = better bullpen)
        if "bp_era" in bullpen_have:
            df["opp_bp_era"]  = opp_vals["bp_era"].values
            df["bp_era_edge"] = df["opp_bp_era"] - df["bp_era"]   # positive = our bullpen is better
        if "bp_whip" in bullpen_have:
            df["opp_bp_whip"] = opp_vals["bp_whip"].values
    else:
        for c in _matchup_cols:
            df[c] = np.nan

    # --- Run-line implied probability (no-vig) ---
    # Look up the opponent's price for the same game and normalise both sides
    # to strip the vig. 0.5 = market sees a coin flip, >0.5 = favoured to cover.
    #
    # Prefer spread_juice. The legacy `moneyline` column is NOT reliably the
    # run-line price: ESPN-seeded seasons stored the H2H moneyline there, so
    # this feature silently meant P(win) for some seasons and P(cover) for
    # others. spread_juice is written only by sources known to price the run
    # line, so falling back to `moneyline` is a last resort for rows predating
    # the split.
    _price_col = "spread_juice" if "spread_juice" in df.columns else "moneyline"
    if _price_col in df.columns and opp_present:
        price = df[_price_col]
        if _price_col == "spread_juice" and "moneyline" in df.columns:
            price = price.fillna(df["moneyline"])

        tmp = df.assign(_price=price)
        ml_lookup = tmp.set_index(["team", "season", "period"])["_price"]
        opp_ml = ml_lookup.reindex(
            list(zip(df["opponent"], df["season"], df["period"]))
        )
        opp_ml.index = df.index

        p_self = price.apply(_american_to_raw_prob)
        p_opp  = opp_ml.apply(_american_to_raw_prob)
        total  = p_self + p_opp
        # Avoid division by zero when both sides are NaN
        df["ml_implied_prob"] = (p_self / total).where(total > 0)
    else:
        df["ml_implied_prob"] = np.nan

    # --- Rolling bullpen fatigue (bp_ip_14d) ---
    # Computed directly here using the fixed window so it flows through the
    # same single code path as every other context feature.
    _bp_fat  = _compute_bp_fatigue(games_df, _BP_FATIGUE_DAYS)
    _bp_dict = _bp_fat.to_dict()
    df["bp_ip_14d"] = [
        _bp_dict.get((t, s, p), np.nan)
        for t, s, p in zip(df["team"], df["season"], df["period"])
    ]

    # --- Rolling 5-start SP ERA ---
    # Per-pitcher rolling ERA computed from in-season boxscore data (sp_ip_game,
    # sp_er_game). Falls back to prior-season sp_era for first 2 starts or when
    # boxscore data hasn't been seeded. This replaces the static prior-season
    # value as the primary starter quality signal.
    _sp_roll     = _compute_sp_rolling_era(games_df, n_starts=5, fallback_col="sp_era")
    _sp_roll_dict = _sp_roll.to_dict()
    df["sp_era_rolling"] = [
        _sp_roll_dict.get((t, s, p), np.nan)
        for t, s, p in zip(df["team"], df["season"], df["period"])
    ]

    # Opponent rolling ERA and edge
    if opp_present:
        opp_keys = list(zip(df["opponent"], df["season"], df["period"]))
        df["opp_sp_era_rolling"] = [_sp_roll_dict.get(k, np.nan) for k in opp_keys]
        df["sp_era_rolling_edge"] = (
            df["opp_sp_era_rolling"].astype(float) - df["sp_era_rolling"].astype(float)
        )
    else:
        df["opp_sp_era_rolling"] = np.nan
        df["sp_era_rolling_edge"] = np.nan

    # Clip implausible ERA values — a prior-season ERA > 10 indicates a pitcher
    # with fewer than ~10 IP (e.g. 4 ER in 1 IP relief = 36.00). These are not
    # meaningful starter quality signals. Null them out so XGBoost treats them
    # as missing rather than as extreme numeric inputs.
    for _era_col in ["sp_era", "opp_sp_era", "bp_era", "opp_bp_era"]:
        if _era_col in df.columns:
            df.loc[df[_era_col] > 10, _era_col] = np.nan
            df.loc[df[_era_col] == 0, _era_col] = np.nan

    out_cols = [c for c in _all_out if c in df.columns]
    return df.set_index(["team", "season", "period"])[out_cols]


def _compute_sp_rolling_era(
    games_df: pd.DataFrame,
    n_starts: int = 5,
    fallback_col: str = "sp_era",
) -> pd.Series:
    """
    Compute rolling n-start ERA for each starting pitcher.

    For each game row, looks back at the previous `n_starts` appearances by
    the same pitcher in the same season and computes ERA = sum(ER)/sum(IP)*9.
    Requires 'sp_ip_game' and 'sp_er_game' columns (seeded by --sp-game-stats).

    Falls back to prior-season `sp_era` when:
      - fewer than 2 starts of in-season data are available (season opener)
      - sp_ip_game / sp_er_game not present in games_df

    Returns a Series indexed by (team, season, period) named 'sp_era_rolling'.
    """
    idx = pd.MultiIndex.from_frame(games_df[["team", "season", "period"]])
    result = pd.Series(np.nan, index=idx, name="sp_era_rolling")

    if "sp_ip_game" not in games_df.columns or "sp_er_game" not in games_df.columns:
        # Fall back to prior-season ERA for all rows
        if fallback_col in games_df.columns:
            result = games_df.set_index(["team", "season", "period"])[fallback_col].rename("sp_era_rolling")
        return result

    # Work on rows that have a sp_name and sp_ip_game
    df = games_df[["team", "season", "period", "date", "sp_name",
                   "sp_ip_game", "sp_er_game"]].copy()
    if fallback_col in games_df.columns:
        df[fallback_col] = games_df[fallback_col].values

    df = df.sort_values(["sp_name", "season", "date"])
    df["sp_ip_game"] = pd.to_numeric(df["sp_ip_game"], errors="coerce")
    df["sp_er_game"] = pd.to_numeric(df["sp_er_game"], errors="coerce")

    rolling_eras = {}   # (team, season, period) -> rolling ERA

    for (sp_name, season_val), grp in df.groupby(["sp_name", "season"]):
        if not sp_name:
            continue
        grp = grp.sort_values("date").reset_index(drop=True)

        for i, row in grp.iterrows():
            key = (row["team"], season_val, int(row["period"]))
            # Prior starts in this season for this pitcher (exclude current game)
            prior = grp.iloc[:i].dropna(subset=["sp_ip_game", "sp_er_game"])
            prior = prior[prior["sp_ip_game"] > 0]

            if len(prior) >= 2:
                # Use last n_starts starts (or all available if fewer)
                window = prior.tail(n_starts)
                total_ip = window["sp_ip_game"].sum()
                total_er = window["sp_er_game"].sum()
                if total_ip > 0:
                    rolling_eras[key] = round(total_er / total_ip * 9, 3)
                    continue

            # Fallback: use prior-season ERA
            if fallback_col in row.index and pd.notna(row[fallback_col]):
                rolling_eras[key] = row[fallback_col]

    for key, val in rolling_eras.items():
        if key in result.index:
            result[key] = val

    return result


def _compute_bp_fatigue(games_df: pd.DataFrame, days: int) -> pd.Series:
    """
    Compute rolling bullpen fatigue: sum of bp_ip_game over the `days` calendar
    days immediately before each game (current game's IP excluded).

    Returns a Series indexed by (team, season, period) named 'bp_ip_14d'.
    Returns all-NaN if 'bp_ip_game' is not present in games_df.
    """
    idx = pd.MultiIndex.from_frame(games_df[["team", "season", "period"]])
    if "bp_ip_game" not in games_df.columns:
        return pd.Series(np.nan, index=idx, name="bp_ip_14d")

    fat = games_df[["team", "season", "date", "period", "bp_ip_game"]].copy()
    fat["date"] = pd.to_datetime(fat["date"])
    fat = fat.sort_values(["team", "season", "date"])
    fat_rows = []
    for (team, season_val), grp in fat.groupby(["team", "season"]):
        grp = grp.set_index("date").sort_index()
        rolling = (
            grp["bp_ip_game"]
            .shift(1, freq="D")
            .rolling(f"{days}D")
            .sum()
        )
        for period, val in zip(grp["period"], rolling.values):
            fat_rows.append({
                "team": team, "season": season_val,
                "period": int(period), "bp_ip_14d": val,
            })
    if not fat_rows:
        return pd.Series(np.nan, index=idx, name="bp_ip_14d")
    fat_df = pd.DataFrame(fat_rows).set_index(["team", "season", "period"])
    return fat_df["bp_ip_14d"]


def compute_bp_fatigue_for_date(
    season_games: pd.DataFrame,
    game_date: str,
    days: int,
) -> dict[str, float]:
    """
    Compute pre-game bullpen fatigue for an upcoming game on `game_date`.

    Unlike _compute_bp_fatigue (which is indexed by period and only covers
    games already in the DB), this function uses the actual game date so it
    works correctly for prediction — the upcoming game has no period entry yet.

    Returns a dict: team -> sum of bp_ip_game over the `days` calendar days
    strictly before `game_date`.  Teams with no bp_ip_game data return NaN.
    """
    if "bp_ip_game" not in season_games.columns:
        return {}

    cutoff = pd.Timestamp(game_date)
    window_start = cutoff - pd.Timedelta(days=days)

    fat = season_games[["team", "date", "bp_ip_game"]].copy()
    fat["date"] = pd.to_datetime(fat["date"])
    fat = fat.dropna(subset=["bp_ip_game"])

    # Games strictly before game_date and within the rolling window
    mask = (fat["date"] >= window_start) & (fat["date"] < cutoff)
    recent = fat[mask]

    if recent.empty:
        return {}

    return recent.groupby("team")["bp_ip_game"].sum().to_dict()


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
    _CTX_COLS = [
        "home", "is_b2b",
        "sp_era", "sp_whip", "sp_k9", "sp_ip_per_start",
        "opp_sp_era", "opp_sp_whip", "sp_era_edge",
        "sp_era_rolling", "opp_sp_era_rolling", "sp_era_rolling_edge",
        "bp_era", "bp_whip", "bp_k9", "bp_hr9", "bp_ip_per_game",
        "opp_bp_era", "opp_bp_whip", "bp_era_edge",
        "ml_implied_prob",
        "bp_ip_14d",
    ]
    try:
        ctx_slice   = context.xs(target, level="period")
        ctx_aligned = ctx_slice.reindex(common_idx)
        for col in _CTX_COLS:
            df[col] = ctx_aligned[col].values if col in ctx_aligned.columns else np.nan
    except KeyError:
        for col in _CTX_COLS:
            df[col] = np.nan

    # style_edge is intentionally not added to the feature matrix — it
    # measured exactly 0.00 on both SHAP and gain. See _USE_STYLE_EDGE.

    # --- Elo ratings ---
    try:
        elo_slice    = elo_ratings.xs(target, level="period")
        elo_aligned  = elo_slice.reindex(common_idx)
        df["elo_diff"]     = elo_aligned["elo_diff"].values
        df["opponent_elo"] = elo_aligned["opp_elo"].values
        for _src, _dst in (("ats_elo_diff", "ats_elo_diff"),
                           ("ats_opp_elo", "ats_opp_elo")):
            df[_dst] = (elo_aligned[_src].values
                        if _src in elo_aligned.columns else np.nan)
    except KeyError:
        df["elo_diff"]     = np.nan
        df["opponent_elo"] = np.nan

    _recast_categoricals(df)
    for col in df.columns:
        if col not in _CAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    X_out.append(df)
    y_out.append(pd.Series((y.values > 0).astype(int), dtype=float))


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
    # sp_era / sp_whip / sp_k9 / matchup pitcher cols used for MLB
    _CTX_COLS = [
        "home", "is_b2b",
        "sp_era", "sp_whip", "sp_k9", "sp_ip_per_start",
        "opp_sp_era", "opp_sp_whip", "sp_era_edge",
        "sp_era_rolling", "opp_sp_era_rolling", "sp_era_rolling_edge",
        "bp_era", "bp_whip", "bp_k9", "bp_hr9", "bp_ip_per_game",
        "opp_bp_era", "opp_bp_whip", "bp_era_edge",
        "ml_implied_prob",
        "bp_ip_14d",
    ]
    if upcoming_context is not None and not upcoming_context.empty:
        for col in _CTX_COLS:
            X[col] = X["team"].map(upcoming_context[col]) if col in upcoming_context.columns else np.nan
    else:
        for col in _CTX_COLS:
            X[col] = np.nan

    # Elo ratings for upcoming game (pre-game, using all completed games this season).
    # Must mirror _precompute exactly — the model expects both the standard and
    # the ATS rating, and a missing column at prediction time would silently
    # become NaN for every team.
    elo_df = elo_mod.compute(completed, k=best_k, window=_ELO_WINDOW)
    elo_df = _attach_ats_elo(elo_df, completed, best_k)
    _elo_cols = ["elo", "opp_elo", "elo_diff", "ats_elo_diff", "ats_opp_elo"]
    latest_elo = (
        elo_df.reset_index()
        .sort_values("period")
        .groupby("team")
        .last()
        [[c for c in _elo_cols if c in elo_df.columns]]
    )
    X["elo_diff"]     = X["team"].map(latest_elo["elo_diff"])
    X["opponent_elo"] = X["team"].map(latest_elo["opp_elo"])
    X["ats_elo_diff"] = X["team"].map(latest_elo["ats_elo_diff"]) \
        if "ats_elo_diff" in latest_elo.columns else np.nan
    X["ats_opp_elo"]  = X["team"].map(latest_elo["ats_opp_elo"]) \
        if "ats_opp_elo" in latest_elo.columns else np.nan

    # style_edge is not produced. It contributed nothing to the model, and
    # fitting the StyleModel here meant a full SVD of the season's matchup
    # matrix on every prediction run. Note this used to refit even when the
    # caller passed style_model=None. See _USE_STYLE_EDGE.

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
) -> tuple[int, float, int, dict]:
    """
    Choose the best (lookback, K, bp_fatigue_days) combination via 5-fold CV.

    Phase 1 — searches _LOOKBACK_CANDIDATES × _K_CANDIDATES (neg MSE).
    Phase 2 — searches _BP_FATIGUE_CANDIDATES at the winning (lookback, K),
              swapping only the bp_ip_14d column so the rest of the precomputed
              feature matrix is reused without recomputation.

    Falls back to (smallest lookback, K=32, 14 days) if no candidate yields
    enough data.

    Returns
    -------
    best_lookback, best_k, best_bp_days, cache
    """
    print("  Precomputing Elo, style embeddings, and SS pivot...")
    cache = _precompute(games_df, next_period, eval_season, _K_CANDIDATES)

    _xgb_cv = XGBClassifier(**_XGB_FIXED, max_depth=3, n_estimators=100, learning_rate=0.1)

    # ---- Phase 1: lookback × K ----
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
                _xgb_cv, X_train, y_train,
                cv=KFold(n_splits=5, shuffle=False),
                scoring="roc_auc",
            ).mean()
            if score > best_score:
                best_score, best_lb, best_k = score, lb, k
    print(f"  Phase 1 -> lookback={best_lb}, K={best_k}")

    return best_lb, best_k, cache


def _tune(
    model_class: type,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_evals: int,
    seed: int = 42,
) -> dict:
    """Run Hyperopt TPE search; return best params with integer types corrected.

    Parameters
    ----------
    seed : random seed passed to hyperopt's TPE sampler. Use a fixed value
           (default 42) for production reproducibility, or vary it across
           runs to measure true hyperparameter sensitivity.
    """
    scoring = "roc_auc" if model_class is XGBClassifier else "neg_mean_squared_error"

    def objective(params: dict) -> float:
        p = {
            **_XGB_FIXED, **params,
            "max_depth": int(params["max_depth"]),
            "n_estimators": int(params["n_estimators"]),
        }
        return -cross_val_score(
            model_class(**p), X_train, y_train, cv=5, scoring=scoring
        ).mean()

    from hyperopt import Trials
    best = fmin(fn=objective, space=_HYPEROPT_SPACE, algo=tpe.suggest,
                max_evals=max_evals, trials=Trials(), rstate=np.random.default_rng(seed))
    return {**best, "max_depth": int(best["max_depth"]),
            "n_estimators": int(best["n_estimators"])}


def train_models(
    games_df: pd.DataFrame,
    next_period: int,
    eval_season: int,
    eval_split_period: int,
    max_evals: int = 10,
    seed: int = 42,
) -> tuple[XGBClassifier, dict, int, int, int, emb.StyleModel]:
    """
    Select lookback, tune hyperparameters, and fit the binary cover classifier.

    Target: 1 = team covered the run line (spreadscore > 0), 0 = did not cover.
    The classifier directly outputs P(cover) via predict_proba[:,1], which is
    used without further transformation to compute EV against the run-line juice.

    Returns
    -------
    clf           : fitted XGBClassifier (predict_proba[:,1] = P(cover run line))
    scores        : dict of train/test/val evaluation metrics (ROC-AUC, accuracy)
    best_lookback : lookback value selected by CV
    best_k        : Elo K value selected by CV — must be forwarded to
                    build_prediction_features so Elo features match training scale
    style_model   : fitted StyleModel for prediction-time style_edge
    """
    print("  Selecting lookback and K...")
    best_lookback, best_k, cache = _select_hyperparams(
        games_df, next_period, eval_season, eval_split_period
    )
    print(f"  Best lookback: {best_lookback}  Best K: {best_k}")

    X_train, X_test, y_train, y_test, X_val, y_val, style_model, win_labels = build_features(
        games_df, next_period, best_lookback, eval_season, eval_split_period,
        best_k=best_k, _cache=cache,
    )
    print(f"  Rows — train: {len(X_train)}, test: {len(X_test)}, val: {len(X_val)}")

    # Feature fill-rate report — warn on any feature that is mostly NaN in training data.
    # XGBoost handles NaN natively but a feature that is e.g. 30% filled contributes
    # very little signal and indicates a seeding or name-matching problem.
    _WARN_FILL  = 0.70
    _ERROR_FILL = 0.40
    _KEY_FEATS  = ["sp_era", "sp_era_edge", "bp_era", "bp_era_edge",
                   "ml_implied_prob", "bp_ip_14d", "elo_diff"]
    print("  Feature fill rates (training set):")
    _any_warn = False
    for _col in _KEY_FEATS:
        if _col not in X_train.columns:
            continue
        _rate = float(X_train[_col].notna().mean())
        _tag  = "OK   " if _rate >= _WARN_FILL else ("WARN " if _rate >= _ERROR_FILL else "ERROR")
        if _tag != "OK   ":
            _any_warn = True
        print(f"    {_tag}  {_col:22s}: {_rate:5.1%}")
    if not _any_warn:
        print("    All key features OK.")

    print("  Tuning binary cover classifier...")
    clf = XGBClassifier(
        **_XGB_FIXED, random_state=42,
        **_tune(XGBClassifier, X_train, y_train, max_evals, seed=seed)
    )
    clf.fit(X_train, y_train)

    # Evaluation metrics
    from sklearn.metrics import roc_auc_score, accuracy_score
    train_acc = float(accuracy_score(y_train, clf.predict(X_train)))
    try:
        train_roc = float(roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]))
    except ValueError:
        train_roc = float("nan")

    scores = {
        "lookback":      best_lookback,
        "elo_k":         best_k,
        "clf_train_acc": round(train_acc, 3),
        "clf_train_roc": round(train_roc, 3),
    }
    if not X_test.empty:
        try:
            test_roc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
            scores["clf_test_roc"] = round(test_roc, 3)
            scores["clf_test_acc"] = round(float(accuracy_score(y_test, clf.predict(X_test))), 3)
        except ValueError:
            pass
    if not X_val.empty:
        try:
            val_roc = float(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))
            scores["clf_val_roc"] = round(val_roc, 3)
            scores["clf_val_acc"] = round(float(accuracy_score(y_val, clf.predict(X_val))), 3)
        except ValueError:
            pass

    # --- Second target: P(win outright), for the moneyline market ---
    # Same features, different label. Trained on the rows where a win label
    # could be derived (spread present), reusing the cover model's tuned
    # hyperparameters rather than paying for a second hyperopt run.
    win_clf = None
    w_train = win_labels.get("train", pd.Series(dtype=float))
    if len(w_train) == len(X_train) and w_train.notna().any():
        _m = w_train.notna().values
        print(f"  Fitting win classifier on {int(_m.sum())}/{len(X_train)} rows "
              f"with a derivable win label...")
        # get_xgb_params() already carries random_state and the fixed settings,
        # so merge rather than splatting both and colliding on duplicate keys.
        _win_params = dict(clf.get_xgb_params())
        _win_params.update(_XGB_FIXED)
        _win_params["random_state"] = 42
        win_clf = XGBClassifier(**_win_params)
        win_clf.fit(X_train[_m], w_train[_m])

        for split, X_s, w_s in (("test", X_test, win_labels.get("test")),
                                ("val", X_val, win_labels.get("val"))):
            if X_s.empty or w_s is None or len(w_s) != len(X_s):
                continue
            m = w_s.notna().values
            if m.sum() < 50:
                continue
            try:
                scores[f"win_{split}_roc"] = round(float(
                    roc_auc_score(w_s[m], win_clf.predict_proba(X_s[m])[:, 1])), 3)
                scores[f"win_{split}_acc"] = round(float(
                    accuracy_score(w_s[m], win_clf.predict(X_s[m]))), 3)
            except ValueError:
                pass
    else:
        print("  No win labels available — skipping win classifier.")

    # Refit on all seasons (including eval_season) once hyperparams are locked in.
    # eval metrics above are computed before this refit so they remain unbiased.
    all_X_parts = [X_train] + ([X_test] if not X_test.empty else []) + ([X_val] if not X_val.empty else [])
    all_y_parts = [y_train] + ([y_test] if not X_test.empty else []) + ([y_val] if not X_val.empty else [])
    _all_X = pd.concat(all_X_parts, ignore_index=True)
    clf.fit(_all_X, pd.concat(all_y_parts, ignore_index=True))

    if win_clf is not None:
        all_w_parts = [win_labels["train"]] \
            + ([win_labels["test"]] if not X_test.empty else []) \
            + ([win_labels["val"]] if not X_val.empty else [])
        _all_w = pd.concat(all_w_parts, ignore_index=True)
        if len(_all_w) == len(_all_X):
            _m = _all_w.notna().values
            win_clf.fit(_all_X[_m], _all_w[_m])

    return clf, scores, best_lookback, best_k, style_model, win_clf


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

import joblib as _joblib
import datetime as _datetime
import os as _os

_DEFAULT_MODEL_PATH = _os.path.join(_os.path.dirname(__file__), "data", "mlb_model.pkl")


def save_model(
    clf: XGBClassifier,
    scores: dict,
    best_lookback: int,
    best_k: int | float,
    style_model,
    next_period: int | None = None,
    train_seasons: list | None = None,
    path: str = _DEFAULT_MODEL_PATH,
    win_clf: "XGBClassifier | None" = None,
) -> None:
    """
    Persist a trained model bundle to disk so analysis scripts can load it
    without retraining.

    Saves a dict with all artefacts needed to reproduce predictions:
        clf, scores, best_lookback, best_k, best_bp_days (fixed constant),
        style_model, next_period, train_seasons, saved_at.

    Usage
    -----
        model.save_model(clf, scores, best_lookback, best_k,
                         style_model, next_period=30)
        bundle = model.load_model()
    """
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    bundle = {
        "clf":           clf,
        # Second model on the same features predicting P(win outright), used to
        # price the moneyline. None in bundles saved before this existed, so
        # callers must handle its absence.
        "win_clf":       win_clf,
        "scores":        scores,
        "best_lookback": best_lookback,
        "best_k":        best_k,
        "best_bp_days":  _BP_FATIGUE_DAYS,   # fixed — not a hyperparameter
        "style_model":   style_model,
        "next_period":   next_period,
        "train_seasons": train_seasons,
        "saved_at":      _datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _joblib.dump(bundle, path, compress=3)
    print(f"  Model saved -> {path}  ({_os.path.getsize(path) // 1024} KB)")


def load_model(path: str = _DEFAULT_MODEL_PATH) -> dict:
    """
    Load a model bundle saved by save_model().

    Returns a dict with keys:
        clf, scores, best_lookback, best_k, best_bp_days,
        style_model, next_period, train_seasons, saved_at.

    Raises FileNotFoundError if no model has been saved yet.
    """
    if not _os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at {path}. "
            "Run _kelly_analysis.py to train and save the model first."
        )
    bundle = _joblib.load(path)
    age = _datetime.datetime.now() - _datetime.datetime.fromisoformat(bundle["saved_at"])
    hours = age.total_seconds() / 3600
    print(f"  Loaded model from {path}  (saved {hours:.1f}h ago, "
          f"next_period={bundle.get('next_period')}, "
          f"seasons={bundle.get('train_seasons')})")
    return bundle


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    clf: XGBClassifier,
    X_pred: pd.DataFrame,
    win_clf: "XGBClassifier | None" = None,
) -> pd.DataFrame:
    """
    Generate probabilities for both markets, indexed by team.

    Each classifier outputs its probability directly — no pairing or normal-CDF
    transformation is needed.

    Parameters
    ----------
    clf     : fitted cover classifier (from train_models / load_model)
    X_pred  : feature matrix with a `team` column (from build_prediction_features)
    win_clf : optional win classifier. Absent in bundles saved before the
              dual-target change, in which case win_prob is NaN and callers
              fall back to spread-only betting.

    Returns
    -------
    DataFrame indexed by team with columns:
        coverprob : P(team covers the run line)
        win_prob  : P(team wins outright), or NaN if no win model
    """
    feat_cols = clf.get_booster().feature_names
    coverprob = clf.predict_proba(X_pred[feat_cols])[:, 1]
    teams     = X_pred["team"].values

    out = pd.DataFrame({"coverprob": coverprob}, index=teams)

    if win_clf is not None:
        win_cols = win_clf.get_booster().feature_names
        out["win_prob"] = win_clf.predict_proba(X_pred[win_cols])[:, 1]
    else:
        out["win_prob"] = np.nan
    return out
