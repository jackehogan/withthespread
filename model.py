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
# Elo K is no longer searched. Walk-forward validation over 2023-2026 (~8,000
# out-of-sample games) found per-fold K selection performed no better than a
# single fixed value, and two independent criteria -- log loss and the
# realised-vs-claimed slope ratio -- both landed on 6. The list is kept at
# length one so the surrounding search plumbing and the saved bundle format
# stay unchanged. See elo.py for the derivation.
_K_CANDIDATES            = [16.0]
# Second Elo at a faster K, giving the tree two memory horizons instead of one.
# The pre-rebuild model carried two (window 20 and window 40) and scored
# win_val_roc 0.615; collapsing to a single rating dropped it to 0.562. This is
# the ablation for whether the second horizon was what mattered. K=24 half-lives
# in roughly 20 games against K=6's 80. Set to None to disable.
_FAST_ELO_K              = 24.0

# How a feature's matchup information is presented to the tree.
#   "edge" : own value + (opponent - own), the differenced form
#   "opp"  : own value + the opponent's raw value
#   "both" : all three columns
# edge = opp - own exactly, so any two of the three span the same space -- but a
# tree splits on axis-aligned thresholds, so which two are supplied changes the
# splits available. A per-feature test on held-out 2025-26 found own+opp beat
# own+edge in 3 of 4 features, while the full-model test found the edges worth
# +0.045 out of sample. Those are in tension: the edges may have helped because
# four form features had NO opponent counterpart at all, in which case opp
# mirrors would do the same job. This switch exists to settle it.
_MATCHUP_MODE = "opp"

# The three edges that predate this work. They were hard-coded into
# _matchup_cols, so the mode switch did not govern them and "opp" mode was
# really own+opp PLUS three stray edges. Routed through the switch so the mode
# means what it says.
_LEGACY_EDGE_FEATURES = ["sp_era_edge", "sp_era_rolling_edge", "bp_era_edge"]

_EDGE_FEATURES = _LEGACY_EDGE_FEATURES + [
    "sp_whip_edge", "sp_k9_edge", "sp_ip_per_start_edge",
    "bp_whip_edge", "bp_k9_edge", "bp_hr9_edge", "bp_ip_per_game_edge",
    "1_ago_diff_edge", "diff_mean_5_edge", "win_streak_edge", "loss_streak_edge",
]
# Mirrors for the same features. sp_era/sp_whip/bp_era/bp_whip already have an
# opp_ column built earlier, so they are not repeated here.
_OPP_FEATURES = [
    "opp_bp_ip_14d", "opp_is_b2b",
    "opp_sp_k9", "opp_sp_ip_per_start",
    "opp_bp_k9", "opp_bp_hr9", "opp_bp_ip_per_game",
    "opp_1_ago_diff", "opp_diff_mean_5", "opp_win_streak", "opp_loss_streak",
]


# Set False to build without the in-season bullpen rates and the 30-day
# workload. Used with _MATCHUP_MODE="none" to reconstruct a leak-free
# baseline, so this session's additions can be measured against something
# rather than against a number that was itself inflated by the opponent
# lookup bug.
_BP_ROLLING = True

_BP_ROLLING_FEATURES = ["bp_era_rolling", "bp_whip_rolling",
                        "bp_k9_rolling", "bp_pitch_30d"]
_BP_ROLLING_OPP = ["opp_bp_era_rolling", "opp_bp_whip_rolling",
                   "opp_bp_k9_rolling", "opp_bp_pitch_30d"]


def _bp_rolling_features() -> list:
    return list(_BP_ROLLING_FEATURES) if _BP_ROLLING else []


def _matchup_extra_features() -> list:
    """Feature names added by the current _MATCHUP_MODE."""
    if _MATCHUP_MODE == "none":
        return []
    _bp = list(_BP_ROLLING_OPP) if _BP_ROLLING else []
    if _MATCHUP_MODE == "opp":
        return list(_OPP_FEATURES) + _bp
    if _MATCHUP_MODE == "both":
        return list(_EDGE_FEATURES) + list(_OPP_FEATURES) + _bp
    return list(_EDGE_FEATURES)
# EXPERIMENT ONLY (branch elo-revert-test): reproduce the pre-rebuild Elo
# feature set exactly -- elo_diff at the league-wide window 20, plus a second
# rating at window 40 standing in for long_elo_diff/long_opp_elo -- to test
# whether reverting the rating recovers win_val_roc 0.615.
_REVERT_TEST_WINDOW      = 40
_BP_FATIGUE_DAYS         = 14    # fixed rolling window for bp_ip_14d — not a hyperparameter

# Bullpen workload window and rate shrinkage, both swept on 2022-2026 and scored
# ORTHOGONALISED against the Elo edge, so these are gains over what a team
# rating already knows.
#
# Workload: 30 days of PITCH COUNT, ex-Elo 0.5432. Pitch count beat innings at
# every window (0.5432 vs 0.5286), and long beat short everywhere -- 1-day and
# 3-day workload score BELOW 0.500 ex-Elo, i.e. worse than nothing. That means
# this is not measuring fatigue: a bullpen that has thrown a lot over a month is
# one whose rotation is not going deep, so it is a rotation-depth proxy. Genuine
# short-term rest carries no signal here, plausibly because a gassed reliever
# simply does not appear.
_BP_WORKLOAD_DAYS        = 30

# Rate shrinkage: w = season_ip / (season_ip + lambda). lambda=200 means the
# bullpen's own season does not outweigh last year's until team game ~55 of 162,
# and last season still holds 25% in October. Four times the starters' lambda,
# because a bullpen is a dozen arms in shifting roles -- more innings, less
# signal per inning. K/9 is the most stable and wants the most evidence.
_BP_RATE_LAMBDA          = {"era": 200.0, "whip": 200.0, "k9": 400.0}

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

# The model does not see any market price. It forms a view from team ratings
# and box-score history alone, and the price is used only to size and select
# the bet afterwards.
#
# Rationale, measured out of sample on 2026:
#   * With ml_implied_prob included, a positive EV partly reflects the model
#     failing to reproduce a price it was shown, which is uninterpretable.
#   * Removing it costs the WIN model almost nothing (AUC .5808 -> .5699,
#     against the market's own .5650) because that feature is derived from the
#     run-line price and carried only 2.45% of the win model's gain.
#   * Moneyline ROI at the 0.05 threshold: 11.7% with the feature, 10.3%
#     without. ~1.4 points buys EV that actually means something.
#
# _compute_context still computes ml_implied_prob — predict_mlb reports it for
# comparison — it is simply excluded from the feature matrix.
_MARKET_BLIND = True
_MARKET_FEATURES = ("ml_implied_prob", "spread")

# MLB bets the moneyline only -- predict_mlb sets bet="ML" unconditionally and
# prices it from win_prob -- so the cover classifier is trained and saved for
# nothing. Set False to skip fitting it; `coverprob` then comes back NaN and
# callers fall back to the win model.
#
# Keep this True for a spread sport. The spreadscore pivot, the SS-derived
# features and _SPREAD_DERIVED_FEATURES are all left in place precisely so the
# cover target can be switched back on without rebuilding them.
#
# Note the hyperopt search still optimises the COVER label either way, and
# win_clf inherits those hyperparameters. That is unchanged here deliberately,
# so this flag alters what is fitted and not how anything is tuned; retuning on
# the win label is a separate change worth measuring on its own.
_TRAIN_COVER_MODEL = False

# Features built on spreadscore ( = diff + spread ). The run line's sign is the
# book's favourite/underdog call, so these carry market information even though
# _MARKET_BLIND drops `spread` and `ml_implied_prob` as explicit columns. They
# are correct for the COVER target and structurally wrong for the WIN target:
# ss_mean_5 sums a performance term and a market term that point opposite ways,
# and scores AUC 0.4992 against winning -- a coin flip -- while its components
# score 0.5272 and 0.5549 separately.
#
# The win model is therefore fit without them, on diff-based equivalents
# instead. Measured on 2026 (trained 2022-2025): dropping them outright costs
# 0.0087 AUC (.5737 -> .5650); replacing them with the diff versions recovers
# .5729, i.e. the signal was performance, not market. Adding recent-favourite
# rate back explicitly gains nothing (.5730), confirming the market term is
# already carried by Elo.
_SPREAD_DERIVED_FEATURES = (
    "1_ago_ss", "ss_mean_5", "cover_streak", "fade_streak",
)

# Innings of in-season work at which season-to-date ERA and last season's ERA
# carry equal weight. Tested over 15,818 starts at lambda 10/25/50/100;
# 25-50 was flat-optimal, 10 chased noise and 100 clung to stale data.
_SP_ERA_BLEND_LAMBDA = 50.0


def _cover_and_win_labels(ss_vals, ctx_aligned) -> tuple[pd.Series, pd.Series]:
    """
    Derive both prediction targets from the same rows.

        cover = spreadscore > 0     (did the team beat the run line)
        win   = diff > 0            (did the team win the game)

    `diff` is the raw run differential and is read straight from the context
    frame, which carries it for exactly this purpose. Winning outright has
    nothing to do with the run line, so the win label must not depend on
    `spread` being present.

    The legacy route reconstructed it as `spreadscore - spread`, inverting
    seed_mlb's `spreadscore = diff + spread`. That was a carry-over from
    sports where the line moves and only the spread-relative result was
    stored. It returns the same number when `spread` is present and NaN when
    it is not, silently dropping those rows from win training. It is kept
    below as a fallback for context frames built before `diff` was carried.

    Win is NaN only where the game result itself is unavailable.
    """
    ss = np.asarray(ss_vals, dtype=float)
    cover = pd.Series((ss > 0).astype(float), dtype=float)

    diff = None
    if ctx_aligned is not None and "diff" in ctx_aligned.columns:
        diff = pd.to_numeric(ctx_aligned["diff"], errors="coerce").values
    if diff is None or np.isnan(diff).all():
        if ctx_aligned is not None and "spread" in ctx_aligned.columns:
            spread = pd.to_numeric(ctx_aligned["spread"], errors="coerce").values
            diff = ss - spread
        else:
            diff = np.full(len(ss), np.nan)

    win = pd.Series(np.where(np.isnan(diff), np.nan, (diff > 0).astype(float)),
                    dtype=float)
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
        er = elo_mod.compute(games_df, k=k)
        if _FAST_ELO_K is not None:
            _fast = elo_mod.compute(games_df, k=k, window=_REVERT_TEST_WINDOW)
            er = er.join(_fast[["elo_diff", "opp_elo"]]
                         .rename(columns={"elo_diff": "fast_elo_diff",
                                          "opp_elo": "fast_opp_elo"}))
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
        "opp_sp_era", "opp_sp_whip",
        "sp_era_rolling", "opp_sp_era_rolling",
        "bp_era", "bp_whip", "bp_k9", "bp_hr9", "bp_ip_per_game",
        "opp_bp_era", "opp_bp_whip",
        "ml_implied_prob",
        "bp_ip_14d",
        "1_ago_diff", "diff_mean_5", "win_streak", "loss_streak",
    ] + _bp_rolling_features() + _matchup_extra_features()

    if _MARKET_BLIND:
        _CTX = [c for c in _CTX if c not in _MARKET_FEATURES]

    def _add_elo(df_base, common, elo_by_p, target):
        """Return (elo_diff, opponent_elo, fast_elo_diff, fast_opp_elo)."""
        _nan = np.full(len(common), np.nan)
        elo_grp = elo_by_p.get(target)
        if elo_grp is None:
            return _nan, _nan, _nan, _nan
        ea = elo_grp.reindex(common)
        return (
            ea["elo_diff"].values,
            ea["opp_elo"].values,
            ea["fast_elo_diff"].values if "fast_elo_diff" in ea.columns else _nan,
            ea["fast_opp_elo"].values if "fast_opp_elo" in ea.columns else _nan,
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
                     df_k["fast_elo_diff"], df_k["fast_opp_elo"]) = _add_elo(
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
                     df_k["fast_elo_diff"], df_k["fast_opp_elo"]) = _add_elo(
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
        # (team, season, period) per row, so predictions can be joined back to
        # prices and outcomes. Without this a backtest cannot price its bets.
        k_train, k_test, k_val = [], [], []

        def _keys(common, target):
            return [(t, s, target) for (t, s) in common]

        for start in range(1, next_period - lookback + 1):
            target = start + lookback
            if target in train_feats and train_n_prior.get(target, 0) >= lookback:
                df, y, w, common = train_feats[target]
                X_train_parts.append(df)
                y_train_parts.append(y)
                w_train_parts.append(w)
                k_train.extend(_keys(common, target))
            if target in eval_feats and eval_n_prior.get(target, 0) >= lookback:
                df, y, w, common = eval_feats[target]
                if target < eval_split_period:
                    X_test_parts.append(df)
                    y_test_parts.append(y)
                    w_test_parts.append(w)
                    k_test.extend(_keys(common, target))
                else:
                    X_val_parts.append(df)
                    y_val_parts.append(y)
                    w_val_parts.append(w)
                    k_val.extend(_keys(common, target))

        if not X_train_parts:
            raise ValueError(
                f"No training windows found (next_period={next_period}, lookback={lookback})."
            )

        def _concat(parts, y_parts):
            X = pd.concat(parts, ignore_index=True)
            _recast_categoricals(X)
            X = _apply_feature_trim(X)
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
            # Row identity, aligned positionally with each split's X and y.
            "keys_train": k_train,
            "keys_test":  k_test,
            "keys_val":   k_val,
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

        elo_ratings = elo_mod.compute(games_df, k=best_k)
        if _FAST_ELO_K is not None:
            _f = elo_mod.compute(games_df, k=best_k, window=_REVERT_TEST_WINDOW)
            elo_ratings = elo_ratings.join(
                _f[["elo_diff", "opp_elo"]].rename(
                    columns={"elo_diff": "fast_elo_diff", "opp_elo": "fast_opp_elo"}))
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
        X = _apply_feature_trim(X)
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


# ---------------------------------------------------------------------------
# Feature trim
# ---------------------------------------------------------------------------
# Chosen by backward elimination in _trim_features.py, scored by
# leave-one-season-out CV inside the TRAINING seasons only; 2026 was read once
# at the end so it stays an honest holdout. Features were dropped in mirrored
# pairs -- half a matchup is not a set anyone would ship.
#
#   full 46 features : CV 0.6366   2026 holdout 0.6126
#   trimmed 16       : CV 0.6443   2026 holdout 0.6158
#
# The holdout gain is inside one SE, so this is not a performance win. It is a
# large simplification for no cost, and it shrinks the surface hyperopt selects
# over. Note what survived: the in-season blends (sp_era_rolling,
# bp_whip_rolling) kept their prior-season counterparts out, which is coherent
# -- a blend already shrinks toward the prior season, so carrying both was
# duplication.
#
# Set to None to disable the trim and train on everything.
_KEEP_FEATURES: "list[str] | None" = [
    "1_ago_diff",
    "bp_pitch_30d",
    "bp_whip_rolling",
    "elo_diff",
    "fast_elo_diff",
    "home",
    "is_b2b",
    "loss_streak",
    "opp_1_ago_diff",
    "opp_bp_pitch_30d",
    "opp_bp_whip_rolling",
    "opp_is_b2b",
    "opp_loss_streak",
    "opp_sp_era_rolling",
    "opponent_elo",
    "sp_era_rolling",
]


def _apply_feature_trim(X: pd.DataFrame) -> pd.DataFrame:
    """Restrict X to _KEEP_FEATURES, preserving identifier columns."""
    if _KEEP_FEATURES is None:
        return X
    keep = [c for c in X.columns if c in _KEEP_FEATURES or c == "team"]
    return X[keep]


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


def _opponent_period(df: pd.DataFrame) -> pd.Series:
    """The opponent's own `period` for THIS game, keyed on game_pk."""
    if "game_pk" not in df.columns or not df["game_pk"].notna().any():
        return pd.Series(np.nan, index=df.index)
    src = df.drop_duplicates(subset=["game_pk", "team"]).set_index(
        ["game_pk", "team"])["period"]
    return pd.Series(
        [src.get((pk, o), np.nan) for pk, o in zip(df["game_pk"], df["opponent"])],
        index=df.index)


def _opponent_values(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Values of `cols` from the OPPONENT's own row for the SAME game.

    MUST key on game_pk, never on (opponent, season, our_period). `period` is a
    per-team game counter, so the two sides of a game sit at different periods
    70% of the time. Keying on our period lands on a different game of theirs,
    and 34.6% of the time that game is in the FUTURE -- median 2 days ahead --
    which leaks the result being predicted.

    Measured on 2023-2025: opp_1_ago_diff scored |AUC-0.5| = 0.1084 under the
    broken lookup and 0.0181 once fixed. All of the difference came from the
    future-resolving rows, which alone scored AUC 0.2116 (0.4958 on the
    past-resolving ones, i.e. nothing). Only 36.5% of the broken values matched
    the correct ones.

    Season-constant columns are unaffected either way; time-varying ones are not.
    """
    have = [c for c in cols if c in df.columns]
    out = pd.DataFrame({c: np.nan for c in cols}, index=df.index)
    if not have or "game_pk" not in df.columns or not df["game_pk"].notna().any():
        return out
    src = df.drop_duplicates(subset=["game_pk", "team"]).set_index(
        ["game_pk", "team"])[have]
    vals = src.reindex(list(zip(df["game_pk"], df["opponent"])))
    vals.index = df.index
    for c in have:
        out[c] = vals[c]
    return out


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
    # Outcome columns carried through for label construction ONLY. These are
    # results, so they must never appear in a feature allowlist -- see _CTX in
    # _precompute and _CTX_COLS in _collect_window / build_upcoming_features,
    # none of which include them.
    _LABEL_COLS   = ["diff"]
    # Derived from `diff` below; features, not labels.
    _DIFF_FEATS   = ["1_ago_diff", "diff_mean_5", "win_streak", "loss_streak"]

    needed = {"team", "season", "period", "date", "home"}
    # Every feature gets a matchup edge. Measured over 11,593 paired games, the
    # edge beat the own-side value in 14 of 14 features tested -- unanimously,
    # by +0.0000 (sp_ip_per_start) to +0.0277 (elo). A baseball game is a
    # contest: our bullpen's 3.50 ERA means nothing without theirs. Only three
    # edges existed before this (sp_era, bp_era, elo), and the four form
    # features had no opponent counterpart at all, which is why elo_diff was
    # carrying the opponent's recent form single-handedly.
    _matchup_cols = [
        "opp_sp_era", "opp_sp_whip",
        "sp_era_rolling", "opp_sp_era_rolling",
        "opp_bp_era", "opp_bp_whip",
        "ml_implied_prob",
        "bp_ip_14d",   # rolling N-day bullpen fatigue (seeded separately)
    ] + _bp_rolling_features() + _matchup_extra_features()
    _all_out = ["home", "is_b2b"] + _CTX_FEAT + _matchup_cols + _DIFF_FEATS + _LABEL_COLS

    if not needed.issubset(games_df.columns):
        idx = pd.MultiIndex.from_frame(games_df[["team", "season", "period"]])
        return pd.DataFrame({c: np.nan for c in _all_out}, index=idx)

    odds_present  = [c for c in _CTX_ODDS if c in games_df.columns]
    label_present = [c for c in _LABEL_COLS if c in games_df.columns]
    opp_present   = "opponent" in games_df.columns

    cols = ["team", "season", "period", "date", "home"] + odds_present + label_present
    if opp_present:
        cols.append("opponent")
    # game_pk is the join key for every opponent lookup -- see _opponent_values.
    # Without it those lookups silently return all-NaN.
    if "game_pk" in games_df.columns:
        cols.append("game_pk")

    df = games_df[cols].copy()
    df = df.sort_values(["team", "season", "period"])
    df["prev_date"] = df.groupby(["team", "season"])["date"].shift(1)
    rest = (pd.to_datetime(df["date"]) - pd.to_datetime(df["prev_date"])).dt.days
    df["is_b2b"] = (rest == 1).astype(float)
    df["home"]   = df["home"].astype(float)

    for c in _CTX_ODDS + _LABEL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # --- Opponent matchup features ---
    # Build a lookup: (team, season, period) -> sp_era, sp_whip, bp_era, bp_whip
    pitchers_have = [c for c in ["sp_era", "sp_whip", "sp_k9", "sp_ip_per_start"]
                     if c in df.columns]
    bullpen_have  = [c for c in ["bp_era", "bp_whip", "bp_k9", "bp_hr9",
                                 "bp_ip_per_game"] if c in df.columns]
    lookup_cols   = pitchers_have + bullpen_have

    if opp_present and lookup_cols:
        opp_vals = _opponent_values(df, lookup_cols)

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

        # Remaining stat edges. Sign is always "positive favours us", so rate
        # stats where lower is better (WHIP, HR/9) subtract ours from theirs.
        _EDGE_SPEC = [
            ("sp_whip_edge",        "sp_whip",        -1),
            ("sp_k9_edge",          "sp_k9",          +1),
            ("sp_ip_per_start_edge","sp_ip_per_start",+1),
            ("bp_whip_edge",        "bp_whip",        -1),
            ("bp_k9_edge",          "bp_k9",          +1),
            ("bp_hr9_edge",         "bp_hr9",         -1),
            ("bp_ip_per_game_edge", "bp_ip_per_game", +1),
        ]
        for _name, _col, _sign in _EDGE_SPEC:
            if _col in lookup_cols:
                _opp = opp_vals[_col].values
                df[_name] = ((df[_col] - _opp) if _sign > 0 else (_opp - df[_col]))
                df["opp_" + _col] = _opp
            else:
                df[_name] = np.nan
                df["opp_" + _col] = np.nan
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

        # Same game_pk keying as every other opponent lookup -- de-vigging
        # against a price from a different game of theirs is meaningless.
        tmp = df.assign(_price=price)
        opp_ml = _opponent_values(tmp, ["_price"])["_price"]

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

    # --- Bullpen workload and in-season rates ---
    _keys = list(zip(df["team"], df["season"], df["period"]))
    _wl = _compute_bp_workload(games_df, _BP_WORKLOAD_DAYS,
                               "bp_pitch_game", "bp_pitch_30d").to_dict()
    df["bp_pitch_30d"] = [_wl.get(k, np.nan) for k in _keys]
    for _rate in ("era", "whip", "k9"):
        _d = _compute_bp_rolling_rate(games_df, _rate).to_dict()
        df[f"bp_{_rate}_rolling"] = [_d.get(k, np.nan) for k in _keys]

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
        _op = _opponent_period(df)
        df["opp_sp_era_rolling"] = [
            _sp_roll_dict.get((o, se, int(pp)), np.nan) if pp == pp else np.nan
            for o, se, pp in zip(df["opponent"], df["season"], _op)]
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

    # --- Diff-based history: the market-free counterpart to the SpreadScore
    # block. spreadscore = diff + spread, so the SS features carry the book's
    # favourite/underdog call; these do not. Strictly pre-game: shift(1) before
    # any window. Used by the win model in place of the SS features.
    _dsrc = df.sort_values(["team", "season", "period"])
    _g = _dsrc.groupby(["team", "season"])["diff"]
    df["1_ago_diff"] = _g.shift(1).reindex(df.index)
    df["diff_mean_5"] = (_g.transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
                         .reindex(df.index))

    _w = np.zeros(len(_dsrc)); _l = np.zeros(len(_dsrc)); _i = 0
    for _, _sub in _dsrc.groupby(["team", "season"], sort=False):
        cw = cl = 0
        for _v in pd.to_numeric(_sub["diff"], errors="coerce").shift(1):
            _w[_i], _l[_i] = cw, cl
            if pd.isna(_v):  cw = cl = 0
            elif _v > 0:     cw, cl = cw + 1, 0
            else:            cw, cl = 0, cl + 1
            _i += 1
    df["win_streak"] = pd.Series(_w, index=_dsrc.index).reindex(df.index)
    df["loss_streak"] = pd.Series(_l, index=_dsrc.index).reindex(df.index)

    # --- Form edges ---
    # These four had NO opponent counterpart, so the tree could not compare
    # recent form at all. That gap is why elo_diff mattered so much: the
    # league-window Elo was a differenced recent-form signal (correlation 0.784
    # with own-minus-opponent last-game margin) and the only feature carrying
    # the opponent's side of it. Built here rather than in the matchup block
    # above because the source columns do not exist until now.
    _FORM_EDGES = [("1_ago_diff_edge",  "1_ago_diff",  +1),
                   ("diff_mean_5_edge", "diff_mean_5", +1),
                   ("win_streak_edge",  "win_streak",  +1),
                   ("loss_streak_edge", "loss_streak", -1)]
    if opp_present:
        _fv = _opponent_values(df, [c for _, c, _s in _FORM_EDGES])
        for _name, _col, _sign in _FORM_EDGES:
            _opp = _fv[_col].values
            df[_name] = ((df[_col] - _opp) if _sign > 0 else (_opp - df[_col]))
            df["opp_" + _col] = _opp
    else:
        for _name, _col, _sign in _FORM_EDGES:
            df[_name] = np.nan
            df["opp_" + _col] = np.nan

    # --- Late mirrors ---
    # bp_ip_14d and is_b2b are built after the matchup lookup, so they need a
    # second pass. Both are comparative by nature: a depleted bullpen matters
    # against a fresh one, and a team on no rest matters against a rested one.
    _LATE_MIRROR = ["bp_ip_14d", "is_b2b"] + _bp_rolling_features()
    if opp_present:
        _lm_have = [c for c in _LATE_MIRROR if c in df.columns]
        if _lm_have:
            _lm = _opponent_values(df, _lm_have)
            for _c in _LATE_MIRROR:
                df["opp_" + _c] = _lm[_c].values if _c in _lm_have else np.nan
    else:
        for _c in _LATE_MIRROR:
            df["opp_" + _c] = np.nan

    out_cols = [c for c in _all_out if c in df.columns]
    return df.set_index(["team", "season", "period"])[out_cols]


def _compute_sp_rolling_era(
    games_df: pd.DataFrame,
    n_starts: int = 5,
    fallback_col: str = "sp_era",
) -> pd.Series:
    """
    Starter ERA blending season-to-date form with last season's baseline.

    Weight shifts toward the current season as innings accumulate:

        w   = season_ip / (season_ip + _SP_ERA_BLEND_LAMBDA)
        era = w * season_to_date + (1 - w) * prior_season

    This replaced a trailing-`n_starts` window with a hard fallback to
    prior-season below 2 starts. That rule measured poorly early (correlation
    with next-start ER/9 of 0.015 under 20 IP) because ~30 innings of ERA is
    mostly noise, and it discarded the prior-season signal entirely once past
    the cliff. `n_starts` is retained for signature compatibility but no longer
    used.

    Requires 'sp_ip_game' and 'sp_er_game' (seeded by --sp-game-stats); falls
    back to prior-season ERA for every row when they are absent.

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

    rolling_eras = {}   # (team, season, period) -> blended ERA

    for (sp_name, season_val), grp in df.groupby(["sp_name", "season"]):
        if not sp_name:
            continue
        grp = grp.sort_values("date").reset_index(drop=True)

        for i, row in grp.iterrows():
            key = (row["team"], season_val, int(row["period"]))
            # Prior starts in this season for this pitcher (exclude current game)
            prior = grp.iloc[:i].dropna(subset=["sp_ip_game", "sp_er_game"])
            prior = prior[prior["sp_ip_game"] > 0]

            prior_era = row[fallback_col] if (fallback_col in row.index
                                              and pd.notna(row[fallback_col])) else np.nan

            season_ip = float(prior["sp_ip_game"].sum()) if len(prior) else 0.0
            season_era = np.nan
            if season_ip > 0:
                season_era = float(prior["sp_er_game"].sum()) / season_ip * 9.0

            # Weight season-to-date against last season by innings accumulated.
            # Season-to-date beats the old trailing-5 window once innings build
            # up (correlation with next-start ER/9 rises 0.016 -> 0.101 from
            # <20 IP to 100+ IP) while last season's number goes stale over the
            # same span (0.078 -> 0.043). Blending tracks both: correlation
            # 0.089 versus 0.059 for the trailing-5 rule this replaces, and it
            # removes that rule's cliff at 2 starts, where it scored just 0.015.
            if np.isnan(season_era) and np.isnan(prior_era):
                continue
            if np.isnan(season_era):
                rolling_eras[key] = round(float(prior_era), 3)
            elif np.isnan(prior_era):
                rolling_eras[key] = round(season_era, 3)
            else:
                w = season_ip / (season_ip + _SP_ERA_BLEND_LAMBDA)
                rolling_eras[key] = round(w * season_era + (1 - w) * float(prior_era), 3)

    for key, val in rolling_eras.items():
        if key in result.index:
            result[key] = val

    return result


def _compute_bp_workload(games_df: pd.DataFrame, days: int, col: str,
                         name: str) -> pd.Series:
    """Sum of `col` over the `days` calendar days before each game, current excluded."""
    idx = pd.MultiIndex.from_frame(games_df[["team", "season", "period"]])
    if col not in games_df.columns:
        return pd.Series(np.nan, index=idx, name=name)
    w = games_df[["team", "season", "date", "period", col]].copy()
    w["date"] = pd.to_datetime(w["date"])
    w[col] = pd.to_numeric(w[col], errors="coerce")
    w = w.sort_values(["team", "season", "date"])
    rows = []
    for (team, season_val), grp in w.groupby(["team", "season"]):
        grp = grp.set_index("date").sort_index()
        roll = grp[col].shift(1, freq="D").rolling(f"{days}D").sum()
        for period, val in zip(grp["period"], roll.values):
            rows.append({"team": team, "season": season_val,
                         "period": int(period), name: val})
    if not rows:
        return pd.Series(np.nan, index=idx, name=name)
    return pd.DataFrame(rows).set_index(["team", "season", "period"])[name]


_BP_RATE_SPEC = {
    "era":  (["bp_er_game"], 9.0, "bp_era"),
    "whip": (["bp_bb_game", "bp_h_game"], 1.0, "bp_whip"),
    "k9":   (["bp_k_game"], 9.0, "bp_k9"),
}


def _compute_bp_rolling_rate(games_df: pd.DataFrame, rate: str) -> pd.Series:
    """
    In-season bullpen rate shrunk toward the prior-season team value.

    Strictly causal: each game sees only the team's earlier games. Falls back to
    the prior-season number where no in-season innings exist yet, and to the
    in-season figure where no prior-season number exists.
    """
    nums, mult, prior_col = _BP_RATE_SPEC[rate]
    lam = _BP_RATE_LAMBDA[rate]
    name = f"bp_{rate}_rolling"
    idx = pd.MultiIndex.from_frame(games_df[["team", "season", "period"]])
    need = set(nums) | {"bp_ip_game"}
    if not need.issubset(games_df.columns):
        return pd.Series(np.nan, index=idx, name=name)

    d = games_df[["team", "season", "period", "date", "bp_ip_game"] + nums].copy()
    if prior_col in games_df.columns:
        d[prior_col] = pd.to_numeric(games_df[prior_col], errors="coerce")
    else:
        d[prior_col] = np.nan
    d["date"] = pd.to_datetime(d["date"])
    for c_ in nums + ["bp_ip_game"]:
        d[c_] = pd.to_numeric(d[c_], errors="coerce")
    sub = d[nums]
    num = sub.sum(axis=1).to_numpy(dtype=float)
    num[sub.isna().any(axis=1).to_numpy()] = np.nan
    d["_num"] = num

    out = []
    for (team, season_val), grp in d.groupby(["team", "season"], sort=False):
        grp = grp.sort_values("date")
        cip = cnum = 0.0
        for _, r in grp.iterrows():
            std = (cnum * mult / cip) if cip > 0 else np.nan
            p = r[prior_col]
            if not np.isfinite(std):
                val = p
            elif not np.isfinite(p):
                val = std
            else:
                wgt = cip / (cip + lam)
                val = wgt * std + (1 - wgt) * p
            out.append({"team": team, "season": season_val,
                        "period": int(r["period"]), name: val})
            if np.isfinite(r["_num"]) and np.isfinite(r["bp_ip_game"]):
                cip += float(r["bp_ip_game"]); cnum += float(r["_num"])
    if not out:
        return pd.Series(np.nan, index=idx, name=name)
    return pd.DataFrame(out).set_index(["team", "season", "period"])[name]


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
        "opp_sp_era", "opp_sp_whip",
        "sp_era_rolling", "opp_sp_era_rolling",
        "bp_era", "bp_whip", "bp_k9", "bp_hr9", "bp_ip_per_game",
        "opp_bp_era", "opp_bp_whip",
        "ml_implied_prob",
        "bp_ip_14d",
        "1_ago_diff", "diff_mean_5", "win_streak", "loss_streak",
    ] + _bp_rolling_features() + _matchup_extra_features()
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
        for _c in ("fast_elo_diff", "fast_opp_elo"):
            df[_c] = (elo_aligned[_c].values
                      if _c in elo_aligned.columns else np.nan)
    except KeyError:
        df["elo_diff"]     = np.nan
        df["opponent_elo"] = np.nan
        df["fast_elo_diff"] = np.nan
        df["fast_opp_elo"]  = np.nan

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
        "opp_sp_era", "opp_sp_whip",
        "sp_era_rolling", "opp_sp_era_rolling",
        "bp_era", "bp_whip", "bp_k9", "bp_hr9", "bp_ip_per_game",
        "opp_bp_era", "opp_bp_whip",
        "ml_implied_prob",
        "bp_ip_14d",
        "1_ago_diff", "diff_mean_5", "win_streak", "loss_streak",
    ] + _bp_rolling_features() + _matchup_extra_features()
    if _MARKET_BLIND:
        _CTX_COLS = [c for c in _CTX_COLS if c not in _MARKET_FEATURES]

    if upcoming_context is not None and not upcoming_context.empty:
        for col in _CTX_COLS:
            X[col] = X["team"].map(upcoming_context[col]) if col in upcoming_context.columns else np.nan
    else:
        for col in _CTX_COLS:
            X[col] = np.nan

    # Elo ratings for upcoming game (pre-game, using all completed games this season).
    # Must mirror _precompute exactly — a column present in training but missing
    # here would silently become NaN for every team.
    elo_df = elo_mod.compute(completed, k=best_k)
    if _FAST_ELO_K is not None:
        _f = elo_mod.compute(completed, k=best_k, window=_REVERT_TEST_WINDOW)
        elo_df = elo_df.join(_f[["elo_diff", "opp_elo"]].rename(
            columns={"elo_diff": "fast_elo_diff", "opp_elo": "fast_opp_elo"}))
    _elo_cols = ["elo", "opp_elo", "elo_diff", "fast_elo_diff", "fast_opp_elo"]
    latest_elo = (
        elo_df.reset_index()
        .sort_values("period")
        .groupby("team")
        .last()
        [[c for c in _elo_cols if c in elo_df.columns]]
    )
    X["elo_diff"]     = X["team"].map(latest_elo["elo_diff"])
    X["opponent_elo"] = X["team"].map(latest_elo["opp_elo"])
    for _c in ("fast_elo_diff", "fast_opp_elo"):
        X[_c] = (X["team"].map(latest_elo[_c])
                 if _c in latest_elo.columns else np.nan)

    # style_edge is not produced. It contributed nothing to the model, and
    # fitting the StyleModel here meant a full SVD of the season's matchup
    # matrix on every prediction run. Note this used to refit even when the
    # caller passed style_model=None. See _USE_STYLE_EDGE.

    _recast_categoricals(X)
    X = _apply_feature_trim(X)
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
    _KEY_FEATS  = ["sp_era", "opp_sp_era", "bp_era", "opp_bp_era",
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
    _tuned_params = {
        **_XGB_FIXED, "random_state": 42,
        **_tune(XGBClassifier, X_train, y_train, max_evals, seed=seed),
    }
    clf = XGBClassifier(**_tuned_params)
    if _TRAIN_COVER_MODEL:
        clf.fit(X_train, y_train)
    else:
        print("  _TRAIN_COVER_MODEL is False -- skipping the cover fit; "
              "its tuned hyperparameters still carry to the win model.")

    # Evaluation metrics
    from sklearn.metrics import roc_auc_score, accuracy_score
    scores = {"lookback": best_lookback, "elo_k": best_k}

    if _TRAIN_COVER_MODEL:
        scores["clf_train_acc"] = round(
            float(accuracy_score(y_train, clf.predict(X_train))), 3)
        try:
            scores["clf_train_roc"] = round(
                float(roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])), 3)
        except ValueError:
            scores["clf_train_roc"] = float("nan")
        if not X_test.empty:
            try:
                scores["clf_test_roc"] = round(float(
                    roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])), 3)
                scores["clf_test_acc"] = round(float(
                    accuracy_score(y_test, clf.predict(X_test))), 3)
            except ValueError:
                pass
        if not X_val.empty:
            try:
                scores["clf_val_roc"] = round(float(
                    roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])), 3)
                scores["clf_val_acc"] = round(float(
                    accuracy_score(y_val, clf.predict(X_val))), 3)
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
        _win_params = dict(_tuned_params)
        _win_params.update(_XGB_FIXED)
        _win_params["random_state"] = 42
        win_clf = XGBClassifier(**_win_params)
        # Fit WITHOUT the spreadscore-derived features -- see
        # _SPREAD_DERIVED_FEATURES. The cover model keeps them (they are built
        # for its target); the win model gets the diff-based equivalents that
        # sit alongside them in the matrix. predict() reads win_clf's own
        # feature_names, so the narrower column set flows through untouched.
        _win_drop = [c for c in _SPREAD_DERIVED_FEATURES if c in X_train.columns]
        print(f"  Win model excludes {len(_win_drop)} spreadscore-derived "
              f"features: {_win_drop}")
        win_clf.fit(X_train.drop(columns=_win_drop)[_m], w_train[_m])

        for split, X_s, w_s in (("test", X_test, win_labels.get("test")),
                                ("val", X_val, win_labels.get("val"))):
            if X_s.empty or w_s is None or len(w_s) != len(X_s):
                continue
            m = w_s.notna().values
            if m.sum() < 50:
                continue
            _Xs = X_s.drop(columns=[c for c in _win_drop if c in X_s.columns])
            try:
                scores[f"win_{split}_roc"] = round(float(
                    roc_auc_score(w_s[m], win_clf.predict_proba(_Xs[m])[:, 1])), 3)
                scores[f"win_{split}_acc"] = round(float(
                    accuracy_score(w_s[m], win_clf.predict(_Xs[m]))), 3)
            except ValueError:
                pass
    else:
        print("  No win labels available — skipping win classifier.")

    # Refit on all seasons (including eval_season) once hyperparams are locked in.
    # eval metrics above are computed before this refit so they remain unbiased.
    all_X_parts = [X_train] + ([X_test] if not X_test.empty else []) + ([X_val] if not X_val.empty else [])
    all_y_parts = [y_train] + ([y_test] if not X_test.empty else []) + ([y_val] if not X_val.empty else [])
    _all_X = pd.concat(all_X_parts, ignore_index=True)
    if _TRAIN_COVER_MODEL:
        clf.fit(_all_X, pd.concat(all_y_parts, ignore_index=True))

    if win_clf is not None:
        all_w_parts = [win_labels["train"]] \
            + ([win_labels["test"]] if not X_test.empty else []) \
            + ([win_labels["val"]] if not X_val.empty else [])
        _all_w = pd.concat(all_w_parts, ignore_index=True)
        if len(_all_w) == len(_all_X):
            _m = _all_w.notna().values
            _wd = [c for c in _SPREAD_DERIVED_FEATURES if c in _all_X.columns]
            win_clf.fit(_all_X.drop(columns=_wd)[_m], _all_w[_m])

    return (clf if _TRAIN_COVER_MODEL else None), scores, best_lookback,         best_k, style_model, win_clf


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

import joblib as _joblib
import datetime as _datetime
import os as _os

_DEFAULT_MODEL_PATH = _os.path.join(_os.path.dirname(__file__), "data", "mlb_model.pkl")


def save_model(
    clf: "XGBClassifier | None",
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
    clf: "XGBClassifier | None",
    X_pred: pd.DataFrame,
    win_clf: "XGBClassifier | None" = None,
) -> pd.DataFrame:
    """
    Generate probabilities for both markets, indexed by team.

    Each classifier outputs its probability directly — no pairing or normal-CDF
    transformation is needed.

    Parameters
    ----------
    clf     : fitted cover classifier, or None when _TRAIN_COVER_MODEL is off
              (MLB is moneyline-only). coverprob is NaN in that case.
    X_pred  : feature matrix with a `team` column (from build_prediction_features)
    win_clf : optional win classifier. Absent in bundles saved before the
              dual-target change, in which case win_prob is NaN and callers
              fall back to spread-only betting.

    Returns
    -------
    DataFrame indexed by team with columns:
        coverprob : P(team covers the run line), or NaN if no cover model
        win_prob  : P(team wins outright), or NaN if no win model
    """
    teams = X_pred["team"].values

    if clf is not None:
        feat_cols = clf.get_booster().feature_names
        coverprob = clf.predict_proba(X_pred[feat_cols])[:, 1]
    else:
        coverprob = np.full(len(teams), np.nan)

    out = pd.DataFrame({"coverprob": coverprob}, index=teams)

    if win_clf is not None:
        win_cols = win_clf.get_booster().feature_names
        out["win_prob"] = win_clf.predict_proba(X_pred[win_cols])[:, 1]
    else:
        out["win_prob"] = np.nan
    return out
