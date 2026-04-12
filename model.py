"""
Machine-learning layer: feature engineering, hyperparameter tuning, training,
and prediction.

Feature design
--------------
Rolling SpreadScore (diff + spread_line) is used as the feature matrix.
SpreadScore captures performance-vs-expectations, which is the signal most
relevant to future spread coverage. ~31% of feature cells are NaN (games
without a spread line); XGBoost handles these natively.

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
from xgboost import XGBClassifier, XGBRegressor

_XGB_FIXED = {"enable_categorical": True, "tree_method": "hist"}

# team is excluded — see module docstring
_CAT_COLS = ("season", "period")

_LOOKBACK_CANDIDATES = [3, 5, 7, 10, 15]

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

def build_features(
    games_df: pd.DataFrame,
    next_period: int,
    lookback: int,
    eval_season: int,
    eval_split_period: int,
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

    Returns
    -------
    X_train, X_test, y_train, y_test, X_val, y_val
    """
    df_ss = games_df[["team", "season", "period", "spreadscore"]]
    ss_pivot = df_ss.pivot_table(
        index=["team", "season"], columns="period", values="spreadscore"
    )

    is_eval = ss_pivot.index.get_level_values("season") == eval_season
    train_pivot = ss_pivot[~is_eval]
    eval_pivot  = ss_pivot[is_eval]

    X_train_parts, y_train_parts = [], []
    X_test_parts,  y_test_parts  = [], []
    X_val_parts,   y_val_parts   = [], []

    for start in range(1, next_period - lookback + 1):
        target = start + lookback
        _collect_window(train_pivot, start, lookback, target,
                        X_train_parts, y_train_parts)
        if target < eval_split_period:
            _collect_window(eval_pivot, start, lookback, target,
                            X_test_parts, y_test_parts)
        else:
            _collect_window(eval_pivot, start, lookback, target,
                            X_val_parts, y_val_parts)

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

    return X_train, X_test, y_train, y_test, X_val, y_val


def _recast_categoricals(df: pd.DataFrame) -> None:
    """Re-cast _CAT_COLS to category dtype in-place (lost after pd.concat)."""
    for col in _CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")


def _collect_window(
    pivot: pd.DataFrame,
    start: int,
    lookback: int,
    target: int,
    X_out: list,
    y_out: list,
) -> None:
    """
    Append one sliding-window sample to X_out / y_out.

    Features: spreadscore for periods [start, start+lookback), NaN allowed.
    Target  : spreadscore at `target`, must be non-NaN.
    `team`  : dropped from feature matrix to prevent identity memorisation.
    """
    if target not in pivot.columns:
        return
    feature_cols = [c for c in range(start, start + lookback) if c in pivot.columns]
    if len(feature_cols) < lookback:
        return

    y = pivot[target].dropna()
    if y.empty:
        return

    common_idx = pivot.index.intersection(y.index)
    X = pivot[feature_cols].loc[common_idx]
    y = y.loc[common_idx]
    if X.empty:
        return

    df = X.copy().reset_index()
    # Drop team — keep only season (temporal context, not identity)
    df = df.drop(columns=["team"], errors="ignore")
    df = df.rename(columns={col: f"{lookback - i}_ago" for i, col in enumerate(feature_cols)})
    df["period"] = target
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
) -> pd.DataFrame:
    """
    Build the prediction feature matrix for upcoming games.

    `team` is kept in the returned DataFrame (used as the output index in
    predict()) but is not included in the trained feature names, so it is
    automatically excluded when the model selects its input columns.
    """
    completed = season_games[season_games["period"] < next_period]
    if completed.empty:
        return pd.DataFrame()

    pivot = completed.pivot_table(index="team", columns="period", values="spreadscore")
    available = min(lookback, pivot.shape[1])
    X = pivot.iloc[:, -available:].copy()

    for missing in range(lookback - available, 0, -1):
        X.insert(0, f"_pad_{missing}", np.nan)

    X.columns = [f"{lookback - i}_ago" for i in range(lookback)]
    X = X.reset_index()          # brings team back as a column for the index
    X["season"] = season
    X["period"] = next_period
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


def _select_lookback(
    games_df: pd.DataFrame,
    next_period: int,
    eval_season: int,
    eval_split_period: int,
) -> int:
    """
    Choose the best lookback from _LOOKBACK_CANDIDATES using 5-fold CV
    on the training set only (eval_season is excluded).

    Scores the classifier (F1) since cover prediction is the primary goal.
    Falls back to the smallest candidate if no candidate yields enough data.
    """
    best_lb, best_score = _LOOKBACK_CANDIDATES[0], -np.inf
    for lb in _LOOKBACK_CANDIDATES:
        try:
            X_train, _, y_train, _, _, _ = build_features(
                games_df, next_period, lb, eval_season, eval_split_period
            )
        except ValueError:
            continue
        if len(X_train) < 30:
            continue
        y_enc = _encode_cover(y_train)
        # Quick probe: fixed shallow model, no full hyperopt
        score = cross_val_score(
            XGBClassifier(**_XGB_FIXED, max_depth=3, n_estimators=100,
                          learning_rate=0.1),
            X_train, y_enc, cv=5, scoring="f1",
        ).mean()
        if score > best_score:
            best_score, best_lb = score, lb
    return best_lb


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
) -> tuple[XGBRegressor, XGBClassifier, dict, int]:
    """
    Select lookback, tune hyperparameters, and fit both models.

    Lookback is chosen via CV on the training set; XGBoost hyperparameters
    are then tuned with Hyperopt using that fixed lookback.

    Returns
    -------
    reg          : fitted XGBRegressor (predicts SpreadScore)
    clas         : fitted XGBClassifier (predicts cover probability)
    scores       : dict of train/test/val evaluation metrics
    best_lookback: lookback value selected by CV
    """
    print("  Selecting lookback...")
    best_lookback = _select_lookback(games_df, next_period, eval_season, eval_split_period)
    print(f"  Best lookback: {best_lookback}")

    X_train, X_test, y_train, y_test, X_val, y_val = build_features(
        games_df, next_period, best_lookback, eval_season, eval_split_period
    )
    print(f"  Rows — train: {len(X_train)}, test: {len(X_test)}, val: {len(X_val)}")

    y_train_enc = _encode_cover(y_train)
    y_test_enc  = _encode_cover(y_test)

    print("  Tuning regression model...")
    reg = XGBRegressor(**_XGB_FIXED, **_tune(XGBRegressor, X_train, y_train, max_evals))
    reg.fit(X_train, y_train)

    print("  Tuning classification model...")
    clas = XGBClassifier(**_XGB_FIXED, **_tune(XGBClassifier, X_train, y_train_enc, max_evals))
    clas.fit(X_train, y_train_enc)

    scores = {
        "lookback": best_lookback,
        "reg_train_r2":   reg.score(X_train, y_train),
        "reg_test_r2":    reg.score(X_test, y_test) if not X_test.empty else None,
        "clas_train_acc": clas.score(X_train, y_train_enc),
        "clas_test_acc":  clas.score(X_test, y_test_enc) if not X_test.empty else None,
    }
    if not X_val.empty:
        scores["reg_val_r2"]   = reg.score(X_val, y_val)
        scores["clas_val_acc"] = clas.score(X_val, _encode_cover(y_val))

    return reg, clas, scores, best_lookback


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    reg: XGBRegressor,
    clas: XGBClassifier,
    X_pred: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions indexed by team.

    Selects only the feature columns the models were trained on, so extra
    columns (e.g. `team`) in X_pred are automatically ignored.

    Returns DataFrame with columns: predspread, coverprob.
    predspread : expected SpreadScore (positive = team expected to cover)
    coverprob  : probability of covering the spread (higher = more likely)
    """
    predspread = reg.predict(X_pred[reg.get_booster().feature_names])
    coverprob  = clas.predict_proba(X_pred[clas.get_booster().feature_names])[:, 1]
    return pd.DataFrame(
        {"predspread": predspread, "coverprob": coverprob},
        index=X_pred["team"].values,
    )
