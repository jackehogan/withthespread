"""
Machine-learning layer: feature engineering, hyperparameter tuning, training,
and prediction.

The core concept is sport-agnostic: the model's only inputs are a team's last
N SpreadScore values (SpreadScore = point_diff + spread_line) plus categorical
identifiers (team, season, period). Both the regression (how much will they
cover by?) and classification (will they cover?) models share the same feature
set and training procedure.

Note on sign inversion
----------------------
Empirically the raw predictions are systematically in the wrong direction.
Both outputs are negated before being returned by predict() until the root
cause is resolved.

Public interface
----------------
build_features(games_df, next_period, lookback, validation_season)
build_prediction_features(season_games, next_period, lookback, season)
train_models(X_train, X_test, y_train, y_test, X_val, y_val, max_evals)
predict(reg, clas, X_pred)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier, XGBRegressor

_XGB_FIXED = {"enable_categorical": True, "tree_method": "hist"}
_CAT_COLS = ("team", "season", "period")

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
    validation_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build train/test/validation splits from long-format historical game data.

    Slides a window of size `lookback` across all completed periods, producing
    one sample per (team, season, window). The validation season is held out
    entirely from the train/test split.

    Parameters
    ----------
    games_df         : must contain columns team, season, period, spreadscore
    next_period      : the period we want to predict; windows stop before it
    lookback         : number of prior periods used as features (tunable)
    validation_season: season held out for evaluation

    Returns
    -------
    X_train, X_test, y_train, y_test, X_val, y_val
    """
    df = games_df[["team", "season", "period", "spreadscore"]].dropna()
    pivot = df.pivot_table(index=["team", "season"], columns="period", values="spreadscore")

    is_val = pivot.index.get_level_values("season") == validation_season
    train_pivot, val_pivot = pivot[~is_val], pivot[is_val]

    X_parts, y_parts, Xv_parts, yv_parts = [], [], [], []
    for start in range(1, next_period - lookback + 1):
        target = start + lookback
        _collect_window(train_pivot, start, lookback, target, X_parts, y_parts)
        _collect_window(val_pivot, start, lookback, target, Xv_parts, yv_parts)

    if not X_parts:
        raise ValueError(
            f"No training windows found (next_period={next_period}, lookback={lookback}). "
            f"Need at least {lookback + 1} periods of data."
        )

    X = pd.concat(X_parts, ignore_index=True)
    # Re-cast after concat: mismatched category sets collapse to object
    for col in _CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y = pd.concat(y_parts, ignore_index=True).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if Xv_parts:
        X_val = pd.concat(Xv_parts, ignore_index=True)
        for col in _CAT_COLS:
            if col in X_val.columns:
                X_val[col] = X_val[col].astype("category")
        y_val = pd.concat(yv_parts, ignore_index=True).astype(float)
    else:
        X_val = pd.DataFrame(columns=X.columns)
        y_val = pd.Series(dtype=float)

    return X_train, X_test, y_train, y_test, X_val, y_val


def _collect_window(
    pivot: pd.DataFrame,
    start: int,
    lookback: int,
    target: int,
    X_out: list,
    y_out: list,
) -> None:
    """Append one sliding-window sample to X_out / y_out if data is available."""
    if target not in pivot.columns:
        return
    feature_cols = [c for c in range(start, start + lookback) if c in pivot.columns]
    if len(feature_cols) < lookback:
        return
    y = pivot[target].dropna()
    X = pivot[feature_cols].loc[y.index].dropna()
    y = y.loc[X.index]
    if X.empty:
        return

    df = X.copy().reset_index()
    df = df.rename(columns={col: f"{lookback - i}_ago" for i, col in enumerate(feature_cols)})
    df["period"] = target
    for col in _CAT_COLS:
        df[col] = df[col].astype("category")
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
    Build the prediction feature matrix using the last `lookback` completed
    periods per team. Teams with fewer periods get NaN-padded features
    (XGBoost handles missing values natively).
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
    X = X.reset_index()
    X["season"] = season
    X["period"] = next_period
    for col in _CAT_COLS:
        X[col] = X[col].astype("category")
    for col in X.columns:
        if col not in _CAT_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _encode_cover(y: pd.Series) -> pd.Series:
    """Map sign of SpreadScore to binary cover label: positive → 1, else → 0."""
    return np.sign(y).replace({-1: 0, 1: 1}).astype(int)


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
        return -cross_val_score(model_class(**p), X_train, y_train, cv=5, scoring=scoring).mean()

    best = fmin(fn=objective, space=_HYPEROPT_SPACE, algo=tpe.suggest, max_evals=max_evals)
    return {**best, "max_depth": int(best["max_depth"]), "n_estimators": int(best["n_estimators"])}


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    max_evals: int = 10,
) -> tuple[XGBRegressor, XGBClassifier, dict]:
    """
    Tune and fit regression + classification models.

    Returns
    -------
    reg    : fitted XGBRegressor (predicts SpreadScore)
    clas   : fitted XGBClassifier (predicts cover probability)
    scores : dict of train/test/val evaluation metrics
    """
    y_train_enc = _encode_cover(y_train)
    y_test_enc = _encode_cover(y_test)

    print("  Tuning regression model...")
    reg = XGBRegressor(**_XGB_FIXED, **_tune(XGBRegressor, X_train, y_train, max_evals))
    reg.fit(X_train, y_train)

    print("  Tuning classification model...")
    clas = XGBClassifier(**_XGB_FIXED, **_tune(XGBClassifier, X_train, y_train_enc, max_evals))
    clas.fit(X_train, y_train_enc)

    scores = {
        "reg_train_r2": reg.score(X_train, y_train),
        "reg_test_r2": reg.score(X_test, y_test),
        "clas_train_acc": clas.score(X_train, y_train_enc),
        "clas_test_acc": clas.score(X_test, y_test_enc),
    }
    if X_val is not None and not X_val.empty:
        scores["reg_val_r2"] = reg.score(X_val, y_val)
        scores["clas_val_acc"] = clas.score(X_val, _encode_cover(y_val))

    return reg, clas, scores


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

    Returns DataFrame with columns: predspread, coverprob.
    Both values are negated — see module docstring for explanation.
    """
    predspread = -reg.predict(X_pred[reg.get_booster().feature_names])
    coverprob = 1.0 - clas.predict_proba(X_pred[clas.get_booster().feature_names])[:, 1]
    return pd.DataFrame(
        {"predspread": predspread, "coverprob": coverprob},
        index=X_pred["team"].values,
    )
