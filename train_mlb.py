"""
Train the MLB cover-probability model and save it to data/mlb_model.pkl.

Trains on every season before EVAL_SEASON and reports on EVAL_SEASON, then
saves the bundle. Hyperparameters are NOT tuned on EVAL_SEASON -- both
_select_hyperparams and _tune cross-validate inside the training set, so the
eval season is a genuine holdout used only for reporting.  The saved bundle is loaded by:
  - _kelly_analysis.py   (backtest / Kelly sizing report)
  - predict_mlb.py       (daily prediction pipeline)

Usage
-----
    python train_mlb.py              # default 50 hyperopt evals
    python train_mlb.py --evals 100  # more thorough search
"""
import argparse
import datetime

import pandas as pd

import db
import data_pipeline as dp
import model as ml
from config import MLB

# The season held out for reporting. Roll this forward each year: everything
# before it becomes training data.
#
# Both selection steps cross-validate INSIDE the training set --
# _select_hyperparams and _tune each call cross_val_score on X_train -- so the
# eval season is never used to choose lookback, K, or hyperparameters. X_val
# appears only in the reported scores. That makes it a genuine holdout, and it
# means holding out a COMPLETE season buys nothing while costing a full season
# of training data.
#
# Pointing it at the in-progress season instead puts every complete season into
# the fit (2022-2025) and turns the reported val score into an honest read on
# the season actually being bet. Walk-forward fold C measured this exact
# configuration at 0.6169 on 2026, against 0.6162 for the old 2022-24 fit.
EVAL_SEASON  = datetime.date.today().year
NEXT_PERIOD  = 163
SPLIT_PERIOD = MLB.eval_split_period

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train the MLB cover model")
parser.add_argument(
    "--evals", type=int, default=50,
    help="Number of Hyperopt TPE evaluations (default 50)",
)
args = parser.parse_args()

# ── Load & filter data ───────────────────────────────────────────────────────
print("Loading data...")
client = db.connect()
try:
    all_games = db.fetch_games(client, "mlb")
finally:
    client.close()

seasons = sorted(all_games["season"].unique())
all_games = pd.concat(
    [dp.filter_regular_season(all_games[all_games["season"] == s], MLB, s)
     for s in seasons],
    ignore_index=True,
)

games_phase1 = all_games[all_games["season"] <= EVAL_SEASON].copy()
# Every season in games_phase1, not just the pre-refit ones: train_models ends
# by refitting the win model on all seasons INCLUDING EVAL_SEASON, so recording
# only the pre-refit seasons understates what the artifact actually saw and
# invites someone to "validate" it on data it trained on.
train_seasons = sorted(games_phase1["season"].unique())

print(f"  Fit seasons (after refit): {train_seasons}  |  Reported on: {EVAL_SEASON}")
print(f"  NOTE: the saved model is refit on {EVAL_SEASON} too — the scores below "
      f"are pre-refit. Use walkforward.py for an honest number.")
print(f"  Rows: {len(games_phase1)}")

# ── Train ─────────────────────────────────────────────────────────────────────
clf, scores, best_lookback, best_k, style_model, win_clf = ml.train_models(
    games_phase1, NEXT_PERIOD, EVAL_SEASON, SPLIT_PERIOD, max_evals=args.evals,
)

# ── Save ──────────────────────────────────────────────────────────────────────
ml.save_model(
    clf, scores, best_lookback, best_k, style_model,
    next_period=NEXT_PERIOD, train_seasons=train_seasons, win_clf=win_clf,
)

print(f"\nDone.")
print(f"  lookback={best_lookback}, Elo K={best_k}, "
      f"bp_ip_14d={ml._BP_FATIGUE_DAYS}d (fixed)")
print(f"  Scores: {scores}")
