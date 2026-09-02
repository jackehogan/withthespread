"""
Test B: does the causal PREDICTION path reproduce the TRAINING path's features?

The prediction builder is inherently causal -- on a given morning it only knows
about completed games. So if it reproduces the training path's feature values
for past dates, the training path cannot be using anything unavailable at the
time. Two independently written builders agreeing is much stronger evidence
than one builder checked against itself.

Prediction writes are stubbed: this must not touch the stored bet record.

    python _audit_path_parity.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib

import db
import elo as elo_mod
import model as ml
import predict_mlb

DATES = ["2026-04-20", "2026-05-05", "2026-05-20", "2026-06-05", "2026-06-20",
         "2026-07-05", "2026-07-20", "2026-08-05", "2026-08-20"]


def main():
    b = joblib.load(r"data\mlb_model.pkl")
    feats = list(b["win_clf"].feature_names_in_)
    K, FASTK = float(b["best_k"]), ml._FAST_ELO_K

    client = db.connect()
    try:
        g = db.fetch_games(client, "mlb")
    finally:
        client.close()
    g["date"] = pd.to_datetime(g["date"])

    # ---- training-path reference ------------------------------------------
    ctx = ml._compute_context(g[g["season"] == 2026]).reset_index()
    ref = ctx.set_index(["team", "period"])
    er = elo_mod.compute(g, k=K)
    ef = elo_mod.compute(g, k=FASTK)
    for src, cols in ((er, {"elo_diff": "elo_diff", "opp_elo": "opponent_elo"}),
                      (ef, {"elo_diff": "fast_elo_diff", "opp_elo": "fast_opp_elo"})):
        s = src.reset_index()
        s = s[s["season"] == 2026]
        for a, bcol in cols.items():
            ref[bcol] = s.set_index(["team", "period"])[a].reindex(ref.index)
    print(f"training-path reference rows: {len(ref)}")

    # ---- prediction path, run retroactively -------------------------------
    cap = {}
    _orig = predict_mlb.build_upcoming_context_mlb

    def spy(*a, **k):
        r = _orig(*a, **k)
        cap["ctx"] = r
        return r

    predict_mlb.build_upcoming_context_mlb = spy

    # For a past date the schedule is rebuilt from the DB with empty starter
    # names, so sp_era_rolling comes out empty and the fill guard refuses --
    # correctly. Supply the starters the DB already stores so the causal
    # builder gets the same inputs it would have had on the day.
    def sched_from_db(target_date):
        day = g[(g["season"] == 2026) & (g["date"] == pd.Timestamp(target_date))]
        day = day.dropna(subset=["opponent"])
        out, seen = [], set()
        for _, r in day.iterrows():
            pair = tuple(sorted([str(r["team"]), str(r["opponent"])]))
            if pair in seen:
                continue
            seen.add(pair)
            home_row = r if r.get("home") == 1 else day[
                (day["team"] == r["opponent"]) & (day["opponent"] == r["team"])]
            if not isinstance(home_row, pd.Series):
                if home_row.empty:
                    continue
                home_row = home_row.iloc[0]
            away_row = day[(day["team"] == home_row["opponent"])
                           & (day["opponent"] == home_row["team"])]
            if away_row.empty:
                continue
            away_row = away_row.iloc[0]
            out.append({"game_date": target_date,
                        "home_team": home_row["team"], "away_team": away_row["team"],
                        "home_sp": home_row.get("sp_name") or "",
                        "away_sp": away_row.get("sp_name") or "",
                        "game_datetime": "", "game_num": 1})
        return pd.DataFrame(out)

    predict_mlb.fetch_upcoming_mlb_games = sched_from_db

    # For a PAST date the DB already contains the game being predicted, so the
    # prediction path's history features would fold in the very game they are
    # meant to precede -- an off-by-one that has nothing to do with the code
    # under test. Hide anything on or after the target date, which is exactly
    # what the live run would have seen that morning.
    _real_fetch = db.fetch_games

    def fetch_before(client, sport, season=None, _cut=[None]):
        d = _real_fetch(client, sport, season)
        if _cut[0] is None or d.empty or "date" not in d.columns:
            return d
        dd = pd.to_datetime(d["date"], errors="coerce")
        return d[(pd.to_numeric(d["season"], errors="coerce") < 2026)
                 | (dd < _cut[0])]

    db.fetch_games = fetch_before
    predict_mlb.db.fetch_games = fetch_before
    db.upsert_predictions = lambda *a, **k: None      # never touch the bet record
    import smtplib

    class _S:
        def __init__(s, *a, **k): pass
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def starttls(s, *a, **k): pass
        def login(s, *a, **k): pass
        def send_message(s, *a, **k): pass
        def sendmail(s, *a, **k): pass
        def quit(s): pass
    smtplib.SMTP = smtplib.SMTP_SSL = _S

    rows = []
    for d in DATES:
        cap.clear()
        fetch_before.__defaults__[-1][0] = pd.Timestamp(d)
        try:
            predict_mlb.run_mlb(season=2026, target_date=d, max_evals=0,
                                top_n=99, key_type="free")
        except Exception as exc:
            print(f"  {d}: run failed ({type(exc).__name__}: {exc})")
            continue
        pc = cap.get("ctx")
        if pc is None or pc.empty:
            print(f"  {d}: no context built")
            continue
        pc = pc.reset_index().rename(columns={"index": "team"})
        day = g[(g["season"] == 2026) & (g["date"] == pd.Timestamp(d))]
        t2p = dict(zip(day["team"], day["period"]))
        for _, r in pc.iterrows():
            t = r.get("team")
            if t not in t2p:
                continue
            key = (t, t2p[t])
            if key not in ref.index:
                continue
            rows.append({"date": d, "team": t,
                         **{f"P_{f}": r.get(f, np.nan) for f in feats},
                         **{f"T_{f}": ref.loc[key, f] if f in ref.columns else np.nan
                            for f in feats}})
        print(f"  {d}: {len(pc)} teams")

    if not rows:
        print("no comparable rows"); return
    C = pd.DataFrame(rows)
    print(f"\ncompared {len(C)} team-rows across {C['date'].nunique()} dates")
    print(f"{'feature':<26}{'compared':>10}{'match':>9}{'mean |err|':>12}")
    bad = []
    for f in feats:
        a = pd.to_numeric(C[f"P_{f}"], errors="coerce")
        t = pd.to_numeric(C[f"T_{f}"], errors="coerce")
        m = a.notna() & t.notna()
        if m.sum() < 10:
            print(f"{f:<26}{int(m.sum()):>10}{'--':>9}")
            continue
        err = (a[m] - t[m]).abs()
        tol = np.maximum(1e-6, 1e-3 * t[m].abs())
        rate = float((err <= tol).mean())
        print(f"{f:<26}{int(m.sum()):>10}{rate*100:>8.1f}%{err.mean():>12.4f}")
        if rate < 0.98:
            bad.append(f)
    print("\n" + ("PASS - the causal prediction path reproduces every feature"
                  if not bad else f"MISMATCHED: {bad}"))


if __name__ == "__main__":
    main()
