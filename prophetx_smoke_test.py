"""
Sandbox smoke test for the ProphetX integration.

Walks the whole path in order, stopping at the first failure so the broken
step is obvious rather than buried:

    config -> auth -> balance -> tournament -> events -> markets
           -> odds ladder -> dry-run placement

Refuses to run unless BOTH safety flags are set for a no-money run:
    sandbox = true   (routes to api-ss-sandbox.betprophet.co)
    dry_run = true   (logs intended wagers, sends nothing)

Pass --allow-live only if you have deliberately turned those off and intend
real wagers. That is your decision to make, not something this script should
let you reach by accident.

    python prophetx_smoke_test.py
    python prophetx_smoke_test.py --date 2026-08-06
"""
import argparse
import datetime
import json
import sys
import traceback

import pandas as pd

import db
import prophetx as px_mod


def step(n, label):
    print(f"\n[{n}] {label}")
    print("    " + "-" * (len(label) + 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--allow-live", action="store_true",
                    help="permit running with sandbox/dry_run disabled")
    args = ap.parse_args()

    step(1, "Config block")
    try:
        cfg = json.load(open("data/config.txt")).get("prophetx")
    except Exception as exc:
        print(f"    FAIL reading data/config.txt: {exc}")
        return 1
    if not cfg:
        print("    FAIL no `prophetx` block in data/config.txt.")
        print("    See _prophetx_config_template.json.")
        return 1
    if not cfg.get("access_key") or "PASTE_YOUR" in str(cfg.get("access_key")):
        print("    FAIL access_key not filled in.")
        return 1

    sandbox = bool(cfg.get("sandbox", True))
    dry_run = bool(cfg.get("dry_run", True))
    print(f"    sandbox={sandbox}  dry_run={dry_run}  "
          f"enabled={cfg.get('enabled')}  min_ev={cfg.get('min_ev')}")
    if not (sandbox and dry_run) and not args.allow_live:
        print("\n    STOPPED. sandbox and dry_run are not both true, so this run")
        print("    could place real wagers. Re-run with --allow-live only if")
        print("    that is what you intend.")
        return 1
    print("    OK — no money can move in this configuration.")

    step(2, "Client + authentication")
    try:
        px = px_mod.ProphetXClient.from_config()
        print(f"    base_url = {px.base_url}")
        tok = px._token()
        print(f"    OK token acquired ({len(tok)} chars)")
    except Exception as exc:
        print(f"    FAIL {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)
        return 1

    step(3, "Account balance")
    try:
        print(f"    OK balance = ${px.get_balance():.2f}")
    except Exception as exc:
        print(f"    FAIL {type(exc).__name__}: {exc}")
        return 1

    step(4, "MLB tournament id")
    try:
        tid = px._get_mlb_tournament_id()
        print(f"    OK tournament_id = {tid}")
    except Exception as exc:
        print(f"    FAIL {type(exc).__name__}: {exc}")
        return 1

    step(5, f"Events for {args.date}")
    try:
        events = px._get_events_for_date(tid, args.date)
        print(f"    OK {len(events)} event(s)")
        if not events:
            print("    NOTE none listed — try --date tomorrow, or the sandbox")
            print("    may not mirror the live schedule.")
    except Exception as exc:
        print(f"    FAIL {type(exc).__name__}: {exc}")
        return 1

    step(6, "Run-line markets")
    try:
        markets = px.get_mlb_run_lines(args.date)
        print(f"    OK markets for {len(markets)} team(s)")
        for t, m in list(markets.items())[:5]:
            print(f"      {t:<26} line_id={m['line_id']}  odds={m['px_odds']:+d}")
    except Exception as exc:
        print(f"    FAIL {type(exc).__name__}: {exc}")
        return 1

    step(7, "Odds ladder")
    try:
        ladder = px._load_odds_ladder()
        print(f"    OK {len(ladder)} rungs, e.g. {sorted(ladder)[:6]} ...")
        for probe in (137, -142):
            print(f"      snap {probe:+d} -> {px.snap_to_ladder(probe):+d}")
    except Exception as exc:
        print(f"    FAIL {type(exc).__name__}: {exc}")
        return 1

    step(8, f"Dry-run placement against today's predictions")
    client = db.connect()
    try:
        preds = db.fetch_predictions(client, "mlb")
    finally:
        client.close()
    todays = preds[preds["prediction_date"] == args.date]
    if todays.empty:
        print(f"    no stored predictions for {args.date} — skipping.")
        return 0
    todays = todays.set_index("team")
    print(f"    {len(todays)} prediction(s); "
          f"{(todays.get('bet') == 'SPREAD').sum()} SPREAD, "
          f"{(todays.get('bet') == 'ML').sum()} ML")
    try:
        res = px.place_model_bets(todays, markets, args.date)
    except Exception as exc:
        print(f"    FAIL {type(exc).__name__}: {exc}")
        return 1
    if res.empty:
        print("    no rows returned.")
        return 0
    cols = [c for c in ("team", "bet", "ev", "stake", "snapped_odds", "status")
            if c in res.columns]
    print(res[cols].to_string(index=False))
    print(f"\n    status counts: {dict(res['status'].value_counts())}")
    placed = res[res["status"] == "PLACED"]
    if not placed.empty:
        print(f"\n    WARNING {len(placed)} wager(s) were actually PLACED.")
    else:
        print("\n    OK nothing was placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
