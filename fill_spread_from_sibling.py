"""
Fill missing run lines by negating the opponent's.

MLB run lines are symmetric -- verified at 99.86% (10,697/10,712) on games where
both rows carry a spread. So when one row of a game has a spread and the other
does not, the missing side is exactly its negation, and
spreadscore = diff + spread (verified exact on all 23,177 populated rows).

This recovers rows no odds feed can supply: ESPN's pickcenter has no data before
2026, and the source that seeded 2025 dropped the Athletics on a name mismatch.

Only fills rows where spreadscore is currently NULL. Never overwrites.
Moneyline is deliberately NOT derived -- each side carries its own juice, so it
is not symmetric.

    python fill_spread_from_sibling.py           # dry run
    python fill_spread_from_sibling.py --apply   # write
"""
import argparse

import pandas as pd
from pymongo import UpdateOne

import db

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true", help="write to MongoDB")
args = parser.parse_args()

client = db.connect()
try:
    df = db.fetch_games(client, "mlb")

    g = df[df["game_pk"].notna() & (df["game_pk"].astype(str) != "")].copy()
    g["game_pk"] = g["game_pk"].astype(str)

    # Sibling spread lookup: game_pk -> {team: spread}
    have = g[g["spread"].notna()]
    sib: dict[str, dict[str, float]] = {}
    for pk, team, sp in zip(have["game_pk"], have["team"], have["spread"]):
        sib.setdefault(pk, {})[team] = float(sp)

    targets = g[g["spreadscore"].isna() & g["diff"].notna()]

    ops, preview = [], []
    skipped_no_sib = 0
    for _, r in targets.iterrows():
        others = {t: s for t, s in sib.get(r["game_pk"], {}).items() if t != r["team"]}
        if len(others) != 1:
            skipped_no_sib += 1
            continue
        opp_team, opp_spread = next(iter(others.items()))
        spread = -opp_spread
        ss = round(float(r["diff"]) + spread, 4)

        ops.append(UpdateOne(
            {"sport": "mlb", "team": r["team"],
             "season": int(r["season"]), "period": int(r["period"])},
            # spread_derived marks these as inferred from the opponent rather
            # than observed from a book — provenance that cannot be
            # reconstructed once the values are written.
            {"$set": {"spread": spread, "spreadscore": ss,
                      "spread_derived": True}},
            upsert=False,
        ))
        preview.append({
            "season": int(r["season"]), "date": str(r["date"])[:10],
            "team": r["team"], "diff": int(r["diff"]),
            "opp": opp_team, "opp_spread": opp_spread,
            "new_spread": spread, "new_spreadscore": ss,
        })

    pv = pd.DataFrame(preview)
    print(f"rows missing spreadscore : {len(targets)}")
    print(f"fillable from sibling    : {len(ops)}")
    print(f"skipped (no usable sibling): {skipped_no_sib}\n")

    if not pv.empty:
        print("By season:")
        print(pv.groupby("season").size().to_string())
        print("\nBy team (top 10):")
        print(pv.groupby("team").size().sort_values(ascending=False).head(10).to_string())
        print("\nSample of 12 rows to be written:")
        print(pv.head(12).to_string(index=False))
        print("\nDerived spread distribution:")
        print(pv["new_spread"].value_counts().to_string())

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
    elif ops:
        res = client[db._MONGO_DB]["games"].bulk_write(ops)
        print(f"\nWrote {res.modified_count} rows.")
    else:
        print("\nNothing to write.")
finally:
    client.close()
