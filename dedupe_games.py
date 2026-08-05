"""
Remove duplicate game rows created by the period-assignment bug.

Incremental --since seeds re-fetched games that were already stored and gave
them fresh period numbers, filing the same game_pk under a second period. The
root cause is fixed (period is now keyed on game identity), but the rows it
already created remain.

For each (season, team, game_pk) stored more than once:
  - keep the LOWEST period -- that is the original, correct slot. The
    duplicate was written at max_period + offset, so it always sits higher.
    Verified against the calendar: a mid-July 2026 game belongs near period
    97, not 121.
  - before deleting, copy across any field the keeper is missing but a
    sibling has, so no data is lost.
  - delete the remaining rows.

Periods are NOT renumbered afterwards. Renumbering would rewrite every row's
period and break the predictions collection, which stores period to resolve
results. The resulting gaps are harmless -- the SS pivot simply sees one fewer
observation and XGBoost handles the NaN.

    python dedupe_games.py           # dry run
    python dedupe_games.py --apply   # delete
"""
import argparse

import pandas as pd

import db

MERGE_FIELDS = [
    "spread", "moneyline", "spreadscore", "over_under", "ml_odds",
    "open_spread", "open_moneyline", "sp_name", "sp_era", "sp_whip", "sp_k9",
    "sp_ip_per_start", "bp_era", "bp_whip", "bp_k9", "bp_hr9",
    "bp_ip_per_game", "bp_ip_game", "sp_ip_game", "sp_er_game",
]

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()

client = db.connect()
try:
    col = client[db._MONGO_DB]["games"]
    df = db.fetch_games(client, "mlb")
    df = df[df["game_pk"].notna() & (df["game_pk"].astype(str) != "")].copy()
    df["game_pk"] = df["game_pk"].astype(str)

    counts = df.groupby(["season", "team", "game_pk"]).size()
    dupes = counts[counts > 1]
    print(f"duplicated (season, team, game_pk) combos: {len(dupes)}")
    print(f"excess rows to remove: {int(dupes.sum() - len(dupes))}\n")
    if dupes.empty:
        print("Nothing to do.")
        raise SystemExit(0)

    print("By season:")
    print(dupes.reset_index().groupby("season").size().to_string())

    delete_keys, merge_ops, rescued = [], [], 0
    for (season, team, pk), _ in dupes.items():
        rows = df[(df["season"] == season) & (df["team"] == team)
                  & (df["game_pk"] == pk)].sort_values("period")
        keeper, others = rows.iloc[0], rows.iloc[1:]

        patch = {}
        for f in MERGE_FIELDS:
            if f not in rows.columns:
                continue
            if pd.isna(keeper.get(f)):
                vals = others[f].dropna()
                if not vals.empty:
                    v = vals.iloc[0]
                    patch[f] = float(v) if isinstance(v, (int, float)) else v
        if patch:
            rescued += 1
            merge_ops.append(({"sport": "mlb", "team": team,
                               "season": int(season),
                               "period": int(keeper["period"])}, patch))

        for _, r in others.iterrows():
            delete_keys.append({"sport": "mlb", "team": team,
                                "season": int(season), "period": int(r["period"])})

    print(f"\nrows to delete            : {len(delete_keys)}")
    print(f"keepers gaining data first: {rescued}")

    sample = pd.DataFrame(delete_keys[:10])
    if not sample.empty:
        print("\nsample of rows to be deleted:")
        print(sample.to_string(index=False))

    if not args.apply:
        print("\nDRY RUN — nothing deleted. Re-run with --apply.")
    else:
        from pymongo import UpdateOne
        if merge_ops:
            col.bulk_write([UpdateOne(k, {"$set": v}) for k, v in merge_ops])
            print(f"\nmerged data into {len(merge_ops)} keepers")
        n = 0
        for k in delete_keys:
            n += col.delete_one(k).deleted_count
        print(f"deleted {n} duplicate rows")
finally:
    client.close()
