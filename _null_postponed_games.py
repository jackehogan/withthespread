"""
Null the outcome fields on rows keyed to postponed game_pks.

43 game_pks are Postponed per statsapi yet carry results. Inspection shows each
row holds a REAL result -- but the two rows under one pk come from different
games, because the postponement was made up as a doubleheader and the seeder
attached one game of it to each side. Example pk 661235: the Twins row is a 4-5
loss, the Yankees row a 7-1 win.

Per-row the outcome is genuine; per-game the pairing is not, so every
game_pk-keyed opponent lookup on these 43 games resolves to the wrong game.
Outcome fields are cleared and `period` is deliberately left alone -- period is
part of the primary key, and it is no longer a model feature, so renumbering
would carry real risk for no gain.

    python _null_postponed_games.py           # dry run
    python _null_postponed_games.py --apply
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from pymongo import UpdateOne

import db

APPLY = "--apply" in sys.argv
AUDIT = (r"C:\Users\Jack\AppData\Local\Temp\claude"
         r"\C--Users-Jack-OneDrive-Documents-GitHub-WithTheSpread"
         r"\6366ec4c-69fb-46b7-9d5c-78324ebcad57\scratchpad\dh_audit.csv")
CLEAR = ["diff", "score", "opp_score", "spreadscore"]


def main():
    pks = pd.read_csv(AUDIT).query("status == 'Postponed'")["pk"].astype(int).tolist()
    client = db.connect()
    try:
        g = db.fetch_games(client, "mlb")
        g["game_pk"] = pd.to_numeric(g["game_pk"], errors="coerce")
        rows = g[g["game_pk"].isin(pks)]
        print(f"postponed game_pks: {len(pks)}   rows: {len(rows)}")
        for c in CLEAR:
            print(f"  {c:<14} currently set on {rows[c].notna().sum()}/{len(rows)}")
        print(f"  seasons: {rows['season'].value_counts().sort_index().to_dict()}")
        if not APPLY:
            print("\nDRY RUN — rerun with --apply to clear.")
            return
        col = client[db._MONGO_DB]["games"]
        ops = [UpdateOne({"sport": "mlb", "team": r["team"],
                          "season": int(r["season"]), "period": int(r["period"])},
                         {"$unset": {c: "" for c in CLEAR},
                          "$set": {"postponed_pk": True}})
               for _, r in rows.iterrows()]
        col.bulk_write(ops)
        print(f"\ncleared outcomes on {len(ops)} rows (flagged postponed_pk=True)")
    finally:
        client.close()


if __name__ == "__main__":
    main()
