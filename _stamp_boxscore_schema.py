"""
One-off: stamp rows that already hold complete boxscore data with the current
schema version.

seed_mlb now decides what to fetch from sp_box_v / bp_box_v rather than from a
data column. No existing row carries those fields, so without this every game
would look pending and the next nightly would re-fetch the whole season --
about 10 minutes per season against a 30-minute workflow timeout.

The data is already complete (verified 100% fill on every per-game column), so
stamping is a statement of fact, not a shortcut.

    python _stamp_boxscore_schema.py           # dry run
    python _stamp_boxscore_schema.py --apply
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import db
import seed_mlb as sm

APPLY = "--apply" in sys.argv
SP_FIELDS = ["sp_ip_game", "sp_er_game", "sp_k_game", "sp_bb_game",
             "sp_h_game", "sp_hr_game", "sp_pitch_game"]


def main():
    client = db.connect()
    try:
        col = client[db._MONGO_DB]["games"]
        sp_q = {"sport": "mlb", "sp_box_v": {"$exists": False},
                **{f: {"$exists": True} for f in SP_FIELDS}}
        bp_q = {"sport": "mlb", "bp_box_v": {"$exists": False},
                "bp_ip_game": {"$exists": True}}
        n_sp, n_bp = col.count_documents(sp_q), col.count_documents(bp_q)
        total = col.count_documents({"sport": "mlb"})
        print(f"mlb rows: {total}")
        print(f"  complete starter line, unstamped: {n_sp}  -> sp_box_v={sm._SP_BOX_SCHEMA}")
        print(f"  have bp_ip_game,        unstamped: {n_bp}  -> bp_box_v={sm._BP_BOX_SCHEMA}")
        if not APPLY:
            print("\nDRY RUN — rerun with --apply to write.")
            return
        r1 = col.update_many(sp_q, {"$set": {"sp_box_v": sm._SP_BOX_SCHEMA}})
        r2 = col.update_many(bp_q, {"$set": {"bp_box_v": sm._BP_BOX_SCHEMA}})
        print(f"stamped sp_box_v on {r1.modified_count}, bp_box_v on {r2.modified_count}")
        left = col.count_documents({"sport": "mlb", "sp_box_v": {"$exists": False}})
        print(f"rows still without sp_box_v (genuinely incomplete): {left}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
