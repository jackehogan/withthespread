"""
ProphetX Trading API client.

Handles authentication, market discovery, and bet placement for the
ProphetX exchange (prophetx.co). Uses the MM (Market Maker) API which
requires an access_key / secret_key issued by ProphetX.

Setup
-----
Add a "prophetx" block to data/config.txt:
    {
      "prophetx": {
        "access_key":     "your_access_key",
        "secret_key":     "your_secret_key",
        "bankroll":       1000.0,
        "kelly_fraction": 0.10,
        "sandbox":        false,
        "enabled":        true,
        "min_ev":         0.05,
        "max_stake":      50.0
      }
    }

Request API access at: https://docs.prophetx.co/docs/getting-started
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SANDBOX_BASE = "https://api-ss-sandbox.betprophet.co"
_PROD_BASE    = "https://api.betprophet.co"

# ProphetX team name → our model team name
# Adjust if ProphetX uses different abbreviations.
_PX_TO_MODEL: dict[str, str] = {
    "Arizona Diamondbacks":    "Arizona Diamondbacks",
    "Atlanta Braves":          "Atlanta Braves",
    "Baltimore Orioles":       "Baltimore Orioles",
    "Boston Red Sox":          "Boston Red Sox",
    "Chicago Cubs":            "Chicago Cubs",
    "Chicago White Sox":       "Chicago White Sox",
    "Cincinnati Reds":         "Cincinnati Reds",
    "Cleveland Guardians":     "Cleveland Guardians",
    "Colorado Rockies":        "Colorado Rockies",
    "Detroit Tigers":          "Detroit Tigers",
    "Houston Astros":          "Houston Astros",
    "Kansas City Royals":      "Kansas City Royals",
    "Los Angeles Angels":      "Los Angeles Angels",
    "Los Angeles Dodgers":     "Los Angeles Dodgers",
    "Miami Marlins":           "Miami Marlins",
    "Milwaukee Brewers":       "Milwaukee Brewers",
    "Minnesota Twins":         "Minnesota Twins",
    "New York Mets":           "New York Mets",
    "New York Yankees":        "New York Yankees",
    "Athletics":               "Athletics",
    "Oakland Athletics":       "Athletics",
    "Philadelphia Phillies":   "Philadelphia Phillies",
    "Pittsburgh Pirates":      "Pittsburgh Pirates",
    "San Diego Padres":        "San Diego Padres",
    "San Francisco Giants":    "San Francisco Giants",
    "Seattle Mariners":        "Seattle Mariners",
    "St. Louis Cardinals":     "St. Louis Cardinals",
    "Tampa Bay Rays":          "Tampa Bay Rays",
    "Texas Rangers":           "Texas Rangers",
    "Toronto Blue Jays":       "Toronto Blue Jays",
    "Washington Nationals":    "Washington Nationals",
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ProphetXClient:
    """
    Thin wrapper around the ProphetX MM API.

    Usage
    -----
        client = ProphetXClient.from_config()
        markets = client.get_mlb_run_lines(target_date)
        results = client.place_model_bets(predictions_df, markets, target_date)
    """

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.10,
        sandbox: bool = False,
        min_ev: float = 0.05,
        max_stake: float = 50.0,
    ) -> None:
        self.access_key     = access_key
        self.secret_key     = secret_key
        self.bankroll       = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_ev         = min_ev
        self.max_stake      = max_stake
        self.base_url       = _SANDBOX_BASE if sandbox else _PROD_BASE
        self.sandbox        = sandbox

        self._access_token:  Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry:  float = 0.0   # epoch seconds

        self._odds_ladder:   Optional[list[int]] = None

    # ------------------------------------------------------------------
    # Class method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str = "data/config.txt") -> "ProphetXClient":
        """Build from the 'prophetx' block in config.txt."""
        with open(config_path) as f:
            cfg = json.load(f)
        px = cfg.get("prophetx", {})
        if not px:
            raise KeyError("No 'prophetx' block found in config.txt.")
        return cls(
            access_key     = px["access_key"],
            secret_key     = px["secret_key"],
            bankroll       = float(px.get("bankroll",       1000.0)),
            kelly_fraction = float(px.get("kelly_fraction", 0.10)),
            sandbox        = bool(px.get("sandbox",         False)),
            min_ev         = float(px.get("min_ev",         0.05)),
            max_stake      = float(px.get("max_stake",      50.0)),
        )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _login(self) -> None:
        """Obtain a fresh access token."""
        resp = requests.post(
            f"{self.base_url}/partner/auth/login",
            json={"access_key": self.access_key, "secret_key": self.secret_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        self._access_token  = data["access_token"]
        self._refresh_token = data["refresh_token"]
        self._token_expiry  = time.time() + 18 * 60  # refresh 2 min before 20-min TTL

    def _refresh(self) -> None:
        """Refresh the access token using the stored refresh token."""
        resp = requests.post(
            f"{self.base_url}/partner/auth/refresh",
            json={"refresh_token": self._refresh_token},
            headers={"Authorization": f"Bearer {self._access_token}"},
            timeout=15,
        )
        resp.raise_for_status()
        self._access_token = resp.json()["data"]["access_token"]
        self._token_expiry = time.time() + 18 * 60

    def _token(self) -> str:
        """Return a valid access token, refreshing/logging in as needed."""
        if self._access_token is None:
            self._login()
        elif time.time() > self._token_expiry:
            try:
                self._refresh()
            except Exception:
                self._login()
        return self._access_token  # type: ignore[return-value]

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token()}"}

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = requests.get(
            f"{self.base_url}/{path}", headers=self._headers(),
            params=params, timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}/{path}", headers=self._headers(),
            json=body, timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        """Return available account balance in dollars."""
        data = self._get("partner/mm/get_balance")
        return float(data["data"]["balance"])

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def _get_mlb_tournament_id(self) -> int:
        """Return the ProphetX tournament ID for MLB."""
        data = self._get("partner/mm/get_tournaments")
        for t in data["data"]:
            if "MLB" in t.get("name", "").upper():
                return int(t["id"])
        raise ValueError(f"MLB tournament not found. Available: {[t.get('name') for t in data['data']]}")

    def _get_events_for_date(self, tournament_id: int, target_date: str) -> list[dict]:
        """Return all sport events for the given date (YYYY-MM-DD)."""
        data = self._get(
            "partner/mm/get_sport_events",
            params={"tournament_id": tournament_id},
        )
        events = data.get("data", [])
        # Filter by date — event start times are ISO strings
        day_events = [
            e for e in events
            if e.get("start_time", "").startswith(target_date)
        ]
        return day_events

    def get_mlb_run_lines(self, target_date: str) -> dict[str, dict]:
        """
        Return a dict mapping model team name → market info for MLB run lines
        on the given date.

        Keys in each value dict:
            line_id   : int   — ProphetX line identifier
            px_odds   : int   — current American odds on the exchange
            team      : str   — model team name
            px_team   : str   — raw ProphetX team name
        """
        tournament_id = self._get_mlb_tournament_id()
        events = self._get_events_for_date(tournament_id, target_date)
        if not events:
            logger.warning("ProphetX: no MLB events found for %s", target_date)
            return {}

        event_ids = [str(e["id"]) for e in events]

        # Fetch markets in one batch call
        data = self._get(
            "partner/mm/get_multiple_markets",
            params={"event_ids": ",".join(event_ids)},
        )

        markets: dict[str, dict] = {}
        event_id_to_event = {str(e["id"]): e for e in events}

        for event_id_str, event_markets in data.get("data", {}).items():
            event = event_id_to_event.get(event_id_str, {})
            participants = event.get("participants", [])  # [{name, side: "home"/"away"}]

            for market in event_markets.get("markets", []):
                if market.get("type") != "spread":
                    continue  # only run lines
                for selection_group in market.get("selections", []):
                    for sel in (selection_group if isinstance(selection_group, list) else [selection_group]):
                        px_team = sel.get("name") or sel.get("team_name") or ""
                        model_team = _PX_TO_MODEL.get(px_team, px_team)
                        line_id = sel.get("line_id")
                        px_odds = sel.get("odds")
                        if line_id and px_odds:
                            markets[model_team] = {
                                "line_id":  int(line_id),
                                "px_odds":  int(px_odds),
                                "team":     model_team,
                                "px_team":  px_team,
                            }

        logger.info("ProphetX: found run-line markets for %d teams on %s", len(markets), target_date)
        return markets

    # ------------------------------------------------------------------
    # Odds ladder
    # ------------------------------------------------------------------

    def _load_odds_ladder(self) -> list[int]:
        """Fetch and cache the valid ProphetX odds ladder."""
        if self._odds_ladder is None:
            data = self._get("partner/mm/get_odds_ladder")
            self._odds_ladder = [int(o) for o in data["data"]]
        return self._odds_ladder

    def snap_to_ladder(self, odds: float) -> int:
        """
        Snap American odds to the nearest valid value on the ProphetX odds ladder.
        ProphetX rejects any odds not on the ladder.
        """
        ladder = self._load_odds_ladder()
        odds_int = int(round(odds))
        return min(ladder, key=lambda x: abs(x - odds_int))

    # ------------------------------------------------------------------
    # Kelly sizing
    # ------------------------------------------------------------------

    def _kelly_stake(self, ev: float, odds: float) -> float:
        """
        Fractional Kelly stake in dollars.
            stake = bankroll * kelly_fraction * ev / payout
        Capped at max_stake and rounded to 2 decimal places.
        """
        try:
            if np.isnan(ev) or ev <= 0 or np.isnan(odds):
                return 0.0
            payout = 100.0 / abs(odds) if odds < 0 else odds / 100.0
            if payout <= 0:
                return 0.0
            stake = self.bankroll * self.kelly_fraction * ev / payout
            stake = min(stake, self.max_stake)
            return round(max(stake, 0.0), 2)
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Bet placement
    # ------------------------------------------------------------------

    def place_wager(
        self,
        line_id: int,
        odds: int,
        stake: float,
        external_id: str | None = None,
    ) -> dict:
        """
        Place a single wager.

        Parameters
        ----------
        line_id     : ProphetX line identifier from get_mlb_run_lines().
        odds        : American odds — must be on the ProphetX odds ladder.
        stake       : Dollar amount to wager.
        external_id : Idempotency key (UUID). Auto-generated if not provided.

        Returns
        -------
        ProphetX response dict with wager details.
        """
        if external_id is None:
            external_id = str(uuid.uuid4())
        body = {
            "external_id": external_id,
            "line_id":     line_id,
            "odds":        odds,
            "stake":       stake,
        }
        return self._post("partner/mm/place_wager", body)

    def place_model_bets(
        self,
        preds: pd.DataFrame,
        markets: dict[str, dict],
        target_date: str,
    ) -> pd.DataFrame:
        """
        Cross-reference model predictions with ProphetX markets and place bets
        on all positive-EV picks above min_ev.

        Parameters
        ----------
        preds       : DataFrame from predict_mlb._run() — indexed by team name,
                      must have columns: ev, bet, moneyline, ml_odds, coverprob.
        markets     : Dict from get_mlb_run_lines() — team → {line_id, px_odds}.
        target_date : 'YYYY-MM-DD' — used as part of the idempotency key.

        Returns
        -------
        DataFrame with one row per attempted bet and columns:
            team, bet, ev, stake, line_id, requested_odds, snapped_odds,
            status, wager_id, error
        """
        results = []

        # Filter to positive-EV bets above threshold
        ev_bets = preds[preds["ev"] >= self.min_ev].copy()
        if ev_bets.empty:
            logger.info("ProphetX: no bets meet min_ev=%.3f threshold.", self.min_ev)
            return pd.DataFrame()

        balance = self.get_balance()
        logger.info("ProphetX: account balance = $%.2f", balance)

        for team, row in ev_bets.iterrows():
            bet_market = row.get("bet", "SPREAD")
            ev         = float(row.get("ev",  float("nan")))
            juice      = float(row.get("ml_odds", float("nan"))) if bet_market == "ML" \
                         else float(row.get("moneyline", float("nan")))

            if team not in markets:
                results.append({
                    "team": team, "bet": bet_market, "ev": ev,
                    "stake": 0.0, "line_id": None,
                    "requested_odds": juice, "snapped_odds": None,
                    "status": "NO_MARKET", "wager_id": None,
                    "error": "Team not found in ProphetX markets",
                })
                continue

            market    = markets[team]
            line_id   = market["line_id"]
            stake     = self._kelly_stake(ev, juice)

            if stake <= 0:
                results.append({
                    "team": team, "bet": bet_market, "ev": ev,
                    "stake": 0.0, "line_id": line_id,
                    "requested_odds": juice, "snapped_odds": None,
                    "status": "ZERO_STAKE", "wager_id": None,
                    "error": "Kelly stake computed to zero",
                })
                continue

            if stake > balance:
                results.append({
                    "team": team, "bet": bet_market, "ev": ev,
                    "stake": stake, "line_id": line_id,
                    "requested_odds": juice, "snapped_odds": None,
                    "status": "INSUFFICIENT_FUNDS", "wager_id": None,
                    "error": f"Stake ${stake:.2f} > balance ${balance:.2f}",
                })
                continue

            # Snap odds to valid ProphetX ladder
            snapped = self.snap_to_ladder(juice)

            # Idempotency key: deterministic per team+date so re-runs don't double-bet
            ext_id = f"wts-{target_date}-{team.lower().replace(' ', '-')}"

            try:
                resp    = self.place_wager(line_id, snapped, stake, external_id=ext_id)
                wager   = resp.get("data", {}).get("wager", resp.get("data", {}))
                wager_id = wager.get("id") or wager.get("wager_id")
                balance -= stake   # deduct from local balance tracker
                results.append({
                    "team": team, "bet": bet_market, "ev": ev,
                    "stake": stake, "line_id": line_id,
                    "requested_odds": juice, "snapped_odds": snapped,
                    "status": "PLACED", "wager_id": wager_id,
                    "error": None,
                })
                logger.info(
                    "ProphetX: PLACED  %s %s  odds=%d  stake=$%.2f  wager_id=%s",
                    team, bet_market, snapped, stake, wager_id,
                )
            except requests.HTTPError as exc:
                err = str(exc)
                try:
                    err = exc.response.json().get("message", err)
                except Exception:
                    pass
                results.append({
                    "team": team, "bet": bet_market, "ev": ev,
                    "stake": stake, "line_id": line_id,
                    "requested_odds": juice, "snapped_odds": snapped,
                    "status": "ERROR", "wager_id": None,
                    "error": err,
                })
                logger.error("ProphetX: FAILED  %s  — %s", team, err)

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Convenience: load enabled flag from config without constructing client
# ---------------------------------------------------------------------------

def is_enabled(config_path: str = "data/config.txt") -> bool:
    """Return True if ProphetX betting is configured and enabled."""
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        px = cfg.get("prophetx", {})
        return bool(px.get("enabled", False)) and bool(px.get("access_key"))
    except Exception:
        return False
