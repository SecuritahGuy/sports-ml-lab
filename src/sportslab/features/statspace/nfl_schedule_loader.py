"""Historical NFL schedule and QB outcome loaders using nfl_data_py."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

NFLVERSE_WEEKLY_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{year}.parquet"
)


def infer_nfl_season_start_year(game_date: str) -> int:
    """Infer NFL season start year from an ISO date using July rollover."""
    parsed = datetime.fromisoformat(game_date[:10])
    return parsed.year if parsed.month >= 7 else parsed.year - 1


@dataclass
class NFLHistoricalGame:
    """Normalized historical NFL game and starter-QB context."""

    game_id: str
    date: str
    season: int
    week: int
    game_type: str
    away_team: str
    home_team: str
    away_score: int
    home_score: int
    completed: bool
    status: str
    away_qb_id: Optional[str] = None
    home_qb_id: Optional[str] = None
    away_qb_name: Optional[str] = None
    home_qb_name: Optional[str] = None
    away_qb_source: str = "missing"
    home_qb_source: str = "missing"
    away_qb_attempts: Optional[float] = None
    home_qb_attempts: Optional[float] = None
    away_qb_completions: Optional[float] = None
    home_qb_completions: Optional[float] = None
    away_qb_passing_yards: Optional[float] = None
    home_qb_passing_yards: Optional[float] = None
    away_qb_passing_tds: Optional[float] = None
    home_qb_passing_tds: Optional[float] = None
    away_qb_interceptions: Optional[float] = None
    home_qb_interceptions: Optional[float] = None
    away_qb_sacks: Optional[float] = None
    home_qb_sacks: Optional[float] = None
    away_qb_passing_epa: Optional[float] = None
    home_qb_passing_epa: Optional[float] = None
    away_qb_epa_per_play: Optional[float] = None
    home_qb_epa_per_play: Optional[float] = None

    @property
    def home_win(self) -> float:
        """Binary/tie label for home result."""
        if self.home_score > self.away_score:
            return 1.0
        if self.home_score < self.away_score:
            return 0.0
        return 0.5

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dictionary."""
        payload = asdict(self)
        payload["home_win"] = self.home_win
        return payload


class NFLScheduleLoader:
    """Load historical NFL schedules and per-game starter QB context."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def load_seasons(
        self,
        years: Iterable[int],
        *,
        include_playoffs: bool = False,
    ) -> list[NFLHistoricalGame]:
        """Load and normalize seasons from nfl_data_py."""
        import nfl_data_py as nfl

        requested_years = sorted({int(year) for year in years})
        if not requested_years:
            return []

        schedules = nfl.import_schedules(requested_years)
        weekly = import_weekly_player_data(requested_years)
        return self.normalize_data(
            schedules,
            weekly,
            include_playoffs=include_playoffs,
        )

    def normalize_data(
        self,
        schedules_df: Any,
        weekly_df: Any,
        *,
        include_playoffs: bool = False,
    ) -> list[NFLHistoricalGame]:
        """Normalize nflverse schedule and weekly tables into games."""
        import pandas as pd

        schedules = schedules_df.copy()
        weekly = weekly_df.copy()

        games: list[NFLHistoricalGame] = []
        if schedules.empty:
            return games

        schedules = schedules.where(pd.notna(schedules), None)
        weekly = weekly.where(pd.notna(weekly), None)
        weekly_index = self._build_weekly_qb_index(weekly)

        for _, row in schedules.iterrows():
            game_type = str(
                row.get("game_type") or row.get("season_type") or "REG"
            ).upper()
            if not include_playoffs and game_type not in {"REG", "R"}:
                continue

            date_iso = str(
                row.get("gameday")
                or row.get("game_date")
                or row.get("date")
                or ""
            )[:10]
            if not date_iso:
                continue

            home_team = self._normalize_team_code(row.get("home_team"))
            away_team = self._normalize_team_code(row.get("away_team"))
            if not home_team or not away_team:
                continue

            home_score = self._safe_int(
                row.get("home_score")
                if row.get("home_score") is not None
                else row.get("home_points")
            )
            away_score = self._safe_int(
                row.get("away_score")
                if row.get("away_score") is not None
                else row.get("away_points")
            )
            if home_score is None or away_score is None:
                continue

            status = str(
                row.get("status")
                or row.get("result")
                or row.get("game_status")
                or "completed"
            )
            completed = bool(
                row.get("result") is not None
                or row.get("home_score") is not None
            )
            season = self._safe_int(
                row.get("season")
            ) or infer_nfl_season_start_year(date_iso)
            week = self._safe_int(row.get("week")) or 0
            game_id = str(
                row.get("game_id")
                or row.get("gsis_id")
                or f"{date_iso}_{away_team}_{home_team}"
            )

            home_qb = self._resolve_team_qb(
                row=row,
                weekly_index=weekly_index,
                season=season,
                week=week,
                side="home",
                team=home_team,
            )
            away_qb = self._resolve_team_qb(
                row=row,
                weekly_index=weekly_index,
                season=season,
                week=week,
                side="away",
                team=away_team,
            )

            games.append(
                NFLHistoricalGame(
                    game_id=game_id,
                    date=date_iso,
                    season=season,
                    week=week,
                    game_type=game_type,
                    away_team=away_team,
                    home_team=home_team,
                    away_score=away_score,
                    home_score=home_score,
                    completed=completed,
                    status=status,
                    away_qb_id=away_qb.get("qb_id"),
                    home_qb_id=home_qb.get("qb_id"),
                    away_qb_name=away_qb.get("qb_name"),
                    home_qb_name=home_qb.get("qb_name"),
                    away_qb_source=str(away_qb.get("source") or "missing"),
                    home_qb_source=str(home_qb.get("source") or "missing"),
                    away_qb_attempts=away_qb.get("attempts"),
                    home_qb_attempts=home_qb.get("attempts"),
                    away_qb_completions=away_qb.get("completions"),
                    home_qb_completions=home_qb.get("completions"),
                    away_qb_passing_yards=away_qb.get("passing_yards"),
                    home_qb_passing_yards=home_qb.get("passing_yards"),
                    away_qb_passing_tds=away_qb.get("passing_tds"),
                    home_qb_passing_tds=home_qb.get("passing_tds"),
                    away_qb_interceptions=away_qb.get("interceptions"),
                    home_qb_interceptions=home_qb.get("interceptions"),
                    away_qb_sacks=away_qb.get("sacks"),
                    home_qb_sacks=home_qb.get("sacks"),
                    away_qb_passing_epa=away_qb.get("passing_epa"),
                    home_qb_passing_epa=home_qb.get("passing_epa"),
                    away_qb_epa_per_play=away_qb.get("epa_per_play"),
                    home_qb_epa_per_play=home_qb.get("epa_per_play"),
                )
            )

        games.sort(key=lambda item: (item.date, item.week, item.game_id))
        return games

    @staticmethod
    def to_dataframe(games: Iterable[NFLHistoricalGame]) -> "Any":
        """Convert games to a pandas DataFrame lazily."""
        import pandas as pd

        return pd.DataFrame([game.to_dict() for game in games])

    def _build_weekly_qb_index(
        self, weekly: Any
    ) -> dict[tuple[int, int, str], list[dict[str, Any]]]:
        """Index weekly player rows by season, week, and team."""
        qbs: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        if getattr(weekly, "empty", True):
            return qbs

        for _, row in weekly.iterrows():
            position = str(
                row.get("position") or row.get("position_group") or ""
            ).upper()
            attempts = self._safe_float(row.get("attempts")) or 0.0
            if position and position != "QB" and attempts <= 0.0:
                continue

            season = self._safe_int(row.get("season"))
            week = self._safe_int(row.get("week"))
            team = self._normalize_team_code(
                row.get("recent_team")
                or row.get("team")
                or row.get("team_abbr")
            )
            if season is None or week is None or not team:
                continue

            qbs.setdefault((season, week, team), []).append(
                self._weekly_qb_payload(row)
            )

        for key, players in qbs.items():
            players.sort(
                key=lambda item: (
                    float(item.get("attempts") or 0.0),
                    float(item.get("passing_yards") or 0.0),
                    float(item.get("passing_epa") or -math.inf),
                ),
                reverse=True,
            )
        return qbs

    def _resolve_team_qb(
        self,
        *,
        row: Any,
        weekly_index: dict[tuple[int, int, str], list[dict[str, Any]]],
        season: int,
        week: int,
        side: str,
        team: str,
    ) -> dict[str, Any]:
        """Resolve a side's starter from explicit or fallback weekly data."""
        explicit = self._extract_explicit_qb(row=row, side=side)
        if explicit.get("qb_id") or explicit.get("qb_name"):
            explicit_player = self._match_explicit_qb(
                explicit=explicit,
                candidates=weekly_index.get((season, week, team), []),
            )
            if explicit_player is not None:
                explicit_player["source"] = "explicit"
                return explicit_player
            explicit["source"] = "explicit"
            explicit.setdefault("epa_per_play", None)
            return explicit

        candidates = weekly_index.get((season, week, team), [])
        if not candidates:
            return {"source": "missing"}

        resolved = dict(candidates[0])
        resolved["source"] = "fallback_attempts"
        return resolved

    def _extract_explicit_qb(self, *, row: Any, side: str) -> dict[str, Any]:
        """Extract explicit game-level QB identifiers when present."""
        candidates_id = [
            f"{side}_qb_id",
            f"{side}_starter_id",
            f"{side}_passer_id",
        ]
        candidates_name = [
            f"{side}_qb_name",
            f"{side}_starter_name",
            f"{side}_passer",
        ]
        payload = {
            "qb_id": None,
            "qb_name": None,
            "attempts": None,
            "completions": None,
            "passing_yards": None,
            "passing_tds": None,
            "interceptions": None,
            "sacks": None,
            "passing_epa": None,
            "epa_per_play": None,
        }
        for key in candidates_id:
            value = row.get(key)
            if value is not None:
                payload["qb_id"] = str(value)
                break
        for key in candidates_name:
            value = row.get(key)
            if value is not None:
                payload["qb_name"] = str(value)
                break
        return payload

    def _match_explicit_qb(
        self,
        *,
        explicit: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Match an explicit starter to weekly stats by id or player name."""
        qb_id = str(explicit.get("qb_id") or "")
        qb_name = str(explicit.get("qb_name") or "").strip().lower()

        for candidate in candidates:
            candidate_id = str(candidate.get("qb_id") or "")
            candidate_name = (
                str(candidate.get("qb_name") or "").strip().lower()
            )
            if qb_id and candidate_id and qb_id == candidate_id:
                return dict(candidate)
            if qb_name and candidate_name and qb_name == candidate_name:
                return dict(candidate)
        return None

    def _weekly_qb_payload(self, row: Any) -> dict[str, Any]:
        """Project weekly QB row into the normalized starter payload."""
        attempts = self._safe_float(row.get("attempts"))
        sacks = self._safe_float(row.get("sacks"))
        passing_epa = self._safe_float(
            row.get("passing_epa")
            if row.get("passing_epa") is not None
            else row.get("epa")
        )
        dropbacks = self._safe_float(row.get("dropbacks"))
        if dropbacks is None:
            dropbacks = (attempts or 0.0) + (sacks or 0.0)
        epa_per_play = (
            passing_epa / dropbacks
            if passing_epa is not None and dropbacks and dropbacks > 0
            else None
        )

        return {
            "qb_id": (
                str(
                    row.get("player_id")
                    or row.get("gsis_id")
                    or row.get("player_gsis_id")
                )
                if row.get("player_id") is not None
                or row.get("gsis_id") is not None
                or row.get("player_gsis_id") is not None
                else None
            ),
            "qb_name": (
                str(row.get("player_display_name") or row.get("player_name"))
                if row.get("player_display_name") is not None
                or row.get("player_name") is not None
                else None
            ),
            "attempts": attempts,
            "completions": self._safe_float(row.get("completions")),
            "passing_yards": self._safe_float(row.get("passing_yards")),
            "passing_tds": self._safe_float(row.get("passing_tds")),
            "interceptions": self._safe_float(
                row.get("interceptions")
                if row.get("interceptions") is not None
                else row.get("interceptions_thrown")
            ),
            "sacks": sacks,
            "passing_epa": passing_epa,
            "epa_per_play": epa_per_play,
        }

    @staticmethod
    def _normalize_team_code(value: Any) -> Optional[str]:
        """Normalize team abbreviations to uppercase strings."""
        if value is None:
            return None
        team = str(value).strip().upper()
        return team or None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Return int when possible."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Return float when possible."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def import_weekly_player_data(years: Iterable[int]) -> pd.DataFrame:
    """Load weekly player stats from the current nflverse release."""
    requested_years = sorted({int(year) for year in years})
    if not requested_years:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for year in requested_years:
        try:
            frame = pd.read_parquet(
                NFLVERSE_WEEKLY_STATS_URL.format(year=year),
                engine="auto",
            )
        except Exception as exc:
            logger.warning(
                "Skipping weekly player data for %s: %s", year, exc
            )
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    float_cols = data.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        data[float_cols] = data[float_cols].astype("float32")
    return data
