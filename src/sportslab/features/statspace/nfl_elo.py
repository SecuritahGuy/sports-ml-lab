"""Canonical NFL team + QB Elo with preseason carryover and artifacts."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from .nfl_schedule_loader import NFLHistoricalGame

logger = logging.getLogger(__name__)

NFL_TEAM_ELO_RATINGS_FILENAME = "nfl_team_elo_ratings.json"
NFL_QB_ELO_RATINGS_FILENAME = "nfl_qb_elo_ratings.json"
NFL_ELO_METADATA_FILENAME = "nfl_elo_ratings_metadata.json"


@dataclass
class NFLEloConfig:
    """Configuration for canonical NFL team + QB Elo."""

    initial_elo: float = 1505.0
    k_factor: float = 20.0
    home_field_advantage: float = 65.0
    season_carryover: float = 0.67
    qb_initial_elo: float = 1500.0
    qb_k_factor: float = 12.0
    qb_adjustment_scale: float = 0.35
    regular_season_game_type: str = "REG"
    use_mov: bool = True
    efficiency_denominator: float = 0.20
    manual_qb_overrides: dict[str, float] = field(default_factory=dict)
    playoff_home_field_advantage: Optional[float] = None
    playoff_k_factor: Optional[float] = None
    ensemble_weights: Optional[dict[str, float]] = None


def build_nfl_elo_metadata(
    config: NFLEloConfig,
    *,
    snapshot_season: Optional[int],
    source_start_season: Optional[int],
    source_end_season: Optional[int],
    snapshot_type: str = "preseason",
) -> dict[str, Any]:
    """Build a machine-readable NFL Elo artifact metadata payload."""
    return {
        "engine": "canonical_nfl_team_qb_elo",
        "version": 1,
        "snapshot_type": snapshot_type,
        "snapshot_season": snapshot_season,
        "source_start_season": source_start_season,
        "source_end_season": source_end_season,
        "initial_elo": config.initial_elo,
        "k_factor": config.k_factor,
        "home_field_advantage": config.home_field_advantage,
        "season_carryover": config.season_carryover,
        "qb_initial_elo": config.qb_initial_elo,
        "qb_k_factor": config.qb_k_factor,
        "qb_adjustment_scale": config.qb_adjustment_scale,
        "regular_season_game_type": config.regular_season_game_type,
        "built_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def write_nfl_elo_artifact(
    output_dir: Path,
    team_ratings: dict[str, float],
    qb_ratings: dict[str, float],
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Persist NFL team ratings, QB ratings, and metadata sidecars together."""
    output_dir.mkdir(parents=True, exist_ok=True)
    team_path = output_dir / NFL_TEAM_ELO_RATINGS_FILENAME
    qb_path = output_dir / NFL_QB_ELO_RATINGS_FILENAME
    metadata_path = output_dir / NFL_ELO_METADATA_FILENAME
    with team_path.open("w", encoding="utf-8") as handle:
        json.dump(team_ratings, handle, indent=2)
        handle.write("\n")
    with qb_path.open("w", encoding="utf-8") as handle:
        json.dump(qb_ratings, handle, indent=2)
        handle.write("\n")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return team_path, qb_path, metadata_path


def load_nfl_elo_artifact(
    candidate_dirs: Iterable[Path],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any], Optional[Path]]:
    """Load the first available NFL Elo artifact directory."""
    for directory in candidate_dirs:
        team_path = directory / NFL_TEAM_ELO_RATINGS_FILENAME
        qb_path = directory / NFL_QB_ELO_RATINGS_FILENAME
        if not team_path.exists() or not qb_path.exists():
            continue
        try:
            with team_path.open("r", encoding="utf-8") as handle:
                team_ratings = json.load(handle)
            with qb_path.open("r", encoding="utf-8") as handle:
                qb_ratings = json.load(handle)
            team_payload = {
                str(team): float(rating)
                for team, rating in team_ratings.items()
            }
            qb_payload = {
                str(qb): float(rating) for qb, rating in qb_ratings.items()
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load NFL Elo artifact from %s: %s", directory, exc
            )
            continue

        metadata_path = directory / NFL_ELO_METADATA_FILENAME
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                logger.warning(
                    "Failed to load NFL Elo metadata from %s: %s",
                    metadata_path,
                    exc,
                )
        return team_payload, qb_payload, metadata, directory
    return {}, {}, {}, None


class NFLEloEngine:
    """Compute sequential pregame NFL team + QB Elo."""

    def __init__(
        self,
        config: NFLEloConfig | None = None,
        initial_team_ratings: Optional[dict[str, float]] = None,
        initial_qb_ratings: Optional[dict[str, float]] = None,
        initial_season: Optional[int] = None,
    ) -> None:
        self.config = config or NFLEloConfig()
        self._seed_team_ratings = {
            str(team): float(rating)
            for team, rating in (initial_team_ratings or {}).items()
        }
        self._seed_qb_ratings = {
            str(qb): float(rating)
            for qb, rating in (initial_qb_ratings or {}).items()
        }
        self._seed_season = initial_season
        self.team_ratings: dict[str, float] = {}
        self.qb_ratings: dict[str, float] = {}
        self.current_season: Optional[int] = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.team_ratings = dict(self._seed_team_ratings)
        self.qb_ratings = dict(self._seed_qb_ratings)
        self.current_season = self._seed_season

    def load_snapshot(
        self,
        *,
        team_ratings: dict[str, float],
        qb_ratings: dict[str, float],
        snapshot_season: Optional[int],
    ) -> None:
        """Replace the engine seed state with a persisted snapshot."""
        self._seed_team_ratings = {
            str(team): float(rating) for team, rating in team_ratings.items()
        }
        self._seed_qb_ratings = {
            str(qb): float(rating) for qb, rating in qb_ratings.items()
        }
        self._seed_season = snapshot_season
        self._reset_state()

    def get_team_ratings(self) -> dict[str, float]:
        """Return a copy of current team ratings."""
        return dict(self.team_ratings)

    def get_qb_ratings(self) -> dict[str, float]:
        """Return a copy of current QB ratings."""
        return dict(self.qb_ratings)

    def expected_home_win_prob(
        self,
        home_team_elo: float,
        away_team_elo: float,
        *,
        home_qb_elo: Optional[float] = None,
        away_qb_elo: Optional[float] = None,
        home_field_advantage: Optional[float] = None,
    ) -> float:
        """Return home win probability using team Elo, HFA, and QB Elo."""
        effective_gap = self.effective_elo_gap(
            home_team_elo=home_team_elo,
            away_team_elo=away_team_elo,
            home_qb_elo=home_qb_elo,
            away_qb_elo=away_qb_elo,
            home_field_advantage=home_field_advantage,
        )
        prob = 1.0 / (1.0 + math.pow(10.0, -effective_gap / 400.0))
        return min(max(prob, 0.0), 1.0)

    def qb_adjustment(
        self, qb_elo: Optional[float], qb_id: Optional[str] = None
    ) -> float:
        """Convert a QB Elo rating into a bounded pregame team adjustment."""
        if qb_id and qb_id in self.config.manual_qb_overrides:
            return float(self.config.manual_qb_overrides[qb_id])
        resolved_qb_elo = (
            self.config.qb_initial_elo if qb_elo is None else float(qb_elo)
        )
        return (
            resolved_qb_elo - self.config.qb_initial_elo
        ) * self.config.qb_adjustment_scale

    def effective_elo_gap(
        self,
        *,
        home_team_elo: float,
        away_team_elo: float,
        home_qb_elo: Optional[float] = None,
        away_qb_elo: Optional[float] = None,
        home_qb_id: Optional[str] = None,
        away_qb_id: Optional[str] = None,
        home_field_advantage: Optional[float] = None,
    ) -> float:
        """Return the effective home-minus-away Elo gap used for prediction."""
        hfa = (
            self.config.home_field_advantage
            if home_field_advantage is None
            else home_field_advantage
        )
        home_qb_adj = self.qb_adjustment(home_qb_elo, home_qb_id)
        away_qb_adj = self.qb_adjustment(away_qb_elo, away_qb_id)
        return float(
            home_team_elo + hfa + home_qb_adj - away_qb_adj - away_team_elo
        )

    def build_game_features(
        self,
        games: Iterable[NFLHistoricalGame],
        *,
        reset: bool = True,
    ) -> pd.DataFrame:
        """Build sequential pregame Elo rows from historical NFL games."""
        if reset:
            self._reset_state()

        rows: list[dict[str, Any]] = []
        ordered_games = sorted(
            self._regular_season_games(games),
            key=lambda item: (item.date, item.week, item.game_id),
        )

        for game in ordered_games:
            self._maybe_revert_for_new_season(game.season)

            home_team_elo = self.team_ratings.get(
                game.home_team, self.config.initial_elo
            )
            away_team_elo = self.team_ratings.get(
                game.away_team, self.config.initial_elo
            )
            home_qb_elo = self._pregame_qb_elo(game.home_qb_id)
            away_qb_elo = self._pregame_qb_elo(game.away_qb_id)
            home_qb_adj = self.qb_adjustment(home_qb_elo, game.home_qb_id)
            away_qb_adj = self.qb_adjustment(away_qb_elo, game.away_qb_id)
            effective_gap = self.effective_elo_gap(
                home_team_elo=home_team_elo,
                away_team_elo=away_team_elo,
                home_qb_elo=home_qb_elo,
                away_qb_elo=away_qb_elo,
                home_qb_id=game.home_qb_id,
                away_qb_id=game.away_qb_id,
            )
            home_prob = self.expected_home_win_prob(
                home_team_elo,
                away_team_elo,
                home_qb_elo=home_qb_elo,
                away_qb_elo=away_qb_elo,
                home_field_advantage=self.config.home_field_advantage,
            )

            rows.append(
                {
                    "game_id": game.game_id,
                    "date": game.date,
                    "season": game.season,
                    "week": game.week,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_elo_pre": float(home_team_elo),
                    "away_elo_pre": float(away_team_elo),
                    "home_qb_elo_pre": (
                        float(home_qb_elo)
                        if home_qb_elo is not None
                        else float(self.config.qb_initial_elo)
                    ),
                    "away_qb_elo_pre": (
                        float(away_qb_elo)
                        if away_qb_elo is not None
                        else float(self.config.qb_initial_elo)
                    ),
                    "home_qb_adjustment": float(home_qb_adj),
                    "away_qb_adjustment": float(away_qb_adj),
                    "effective_elo_diff": float(effective_gap),
                    "elo_diff": float(home_team_elo - away_team_elo),
                    "elo_home_win_prob": float(home_prob),
                    "home_qb_id": game.home_qb_id,
                    "away_qb_id": game.away_qb_id,
                    "home_qb_source": game.home_qb_source,
                    "away_qb_source": game.away_qb_source,
                }
            )

            self._update_ratings(
                game=game,
                home_team_elo=home_team_elo,
                away_team_elo=away_team_elo,
                expected_home=home_prob,
                home_qb_elo=home_qb_elo,
                away_qb_elo=away_qb_elo,
                effective_gap=effective_gap,
            )

        return pd.DataFrame(rows)

    def process_games(
        self,
        games: Iterable[NFLHistoricalGame],
        *,
        reset: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Process games and return final team and QB rating snapshots."""
        self.build_game_features(games, reset=reset)
        return {
            "team_ratings": self.get_team_ratings(),
            "qb_ratings": self.get_qb_ratings(),
        }

    def preseason_ratings_for_season(
        self,
        games: Iterable[NFLHistoricalGame],
        season: int,
    ) -> dict[str, dict[str, float]]:
        """Return the preseason team/QB snapshot for a target season."""
        self._reset_state()
        ordered_games = sorted(
            self._regular_season_games(games),
            key=lambda item: (item.date, item.week, item.game_id),
        )
        for game in ordered_games:
            if game.season >= season:
                break
            self._maybe_revert_for_new_season(game.season)
            home_team_elo = self.team_ratings.get(
                game.home_team, self.config.initial_elo
            )
            away_team_elo = self.team_ratings.get(
                game.away_team, self.config.initial_elo
            )
            home_qb_elo = self._pregame_qb_elo(game.home_qb_id)
            away_qb_elo = self._pregame_qb_elo(game.away_qb_id)
            expected_home = self.expected_home_win_prob(
                home_team_elo,
                away_team_elo,
                home_qb_elo=home_qb_elo,
                away_qb_elo=away_qb_elo,
            )
            effective_gap = self.effective_elo_gap(
                home_team_elo=home_team_elo,
                away_team_elo=away_team_elo,
                home_qb_elo=home_qb_elo,
                away_qb_elo=away_qb_elo,
                home_qb_id=game.home_qb_id,
                away_qb_id=game.away_qb_id,
            )
            self._update_ratings(
                game=game,
                home_team_elo=home_team_elo,
                away_team_elo=away_team_elo,
                expected_home=expected_home,
                home_qb_elo=home_qb_elo,
                away_qb_elo=away_qb_elo,
                effective_gap=effective_gap,
            )

        self._maybe_revert_for_new_season(season)
        return {
            "team_ratings": self.get_team_ratings(),
            "qb_ratings": self.get_qb_ratings(),
        }

    def _pregame_qb_elo(self, qb_id: Optional[str]) -> Optional[float]:
        """Return the pregame QB Elo if the starter is known."""
        if not qb_id:
            return None
        return self.qb_ratings.get(str(qb_id), self.config.qb_initial_elo)

    def _maybe_revert_for_new_season(self, season: int) -> None:
        if self.current_season is None:
            self.current_season = season
            return
        if season == self.current_season:
            return
        while self.current_season < season:
            self.current_season += 1
            for team, rating in list(self.team_ratings.items()):
                self.team_ratings[team] = (
                    rating * self.config.season_carryover
                ) + (
                    self.config.initial_elo
                    * (1.0 - self.config.season_carryover)
                )
            for qb, rating in list(self.qb_ratings.items()):
                self.qb_ratings[qb] = (
                    rating * self.config.season_carryover
                ) + (
                    self.config.qb_initial_elo
                    * (1.0 - self.config.season_carryover)
                )

    def _regular_season_games(
        self,
        games: Iterable[NFLHistoricalGame],
    ) -> list[NFLHistoricalGame]:
        regular_type = self.config.regular_season_game_type.upper()
        return [
            game
            for game in games
            if str(getattr(game, "game_type", regular_type)).upper()
            == regular_type
        ]

    def _mov_multiplier(
        self,
        point_diff: int,
        winner_elo: float,
        loser_elo: float,
    ) -> float:
        """FiveThirtyEight NFL MOV multiplier with autocorrelation."""
        return math.log(abs(point_diff) + 1.0) * (
            2.2 / (((winner_elo - loser_elo) * 0.001) + 2.2)
        )

    def _team_update_k(self, game: NFLHistoricalGame) -> float:
        """Return effective K-factor, leaving room for playoff overrides."""
        if (
            str(game.game_type).upper()
            != self.config.regular_season_game_type.upper()
        ):
            return float(self.config.playoff_k_factor or self.config.k_factor)
        return self.config.k_factor

    def _update_ratings(
        self,
        *,
        game: NFLHistoricalGame,
        home_team_elo: float,
        away_team_elo: float,
        expected_home: float,
        home_qb_elo: Optional[float],
        away_qb_elo: Optional[float],
        effective_gap: float,
    ) -> None:
        actual_home = game.home_win
        multiplier = 1.0
        if self.config.use_mov and actual_home != 0.5:
            winner_effective = (
                effective_gap if actual_home == 1.0 else -effective_gap
            )
            winner_elo = self.config.initial_elo + winner_effective
            loser_elo = self.config.initial_elo
            multiplier = self._mov_multiplier(
                abs(int(game.home_score) - int(game.away_score)),
                winner_elo=winner_elo,
                loser_elo=loser_elo,
            )

        team_change = (
            self._team_update_k(game)
            * multiplier
            * (actual_home - expected_home)
        )
        self.team_ratings[game.home_team] = home_team_elo + team_change
        self.team_ratings[game.away_team] = away_team_elo - team_change

        self._update_qb_ratings(
            game=game,
            expected_home=expected_home,
            actual_home=actual_home,
            multiplier=multiplier,
            home_qb_elo=home_qb_elo,
            away_qb_elo=away_qb_elo,
        )

    def _update_qb_ratings(
        self,
        *,
        game: NFLHistoricalGame,
        expected_home: float,
        actual_home: float,
        multiplier: float,
        home_qb_elo: Optional[float],
        away_qb_elo: Optional[float],
    ) -> None:
        home_actual, away_actual = self._qb_actual_scores(game, actual_home)
        if game.home_qb_id:
            pre_home = (
                self.config.qb_initial_elo
                if home_qb_elo is None
                else home_qb_elo
            )
            self.qb_ratings[str(game.home_qb_id)] = pre_home + (
                self.config.qb_k_factor
                * multiplier
                * (home_actual - expected_home)
            )
        if game.away_qb_id:
            pre_away = (
                self.config.qb_initial_elo
                if away_qb_elo is None
                else away_qb_elo
            )
            expected_away = 1.0 - expected_home
            self.qb_ratings[str(game.away_qb_id)] = pre_away + (
                self.config.qb_k_factor
                * multiplier
                * (away_actual - expected_away)
            )

    def _qb_actual_scores(
        self,
        game: NFLHistoricalGame,
        actual_home: float,
    ) -> tuple[float, float]:
        """Return actual QB scores with optional EPA/play residual."""
        home_epa = self._safe_float(game.home_qb_epa_per_play)
        away_epa = self._safe_float(game.away_qb_epa_per_play)
        if home_epa is None or away_epa is None:
            return actual_home, 1.0 - actual_home

        delta = (home_epa - away_epa) / self.config.efficiency_denominator
        efficiency_home = self._clamp(0.5 + 0.5 * math.tanh(delta), 0.0, 1.0)
        efficiency_away = 1.0 - efficiency_home
        home_actual = (0.7 * actual_home) + (0.3 * efficiency_home)
        away_actual = (0.7 * (1.0 - actual_home)) + (0.3 * efficiency_away)
        return home_actual, away_actual

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Return float when possible."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        """Clamp a value to a closed interval."""
        return min(max(value, lower), upper)
