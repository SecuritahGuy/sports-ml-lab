"""QB strength adjustment features — shrunken Elo-point QB ratings.

Computes per-game QB rating adjustments in Elo points based on each QB's
historical performance relative to team Elo expectation. Uses Bayesian
shrinkage so small-sample QBs regress toward a replacement-level baseline.

Usage:
    df = compute_qb_adjustments(df)
    # Adds columns: home_qb_adj, away_qb_adj, home_qb_starts, away_qb_starts

Reference:
    team_effective_elo = team_elo + qb_adjustment
    effective_elo_prob = 1/(1 + 10^(-(h_elo + h_adj - a_elo - a_adj + HFA)/400))
"""

import numpy as np
import pandas as pd

from sportslab.features.build_features import SPORTSLAB_MIN_SEASON

QB_ADJUSTMENT_COLUMNS = [
    "home_qb_adj",
    "away_qb_adj",
    "home_qb_starts",
    "away_qb_starts",
]

# — Hyperparameters (tunable but sensible defaults) —

# Number of replacement-level starts to add as prior (Bayesian shrinkage).
# ~1 season = 17 games centers small-sample QBs toward replacement.
PRIOR_STARTS = 17

# Replacement-level impact: how many wins above/below .500 a replacement QB
# produces relative to Elo expectation.  -0.03 means ~0.5 game worse per 17.
PRIOR_IMPACT = -0.03

# Maximum |qb_adj| in Elo points to prevent extreme ratings for tiny samples.
MAX_ADJUSTMENT = 120.0


def _expected_win_prob(
    team_elo: float,
    opp_elo: float,
    hfa: float = 0.0,
) -> float:
    """Elo expected score (home win prob from team's perspective)."""
    diff = (team_elo - opp_elo + hfa) / 400.0
    return 1.0 / (1.0 + 10.0 ** (-diff))


def _impact_to_elo_pts(impact: float) -> float:
    """Convert a win-prob impact (delta above/below .500) to Elo points.

    P = 0.5 + impact  →  Elo diff = 400 * log10(P / (1-P))
    Clamped to prevent extreme values.
    """
    p = np.clip(0.5 + impact, 0.01, 0.99)
    return float(400.0 * np.log10(p / (1.0 - p)))


def compute_qb_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game QB strength adjustments with Bayesian shrinkage.

    Requires columns produced by compute_elo_features():
        season, week, gameday, home_team, away_team,
        home_elo_pre, away_elo_pre, elo_prob,
        home_qb_id, away_qb_id, home_win

    Adds columns:
        home_qb_adj, away_qb_adj — Elo-point adjustment (positive = stronger)
        home_qb_starts, away_qb_starts — number of recorded starts for that QB

    Shrinkage:
        shrunken_impact = (observed_impact * n + PRIOR_IMPACT * PRIOR_STARTS)
                          / (n + PRIOR_STARTS)
        where observed_impact = mean(actual - expected) across starts.

    Leakage:
        Only games chronologically before the current game are used.
        No post-game data, final score, or market info leaks into the rating.
    """
    if df.empty:
        return df

    bad = df[df["season"] < SPORTSLAB_MIN_SEASON]
    if not bad.empty:
        raise ValueError(
            f"Found {len(bad)} rows with season < {SPORTSLAB_MIN_SEASON}. "
            f"This project only supports {SPORTSLAB_MIN_SEASON}+."
        )

    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    # Per-QB accumulator: starts, sum(expected), sum(actual)
    qb_stats: dict[str, dict] = {}

    adjustments: dict[str, list] = {
        "home_qb_adj": [],
        "away_qb_adj": [],
        "home_qb_starts": [],
        "away_qb_starts": [],
    }

    for _, row in out.iterrows():
        for side, team_col, qb_col, elo_col, opp_elo_col, is_home in [
            ("home", "home_team", "home_qb_id", "home_elo_pre", "away_elo_pre", True),
            ("away", "away_team", "away_qb_id", "away_elo_pre", "home_elo_pre", False),
        ]:
            qb_id = row.get(qb_col)
            qb_missing = pd.isna(qb_id) or str(qb_id).strip() in ("", "nan", "None")

            if qb_missing:
                adjustments[f"{side}_qb_adj"].append(0.0)
                adjustments[f"{side}_qb_starts"].append(0)
                continue

            stats = qb_stats.get(qb_id, {"starts": 0, "sum_expected": 0.0, "sum_actual": 0.0})
            n = stats["starts"]
            adjustments[f"{side}_qb_starts"].append(n)

            # Compute shrunken impact
            if n > 0:
                observed_impact = (stats["sum_actual"] - stats["sum_expected"]) / n
            else:
                observed_impact = 0.0

            shrunken_impact = (
                observed_impact * n + PRIOR_IMPACT * PRIOR_STARTS
            ) / (n + PRIOR_STARTS)

            elo_pts = _impact_to_elo_pts(shrunken_impact)
            elo_pts = np.clip(elo_pts, -MAX_ADJUSTMENT, MAX_ADJUSTMENT)
            adjustments[f"{side}_qb_adj"].append(round(float(elo_pts), 1))

        # --- Post-game: update QB stats ---
        for team_col, qb_col, is_home in [
            ("home_team", "home_qb_id", True),
            ("away_team", "away_qb_id", False),
        ]:
            qb_id = row.get(qb_col)
            qb_missing = pd.isna(qb_id) or str(qb_id).strip() in ("", "nan", "None")
            if qb_missing:
                continue

            team_elo = row.get("home_elo_pre" if is_home else "away_elo_pre", 1500.0)
            opp_elo = row.get("away_elo_pre" if is_home else "home_elo_pre", 1500.0)
            hfa = 40.0 if is_home else 0.0

            expected = _expected_win_prob(team_elo, opp_elo, hfa)

            home_won = row.get("home_win")
            if pd.isna(home_won):
                actual = 0.5
            else:
                actual = float(home_won == 1) if is_home else float(home_won == 0)

            if qb_id not in qb_stats:
                qb_stats[qb_id] = {"starts": 0, "sum_expected": 0.0, "sum_actual": 0.0}
            qb_stats[qb_id]["starts"] += 1
            qb_stats[qb_id]["sum_expected"] += expected
            qb_stats[qb_id]["sum_actual"] += actual

    for col, vals in adjustments.items():
        out[col] = vals

    return out


def compute_qb_adjusted_elo_prob(
    home_elo: np.ndarray,
    away_elo: np.ndarray,
    home_adj: np.ndarray,
    away_adj: np.ndarray,
    hfa: float = 40.0,
) -> np.ndarray:
    """Compute home win probability with QB-adjusted Elo ratings.

    team_effective_elo = team_elo + qb_adjustment
    Then standard Elo formula:
        P(home) = 1/(1 + 10^(-(home_effective - away_effective + HFA)/400))
    """
    home_effective = home_elo + home_adj
    away_effective = away_elo + away_adj
    diff = (home_effective - away_effective + hfa) / 400.0
    return 1.0 / (1.0 + 10.0 ** (-diff))


# ── Gated QB adjustment variants ──

DEFAULT_STABLE_SHRINK = 0.3
DEFAULT_MIN_STARTS_FOR_STABLE = 4
DEFAULT_MIN_GAMES_CONTINUITY = 4

GATE_MODES = {
    "full": "No gating.  Standard V0 adjustment applied to all games.",
    "qb_changed_only": (
        "Adjustment applied only when that side has a QB change. "
        "Stable-QB games get adjustment = 0."
    ),
    "low_continuity": (
        "Adjustment applied when the QB has changed OR has fewer than "
        "min_starts_for_stable starts with the team.  Established QBs get 0."
    ),
    "shrunk_stable": (
        "Full adjustment for changed-QB sides.  Stable-QB adjustments "
        "are multiplied by stable_shrink (default 0.3)."
    ),
    "capped_only": (
        "Standard full adjustment but with lower max absolute "
        "adjustment (max_adj_cap)."
    ),
    "aggressive_diagnostic": (
        "DIAGNOSTIC ONLY: 2x multiplier for changed-QB sides, "
        "0 adjustment for stable-QB sides."
    ),
    "recency_decay": (
        "Standard full adjustment but older games decayed "
        "(recency_weighted).  Uses compute_recency_weighted_qb_adjustments."
    ),
}


def apply_qb_adjustment_gate(
    home_qb_adj: np.ndarray,
    away_qb_adj: np.ndarray,
    home_qb_changed: np.ndarray,
    away_qb_changed: np.ndarray,
    home_qb_team_starts_pre: np.ndarray | None = None,
    away_qb_team_starts_pre: np.ndarray | None = None,
    home_games_since_qb_change: np.ndarray | None = None,
    away_games_since_qb_change: np.ndarray | None = None,
    gate_mode: str = "full",
    stable_shrink: float = DEFAULT_STABLE_SHRINK,
    min_starts_for_stable: int = DEFAULT_MIN_STARTS_FOR_STABLE,
    max_adj_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a pregame gate to QB adjustments.

    All gating information (qb_changed, starts, continuity) is pregame-safe:
    it depends only on data from games played before the current game.

    Args:
        home_qb_adj: Full QB adjustments for home QB.
        away_qb_adj: Full QB adjustments for away QB.
        home_qb_changed: 1 if home QB changed from prior game.
        away_qb_changed: 1 if away QB changed from prior game.
        home_qb_team_starts_pre: Prior starts for home QB with team.
        away_qb_team_starts_pre: Prior starts for away QB with team.
        home_games_since_qb_change: Games since last QB change for home.
        away_games_since_qb_change: Games since last QB change for away.
        gate_mode: One of GATE_MODES keys.
        stable_shrink: Multiplier (0-1) for stable-QB adj in shrunk_stable mode.
        min_starts_for_stable: Minimum starts before QB is considered stable.
        max_adj_cap: Override max absolute adjustment.  None = use default.

    Returns:
        (gated_home_adj, gated_away_adj)
    """
    h_adj = home_qb_adj.copy().astype(float)
    a_adj = away_qb_adj.copy().astype(float)
    n = len(h_adj)

    if gate_mode == "full":
        pass

    elif gate_mode == "qb_changed_only":
        h_changed = np.array(home_qb_changed, dtype=float).ravel()
        a_changed = np.array(away_qb_changed, dtype=float).ravel()
        h_adj = h_adj * h_changed
        a_adj = a_adj * a_changed

    elif gate_mode == "low_continuity":
        h_changed = np.array(home_qb_changed, dtype=float).ravel()
        a_changed = np.array(away_qb_changed, dtype=float).ravel()
        h_starts = (
            np.array(home_qb_team_starts_pre, dtype=float).ravel()
            if home_qb_team_starts_pre is not None
            else np.zeros(n)
        )
        a_starts = (
            np.array(away_qb_team_starts_pre, dtype=float).ravel()
            if away_qb_team_starts_pre is not None
            else np.zeros(n)
        )
        h_mask = (h_changed == 1) | (h_starts < min_starts_for_stable)
        a_mask = (a_changed == 1) | (a_starts < min_starts_for_stable)
        h_adj = h_adj * h_mask.astype(float)
        a_adj = a_adj * a_mask.astype(float)

    elif gate_mode == "shrunk_stable":
        h_changed = np.array(home_qb_changed, dtype=float).ravel()
        a_changed = np.array(away_qb_changed, dtype=float).ravel()
        h_gate = np.where(h_changed == 1, 1.0, stable_shrink)
        a_gate = np.where(a_changed == 1, 1.0, stable_shrink)
        h_adj = h_adj * h_gate
        a_adj = a_adj * a_gate

    elif gate_mode == "capped_only":
        if max_adj_cap is not None:
            h_adj = np.clip(h_adj, -max_adj_cap, max_adj_cap)
            a_adj = np.clip(a_adj, -max_adj_cap, max_adj_cap)

    elif gate_mode == "aggressive_diagnostic":
        h_changed = np.array(home_qb_changed, dtype=float).ravel()
        a_changed = np.array(away_qb_changed, dtype=float).ravel()
        h_adj = h_adj * np.where(h_changed == 1, 2.0, 0.0)
        a_adj = a_adj * np.where(a_changed == 1, 2.0, 0.0)

    h_adj = np.nan_to_num(h_adj, nan=0.0)
    a_adj = np.nan_to_num(a_adj, nan=0.0)

    return h_adj, a_adj


def compute_gated_qb_adjustments(
    df: pd.DataFrame,
    gate_mode: str = "full",
    stable_shrink: float = DEFAULT_STABLE_SHRINK,
    min_starts_for_stable: int = DEFAULT_MIN_STARTS_FOR_STABLE,
    max_adj_cap: float | None = None,
) -> pd.DataFrame:
    """Compute QB adjustments and apply a pregame gate.

    Runs the standard compute_qb_adjustments, then applies the
    specified gate using pregame QB state columns.

    Args:
        df: DataFrame with required QB columns.
        gate_mode: Gating strategy (see GATE_MODES).
        stable_shrink: Multiplier for stable-QB adjustments.
        min_starts_for_stable: Min starts before QB is stable.
        max_adj_cap: Override max absolute adjustment.

    Returns:
        DataFrame with home_qb_adj, away_qb_adj columns gated.
    """
    if df.empty:
        return df.copy()

    out = compute_qb_adjustments(df)

    # Build gating columns if not present
    if "home_qb_changed" not in out.columns:
        from sportslab.features.qb import compute_qb_features
        out = compute_qb_features(out)

    h_adj = out["home_qb_adj"].values
    a_adj = out["away_qb_adj"].values
    h_changed = out["home_qb_changed"].values
    a_changed = out["away_qb_changed"].values
    h_starts = out.get("home_qb_team_starts_pre", None)
    a_starts = out.get("away_qb_team_starts_pre", None)
    h_continuity = out.get("home_games_since_qb_change", None)
    a_continuity = out.get("away_games_since_qb_change", None)

    if gate_mode == "recency_decay":
        out_gated = compute_recency_weighted_qb_adjustments(
            df, max_adj_cap=max_adj_cap
        )
        out["home_qb_adj"] = out_gated["home_qb_adj"]
        out["away_qb_adj"] = out_gated["away_qb_adj"]
        return out

    g_h, g_a = apply_qb_adjustment_gate(
        h_adj, a_adj,
        h_changed, a_changed,
        home_qb_team_starts_pre=h_starts.values if h_starts is not None else None,
        away_qb_team_starts_pre=a_starts.values if a_starts is not None else None,
        home_games_since_qb_change=h_continuity.values if h_continuity is not None else None,
        away_games_since_qb_change=a_continuity.values if a_continuity is not None else None,
        gate_mode=gate_mode,
        stable_shrink=stable_shrink,
        min_starts_for_stable=min_starts_for_stable,
        max_adj_cap=max_adj_cap,
    )

    out["home_qb_adj"] = g_h
    out["away_qb_adj"] = g_a
    return out


def compute_recency_weighted_qb_adjustments(
    df: pd.DataFrame,
    decay_half_life: float = 32.0,
    max_adj_cap: float | None = None,
) -> pd.DataFrame:
    """Compute QB adjustments with exponential decay of older games.

    Older QB starts are downweighted so recent performance matters
    more than distant history.  Uses the same Bayesian shrinkage
    framework but with a recency-weight on the prior.

    Args:
        df: DataFrame with required QB/Elio columns.
        decay_half_life: Number of games to halve the weight of old games.
        max_adj_cap: Override max absolute adjustment.  None = default.

    Returns:
        DataFrame with home_qb_adj, away_qb_adj columns.
    """
    if df.empty:
        return df

    bad = df[df["season"] < SPORTSLAB_MIN_SEASON]
    if not bad.empty:
        raise ValueError(
            f"Found {len(bad)} rows with season < {SPORTSLAB_MIN_SEASON}. "
            f"This project only supports {SPORTSLAB_MIN_SEASON}+."
        )

    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    qb_stats: dict[str, dict] = {}
    cap = max_adj_cap if max_adj_cap is not None else MAX_ADJUSTMENT

    adjustments: dict[str, list] = {
        "home_qb_adj": [],
        "away_qb_adj": [],
        "home_qb_starts": [],
        "away_qb_starts": [],
    }

    for _, row in out.iterrows():
        for side, team_col, qb_col, elo_col, opp_elo_col, is_home in [
            ("home", "home_team", "home_qb_id", "home_elo_pre", "away_elo_pre", True),
            ("away", "away_team", "away_qb_id", "away_elo_pre", "home_elo_pre", False),
        ]:
            qb_id = row.get(qb_col)
            qb_missing = pd.isna(qb_id) or str(qb_id).strip() in ("", "nan", "None")

            if qb_missing:
                adjustments[f"{side}_qb_adj"].append(0.0)
                adjustments[f"{side}_qb_starts"].append(0)
                continue

            stats = qb_stats.get(qb_id, {
                "starts": 0, "sum_expected": 0.0, "sum_actual": 0.0,
                "weight_sum": 0.0, "weighted_sum_expected": 0.0,
                "weighted_sum_actual": 0.0,
            })
            n = stats["starts"]
            effective_n = stats["weight_sum"]
            adjustments[f"{side}_qb_starts"].append(n)

            if effective_n > 0:
                weighted_impact = (
                    stats["weighted_sum_actual"] - stats["weighted_sum_expected"]
                ) / effective_n
            else:
                weighted_impact = 0.0

            # Use effective_n (weight sum) for shrinkage
            shrunken_impact = (
                weighted_impact * effective_n + PRIOR_IMPACT * PRIOR_STARTS
            ) / (effective_n + PRIOR_STARTS)

            elo_pts = _impact_to_elo_pts(shrunken_impact)
            elo_pts = np.clip(elo_pts, -cap, cap)
            adjustments[f"{side}_qb_adj"].append(round(float(elo_pts), 1))

        # Post-game: update QB stats with decay
        for team_col, qb_col, is_home in [
            ("home_team", "home_qb_id", True),
            ("away_team", "away_qb_id", False),
        ]:
            qb_id = row.get(qb_col)
            qb_missing = pd.isna(qb_id) or str(qb_id).strip() in ("", "nan", "None")
            if qb_missing:
                continue

            team_elo = row.get("home_elo_pre" if is_home else "away_elo_pre", 1500.0)
            opp_elo = row.get("away_elo_pre" if is_home else "home_elo_pre", 1500.0)
            hfa = 40.0 if is_home else 0.0

            expected = _expected_win_prob(team_elo, opp_elo, hfa)

            home_won = row.get("home_win")
            if pd.isna(home_won):
                actual = 0.5
            else:
                actual = float(home_won == 1) if is_home else float(home_won == 0)

            if qb_id not in qb_stats:
                qb_stats[qb_id] = {
                    "starts": 0, "sum_expected": 0.0, "sum_actual": 0.0,
                    "weight_sum": 0.0, "weighted_sum_expected": 0.0,
                    "weighted_sum_actual": 0.0,
                }

            # Apply decay to existing stats: weight all old data by decay factor
            if decay_half_life > 0 and qb_stats[qb_id]["starts"] > 0:
                decay_factor = 2.0 ** (-1.0 / decay_half_life)
                qb_stats[qb_id]["sum_expected"] *= decay_factor
                qb_stats[qb_id]["sum_actual"] *= decay_factor
                qb_stats[qb_id]["weighted_sum_expected"] *= decay_factor
                qb_stats[qb_id]["weighted_sum_actual"] *= decay_factor
                qb_stats[qb_id]["weight_sum"] *= decay_factor

            qb_stats[qb_id]["starts"] += 1
            qb_stats[qb_id]["sum_expected"] += expected
            qb_stats[qb_id]["sum_actual"] += actual
            qb_stats[qb_id]["weight_sum"] += 1.0
            qb_stats[qb_id]["weighted_sum_expected"] += expected
            qb_stats[qb_id]["weighted_sum_actual"] += actual

    for col, vals in adjustments.items():
        out[col] = vals

    return out
