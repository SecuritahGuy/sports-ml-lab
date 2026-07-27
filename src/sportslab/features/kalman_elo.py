"""Kalman-filter Elo rating: uncertainty-aware team strength estimation.

Each team tracks a latent strength mu and a variance sigma². After each game,
ratings update proportionally to their uncertainty (Kalman gain), allowing
higher uncertainty teams to adjust more quickly.

References:
  - research_backlog.md (Priority 2: Kalman-filter Elo)
  - Glicko-1 system (predecessor, but Kalman has separate process/obs noise)
"""

import numpy as np
import pandas as pd

DEFAULT_MU = 1500.0
DEFAULT_K = 36
ELO_DIVISOR = 400.0


def _effective_expected(rating_a: float, rating_b: float, hfa: float = 0.0) -> float:
    diff = (rating_a - rating_b + hfa) / ELO_DIVISOR
    return 1.0 / (1.0 + 10.0 ** (-diff))


def compute_kalman_elo_features(
    df: pd.DataFrame,
    k_factor: float = DEFAULT_K,
    home_advantage: float = 40.0,
    default_mu: float = DEFAULT_MU,
    initial_sigma: float = 200.0,
    obs_noise: float = 100.0,
    preseason_regression: float = 0.1,
    team_regression_overrides: dict[str, float] | None = None,
    decay_half_life: float | None = 32,
    mov_type: str = "none",
    mov_scale: float = 0.0,
    mov_cap: float | None = None,
) -> pd.DataFrame:
    """Add Kalman Elo features to a game-level DataFrame.

    Each team maintains (mu, sigma²). Game outcomes update mu via
    Kalman gain proportional to sigma² / (sigma_h² + sigma_a² + obs_noise²).

    Args:
        df: DataFrame with season, week, gameday, home_team, away_team,
            home_win, home_score, away_score.
        k_factor: Base update scale (like Elo K).
        home_advantage: HFA added to home team's expected score.
        default_mu: Starting rating for new teams.
        initial_sigma: Starting uncertainty for new teams.
        obs_noise: Game outcome observation noise (higher = slower updates).
        preseason_regression: Fraction (0-1) to regress mu toward default_mu
            at season boundary.
        team_regression_overrides: Per-team regression fractions for QB change etc.
        decay_half_life: Half-life in games for rating decay toward default_mu
            (None = disabled).
        mov_type: Margin-of-victory multiplier type ('none', 'log', 'capped_linear').
        mov_scale: Scaling for MOV multiplier.
        mov_cap: Max MOV multiplier.

    Returns:
        DataFrame with additional columns: home_kalman_mu, away_kalman_mu,
        home_kalman_sigma, away_kalman_sigma, kalman_diff, kalman_prob.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    mus: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    prev_season: int | None = None

    if decay_half_life is not None and decay_half_life > 0:
        decay_factor = 2.0 ** (-1.0 / decay_half_life)
    else:
        decay_factor = None

    home_mu = []
    away_mu = []
    home_sigma = []
    away_sigma = []
    diffs = []
    probs = []

    for _, row in out.iterrows():
        home_team: str = row["home_team"]
        away_team: str = row["away_team"]
        season: int = row["season"]

        # Preseason regression at season boundary
        if prev_season is not None and season > prev_season and preseason_regression > 0:
            for team in mus:
                reg_frac = (
                    team_regression_overrides.get(team, preseason_regression)
                    if team_regression_overrides is not None
                    else preseason_regression
                )
                if reg_frac > 0:
                    mus[team] = default_mu + (1.0 - reg_frac) * (mus[team] - default_mu)
                    # Increase uncertainty after regression
                    sigmas[team] = np.sqrt(
                        sigmas[team]**2 + (reg_frac * initial_sigma)**2
                    )
        prev_season = season

        h_mu = mus.get(home_team, default_mu)
        a_mu = mus.get(away_team, default_mu)
        h_sigma = sigmas.get(home_team, initial_sigma)
        a_sigma = sigmas.get(away_team, initial_sigma)

        kalman_prob = _effective_expected(h_mu, a_mu, hfa=home_advantage)

        home_mu.append(h_mu)
        away_mu.append(a_mu)
        home_sigma.append(h_sigma)
        away_sigma.append(a_sigma)
        diffs.append(h_mu - a_mu)
        probs.append(kalman_prob)

        # Update ratings using Kalman gain
        home_won = row["home_win"]
        is_future = pd.isna(home_won) and pd.isna(row.get("home_score"))
        if not is_future:
            if pd.isna(home_won):
                actual_home = 0.5
            else:
                actual_home = float(home_won)

            expected_home = _effective_expected(h_mu, a_mu, hfa=home_advantage)

            # MOV multiplier
            mov_mult = 1.0
            if mov_type != "none" and mov_scale > 0 and not pd.isna(row.get("home_score", None)):
                margin = abs(row["home_score"] - row["away_score"])
                if mov_type == "log":
                    mov_mult = 1.0 + mov_scale * np.log(margin + 1.0)
                elif mov_type == "capped_linear":
                    mov_mult = 1.0 + mov_scale * min(margin, mov_cap or 999)
                mov_mult = min(mov_mult, mov_cap or 999) if mov_cap else mov_mult

            # Kalman gain
            combined_var = h_sigma**2 + a_sigma**2 + obs_noise**2
            gain_h = h_sigma**2 / combined_var
            gain_a = a_sigma**2 / combined_var

            update = k_factor * (actual_home - expected_home) * mov_mult
            mus[home_team] = h_mu + gain_h * update
            mus[away_team] = a_mu - gain_a * update

            # Shrink variance after update
            sigmas[home_team] = h_sigma * np.sqrt(1.0 - gain_h)
            sigmas[away_team] = a_sigma * np.sqrt(1.0 - gain_a)

            # Exponential decay toward mean
            if decay_factor is not None:
                mus[home_team] = default_mu + (mus[home_team] - default_mu) * decay_factor
                mus[away_team] = default_mu + (mus[away_team] - default_mu) * decay_factor
                # Sigma also decays toward initial_sigma
                h_sig = sigmas[home_team]
                a_sig = sigmas[away_team]
                sigmas[home_team] = initial_sigma + (h_sig - initial_sigma) * decay_factor
                sigmas[away_team] = initial_sigma + (a_sig - initial_sigma) * decay_factor

    out["home_kalman_mu"] = home_mu
    out["away_kalman_mu"] = away_mu
    out["home_kalman_sigma"] = home_sigma
    out["away_kalman_sigma"] = away_sigma
    out["kalman_diff"] = diffs
    out["kalman_prob"] = probs

    return out
