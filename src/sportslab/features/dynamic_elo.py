"""Dynamic Bayesian Elo: state-space model with MLE-estimated variance parameters.

Key differences from kalman_elo.py:
  - Correct state-space formulation (margin as observation, not binary)
  - MLE estimation of sigma_evolution, sigma_observation, HFA
  - No grid search — parameters fitted from data
  - Standard Kalman filter (not custom gain formula)

Model:
  theta_k = theta_{k-1} + epsilon_k    epsilon_k ~ N(0, sigma_evolution^2 * I)
  y_k = theta_k[home] - theta_k[away]
        + HFA + v_k                     v_k ~ N(0, sigma_observation^2)
"""

import numpy as np
import pandas as pd
from scipy import optimize

DEFAULT_INIT_SIGMA = 20.0


def _log_likelihood(  # noqa: N803, N806
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    margins: np.ndarray,
    n_teams: int,
    init_var: float,
) -> float:
    log_evo_std, log_obs_std, hfa = params
    evo_var = np.exp(2.0 * log_evo_std)
    obs_var = np.exp(2.0 * log_obs_std)

    theta = np.zeros(n_teams, dtype=np.float64)
    P = np.eye(n_teams, dtype=np.float64) * init_var  # noqa: N806

    ll = 0.0
    for k in range(len(margins)):
        h = int(home_idx[k])
        a = int(away_idx[k])

        pred = theta[h] - theta[a] + hfa
        S = P[h, h] + P[a, a] - 2.0 * P[h, a] + obs_var  # noqa: N806
        e = margins[k] - pred

        ll += -0.5 * (np.log(2.0 * np.pi * S) + e * e / S)

        gain = (P[:, h] - P[:, a]) / S
        theta += gain * e
        P -= np.outer(gain, gain) * S  # noqa: N806
        P = (P + P.T) * 0.5  # noqa: N806
        np.fill_diagonal(P, P.diagonal() + evo_var)

    return -ll


class DynamicBayesianElo:
    """State-space team strength model with MLE-estimated variance parameters.

    Parameters estimated by maximum likelihood:
      sigma_evolution   — per-game drift in team strength (points)
      sigma_observation — observation noise (points)
      hfa               — home field advantage (points)
    """

    def __init__(self, initial_sigma: float = DEFAULT_INIT_SIGMA):
        self.team_to_idx: dict[str, int] = {}
        self.teams: list[str] = []
        self.n_teams: int = 0
        self.initial_sigma = initial_sigma
        self.sigma_evolution: float | None = None
        self.sigma_observation: float | None = None
        self.hfa: float | None = None
        self.fitted = False
        self.theta: np.ndarray | None = None
        self.P: np.ndarray | None = None  # noqa: N806
        self._training_margins: np.ndarray | None = None

    def fit(
        self,
        games: pd.DataFrame,
        maxiter: int = 500,
        verbose: bool = False,
    ) -> dict:
        all_teams = sorted(
            set(games["home_team"].unique()) | set(games["away_team"].unique())
        )
        self.team_to_idx = {t: i for i, t in enumerate(all_teams)}
        self.teams = list(all_teams)
        self.n_teams = len(all_teams)

        home_idx = np.array([self.team_to_idx[t] for t in games["home_team"]], dtype=np.int32)
        away_idx = np.array([self.team_to_idx[t] for t in games["away_team"]], dtype=np.int32)
        margins = (games["home_score"].values - games["away_score"].values).astype(np.float64)
        init_var = self.initial_sigma ** 2

        result = optimize.minimize(
            _log_likelihood,
            x0=[np.log(2.0), np.log(14.0), 2.5],
            args=(home_idx, away_idx, margins, self.n_teams, init_var),
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-8},
        )

        self.sigma_evolution = float(np.exp(result.x[0]))
        self.sigma_observation = float(np.exp(result.x[1]))
        self.hfa = float(result.x[2])
        self.fitted = True

        if verbose:
            print(
                f"  MLE: sigma_evo={self.sigma_evolution:.3f} "
                f"sigma_obs={self.sigma_observation:.3f} "
                f"HFA={self.hfa:.3f} LL={-result.fun:.1f}"
            )

        # Run sweep from scratch to get pre-game training predictions + final state
        sweep_result = self._kalman_sweep(games)
        self._training_margins = sweep_result[0]
        self.theta = sweep_result[5].copy()
        self.P = sweep_result[6].copy()  # noqa: N806

        return {
            "sigma_evolution": self.sigma_evolution,
            "sigma_observation": self.sigma_observation,
            "hfa": self.hfa,
            "log_likelihood": -result.fun,
            "converged": result.success,
        }

    def _kalman_sweep(  # noqa: N803
        self,
        games: pd.DataFrame,
        theta_init: np.ndarray | None = None,
        P_init: np.ndarray | None = None,  # noqa: N803
    ) -> tuple:
        """Run Kalman filter over games and return pre-game predictions.

        Args:
            games: DataFrame with home_team, away_team, home_score, away_score.
            theta_init: Starting state (None = zeros).
            P_init: Starting covariance (None = init_var * I).

        Returns:
            (pred_margins, pred_vars, innovations, home_mu, away_mu, theta_final, P_final)
        """
        evo_var = self.sigma_evolution ** 2
        obs_var = self.sigma_observation ** 2
        hfa = self.hfa
        n = len(games)

        if theta_init is not None:
            theta = theta_init.copy()
        else:
            theta = np.zeros(self.n_teams, dtype=np.float64)

        if P_init is not None:
            P = P_init.copy()  # noqa: N806
        else:
            P = np.eye(self.n_teams, dtype=np.float64) * self.initial_sigma ** 2  # noqa: N806

        pred_margins = np.full(n, np.nan, dtype=np.float64)
        pred_vars = np.full(n, np.nan, dtype=np.float64)
        innovations = np.full(n, np.nan, dtype=np.float64)
        home_mu = np.full(n, np.nan, dtype=np.float64)
        away_mu = np.full(n, np.nan, dtype=np.float64)

        for k, (_, row) in enumerate(games.iterrows()):
            h = self.team_to_idx[row["home_team"]]
            a = self.team_to_idx[row["away_team"]]

            home_mu[k] = theta[h]
            away_mu[k] = theta[a]

            pred = theta[h] - theta[a] + hfa
            S = P[h, h] + P[a, a] - 2.0 * P[h, a] + obs_var  # noqa: N806

            pred_margins[k] = pred
            pred_vars[k] = S

            has_score = not pd.isna(row.get("home_score"))
            if has_score:
                margin = float(row["home_score"] - row["away_score"])
                e = margin - pred
                innovations[k] = e
                gain = (P[:, h] - P[:, a]) / S
                theta += gain * e
                P -= np.outer(gain, gain) * S  # noqa: N806
                P = (P + P.T) * 0.5  # noqa: N806
                np.fill_diagonal(P, P.diagonal() + evo_var)  # noqa: N806

        return pred_margins, pred_vars, innovations, home_mu, away_mu, theta, P

    def predict(self, games: pd.DataFrame) -> np.ndarray:
        """Generate pre-game margin predictions for new games.

        Continues from the stored fitted state (theta, P) and generates
        predictions. The stored state is NOT modified — each call is
        independent.

        Returns array of predicted margins.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        sweep = self._kalman_sweep(games, theta_init=self.theta, P_init=self.P)
        return sweep[0]

    def training_margins(self) -> np.ndarray | None:
        """Pre-game margin predictions from the fitted data."""
        return self._training_margins

    def predict_loop(
        self,
        games: pd.DataFrame,
    ) -> np.ndarray:
        """Predict margins, updating state sequentially with observed scores.

        Like predict() but mutates the stored state as it processes games.
        Useful for online evaluation where state evolves with each outcome.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        sweep = self._kalman_sweep(games, theta_init=self.theta, P_init=self.P)
        self.theta = sweep[5].copy()
        self.P = sweep[6].copy()
        return sweep[0]


def compute_dynamic_elo_features(
    df: pd.DataFrame,
    initial_sigma: float = DEFAULT_INIT_SIGMA,
    maxiter: int = 500,
) -> pd.DataFrame:
    """Add Dynamic Bayesian Elo features to a game-level DataFrame.

    Fits the state-space model on all available data, then generates
    pre-game margin predictions for every game.

    Returns:
        DataFrame with added dynamic_elo_margin, dynamic_elo_pred_var,
        dynamic_elo_home_mu, dynamic_elo_away_mu, dynamic_elo_hfa,
        dynamic_elo_sigma_evo, dynamic_elo_sigma_obs.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    model = DynamicBayesianElo(initial_sigma=initial_sigma)
    model.fit(out, maxiter=maxiter, verbose=False)

    out["dynamic_elo_margin"] = model.training_margins()
    out["dynamic_elo_pred_var"] = model._kalman_sweep(out)[1]
    out["dynamic_elo_home_mu"] = model._kalman_sweep(out)[3]
    out["dynamic_elo_away_mu"] = model._kalman_sweep(out)[4]

    out["dynamic_elo_hfa"] = model.hfa
    out["dynamic_elo_sigma_evo"] = model.sigma_evolution
    out["dynamic_elo_sigma_obs"] = model.sigma_observation

    return out
