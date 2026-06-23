import click

from sportslab.data.ingest_nfl import ingest_nfl
from sportslab.evaluation.coach_season_regression_experiment import (
    run_coach_season_regression_experiment,
)
from sportslab.evaluation.confidence_calibration_experiment import (
    run_confidence_calibration_experiment,
)
from sportslab.evaluation.decayed_elo_experiment import run_decayed_elo_experiment
from sportslab.evaluation.elo_tuning import run_elo_tuning
from sportslab.evaluation.epa_features_experiment import run_epa_features_experiment
from sportslab.evaluation.expressive_models_experiment import run_expressive_models_experiment
from sportslab.evaluation.margin_aware_elo import run_margin_aware_experiment
from sportslab.evaluation.market_baseline import run_market_baseline
from sportslab.evaluation.market_benchmark import run_market_benchmark
from sportslab.evaluation.qb_features_experiment import run_qb_features_experiment
from sportslab.evaluation.residual_blending_experiment import run_residual_blending_experiment
from sportslab.evaluation.residual_diagnostics import run_residual_diagnostics
from sportslab.evaluation.rolling_origin_elo_validation import (
    run_rolling_origin_validation,
)
from sportslab.evaluation.schedule_rest_experiment import run_schedule_rest_experiment
from sportslab.evaluation.season_regression_experiment import run_season_regression_experiment
from sportslab.evaluation.team_hfa_experiment import run_team_hfa_experiment
from sportslab.evaluation.train_baseline import train_baseline
from sportslab.evaluation.weather_features_experiment import run_weather_features_experiment
from sportslab.features.build_features import build_feature_table


@click.group()
def cli():
    """Sports ML Lab CLI — NFL-first sports research tools."""
    pass


@cli.command()
@click.argument("seasons", nargs=-1, type=int)
def ingest_nfl_cmd(seasons):
    """Ingest NFL schedule data for specified seasons using nflreadpy.

    SEASONS are one or more season years (minimum: 2021).

    Example: sportslab ingest-nfl 2021 2022 2023 2024 2025
    """
    ingest_nfl(list(seasons))


@cli.command()
@click.option(
    "--weather/--no-weather",
    default=True,
    help="Fetch real weather data via meteostat (default: True)",
)
def build_features_cmd(weather):
    """Build the first feature table from ingested schedules."""
    build_feature_table(fetch_weather=weather)


@cli.command()
@click.option(
    "--feature-set",
    type=click.Choice(["baseline", "team_strength"], case_sensitive=False),
    default="baseline",
    help="Feature set to use for training (default: baseline)",
)
def train_baseline_cmd(feature_set):
    """Train a logistic regression baseline.

    Supports two feature sets:
    - baseline: label-encoded identity features (team, QB, coach, stadium)
    - team_strength: Elo ratings + rolling features + structural features
    """
    train_baseline(feature_set=feature_set)


@cli.command()
def elo_tuning_cmd():
    """Run Elo tuning, calibration, and comparison experiment."""
    run_elo_tuning()


@cli.command()
def rolling_origin_cmd():
    """Run rolling-origin Elo validation with expanded grid."""
    run_rolling_origin_validation()


@cli.command()
def schedule_features_cmd():
    """Run scheduling/rest feature experiment on top of Elo+Platt."""
    run_schedule_rest_experiment()


@cli.command()
def margin_aware_elo_cmd():
    """Run margin-aware Elo experiment with rolling-origin validation."""
    run_margin_aware_experiment()


@cli.command()
def qb_features_cmd():
    """Run QB starter/change feature experiment on top of MOV Elo+Platt."""
    run_qb_features_experiment()


@cli.command()
def weather_features_cmd():
    """Run weather feature experiment on top of MOV Elo+Platt."""
    run_weather_features_experiment()


@cli.command()
def expressive_models_cmd():
    """Run constrained expressive models experiment on curated features."""
    run_expressive_models_experiment()


@cli.command()
def market_baseline_cmd():
    """Run market baseline comparison against incumbent."""
    run_market_baseline()


@cli.command()
def market_benchmark_cmd():
    """Run comprehensive market benchmark and Elo-vs-market diagnostics."""
    run_market_benchmark()


@cli.command()
def residual_diagnostics_cmd():
    """Run residual diagnostics on the incumbent."""
    run_residual_diagnostics()


@cli.command()
def epa_features_cmd():
    """Run EPA team-efficiency feature experiment."""
    run_epa_features_experiment()


@cli.command()
def confidence_calibration_cmd():
    """Run confidence calibration and probability shrinkage experiment."""
    run_confidence_calibration_experiment()


@cli.command()
def decayed_elo_cmd():
    """Run decayed Elo (exponential momentum) experiment."""
    run_decayed_elo_experiment()


@cli.command()
def team_hfa_cmd():
    """Run team-specific HFA experiment."""
    run_team_hfa_experiment()


@cli.command()
def season_regression_cmd():
    """Run season-specific regression experiment."""
    run_season_regression_experiment()


@cli.command()
def residual_blending_cmd():
    """Run residual blending experiment."""
    run_residual_blending_experiment()


@cli.command()
def coach_season_regression_cmd():
    """Run coach+QB season regression experiment."""
    run_coach_season_regression_experiment()
