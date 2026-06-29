import click

from sportslab.data.ingest_nfl import ingest_nfl
from sportslab.evaluation.adaptive_k_experiment import run_adaptive_k_experiment
from sportslab.evaluation.audit_artifacts import run_audit
from sportslab.evaluation.autogluon_experiment import run_autogluon_experiment
from sportslab.evaluation.backtest_2025 import run_backtest, run_backtest_2025
from sportslab.evaluation.build_dashboard import build_all
from sportslab.evaluation.calibration_improvements_experiment import (
    run_calibration_experiment,
)
from sportslab.evaluation.coach_qb_tenure_experiment import (
    run_coach_qb_tenure_experiment,
)
from sportslab.evaluation.coach_season_regression_experiment import (
    run_coach_season_regression_experiment,
)
from sportslab.evaluation.combined_features_experiment import run_combined_experiment
from sportslab.evaluation.comprehensive_efficiency_experiment import (
    run_comprehensive_efficiency_experiment,
)
from sportslab.evaluation.confidence_calibration_experiment import (
    run_confidence_calibration_experiment,
)
from sportslab.evaluation.decayed_elo_experiment import run_decayed_elo_experiment
from sportslab.evaluation.elo_tuning import run_elo_tuning
from sportslab.evaluation.epa_features_experiment import run_epa_features_experiment
from sportslab.evaluation.expressive_models_experiment import run_expressive_models_experiment
from sportslab.evaluation.feature_selection_experiment import run_feature_selection_experiment
from sportslab.evaluation.glicko_experiment import run_glicko_experiment
from sportslab.evaluation.injury_features_experiment import (
    run_injury_features_experiment,
)
from sportslab.evaluation.margin_aware_elo import run_margin_aware_experiment
from sportslab.evaluation.market_baseline import run_market_baseline
from sportslab.evaluation.market_benchmark import run_market_benchmark
from sportslab.evaluation.no_qb_baseline import run_no_qb_baseline
from sportslab.evaluation.qb_ablation import run_qb_ablation
from sportslab.evaluation.qb_continuity import run_qb_continuity
from sportslab.evaluation.qb_depth_experiment import run_qb_depth_experiment
from sportslab.evaluation.qb_gated_experience import run_qb_gated_experience
from sportslab.evaluation.turnover_experiment import run_turnover_experiment
from sportslab.evaluation.optuna_elo_search import run_optuna_search
from sportslab.evaluation.optuna_feature_selection_experiment import run_optuna_feature_selection
from sportslab.evaluation.predict_future import run_predict_future
from sportslab.evaluation.predict_incumbent import generate_incumbent_predictions
from sportslab.evaluation.qb_features_experiment import run_qb_features_experiment
from sportslab.evaluation.qb_injury_experiment import run_qb_injury_experiment
from sportslab.evaluation.qb_magnitude_experiment import run_qb_magnitude_experiment
from sportslab.evaluation.qb_market_delta import run_qb_market_delta_experiment
from sportslab.evaluation.residual_blending_experiment import run_residual_blending_experiment
from sportslab.evaluation.residual_diagnostics import run_residual_diagnostics
from sportslab.evaluation.rolling_origin_elo_validation import (
    run_rolling_origin_validation,
)
from sportslab.evaluation.schedule_rest_experiment import run_schedule_rest_experiment
from sportslab.evaluation.season_regression_experiment import run_season_regression_experiment
from sportslab.evaluation.situational_micro_experiment import run_situational_micro_experiment
from sportslab.evaluation.weekly_pipeline import grade_week, predict_week, season_report
from sportslab.evaluation.team_hfa_experiment import run_team_hfa_experiment
from sportslab.evaluation.train_baseline import train_baseline
from sportslab.evaluation.weather_features_experiment import run_weather_features_experiment
from sportslab.evaluation.weekly_report import generate_weekly_report
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


@cli.command()
def autogluon_cmd():
    """Run AutoGluon AutoML experiment against O/D Elo+Platt incumbent."""
    run_autogluon_experiment()


@cli.command()
def injury_features_cmd():
    """Run injury report feature experiment against O/D Elo+Platt incumbent."""
    run_injury_features_experiment()


@cli.command()
def optuna_search_cmd():
    """Run Optuna joint parameter search for all Elo parameters."""
    run_optuna_search()


@cli.command()
def qb_injury_cmd():
    """Run single-feature QB injury flag experiment."""
    run_qb_injury_experiment()


@cli.command()
def glicko_cmd():
    """Run Glicko rating system experiment against O/D Elo+Platt incumbent."""
    run_glicko_experiment()


@cli.command()
def qb_market_delta_cmd():
    """Run QB-change market-delta diagnostics experiment."""
    run_qb_market_delta_experiment()


@cli.command()
def combined_features_cmd():
    """Run combined feature experiment (qb_changed, mov3, coach)."""
    run_combined_experiment()


@cli.command()
def feature_selection_cmd():
    """Run forward feature selection experiment."""
    run_feature_selection_experiment()


@cli.command()
def optuna_feature_selection_cmd():
    """Run Optuna combinatorial feature selection (500 trials)."""
    run_optuna_feature_selection()


@cli.command()
def predict_incumbent_cmd():
    """Generate incumbent prediction artifacts for all eligible games."""
    generate_incumbent_predictions()


@cli.command()
@click.option("--season", type=int, default=None, help="Season year (default: latest)")
@click.option("--week", type=int, default=None, help="Week number (default: latest)")
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output path (default: reports/predictions/weekly_report.md)",
)
@click.option(
    "--input",
    "input_path",
    type=str,
    default=None,
    help="Input prediction CSV path (default: auto-detect incumbent or future)",
)
def weekly_report_cmd(season, week, output, input_path):
    """Generate weekly report from incumbent or future predictions."""
    generate_weekly_report(season=season, week=week, output=output, input_path=input_path)


@cli.command(name="no-qb-baseline")
def no_qb_baseline_cmd():
    """Compare incumbent vs no-QB live-safe baseline on 2025 simulation."""
    run_no_qb_baseline()


@cli.command(name="qb-ablation")
@click.option("--qb-input", default=None,
              help="Path to QB input CSV for live-QB fixture mode")
def qb_ablation_cmd(qb_input):
    """Comprehensive QB ablation: oracle vs no-QB vs live fixture."""
    run_qb_ablation(qb_input_path=qb_input)


@cli.command(name="qb-continuity")
def qb_continuity_cmd():
    """Narrow QB-continuity refinement experiment (6 model variants)."""
    run_qb_continuity()


@cli.command(name="qb-gated-experience")
def qb_gated_experience_cmd():
    """Gated QB-experience diagnostic experiment (5 variants)."""
    run_qb_gated_experience()


@cli.command(name="qb-depth-experiment")
def qb_depth_cmd():
    """QB depth features experiment (rust, first-season-start, career starts, win %, missing)."""
    run_qb_depth_experiment()


@cli.command(name="turnover-experiment")
def turnover_cmd():
    """Focused turnover features experiment (rolling TO differential windows)."""
    run_turnover_experiment()


@cli.command(name="situational-micro")
def situational_micro_cmd():
    """Situational micro-features experiment (divisional, first-year coach, surface)."""
    run_situational_micro_experiment()


@cli.command()
def comprehensive_efficiency_cmd():
    """Run comprehensive efficiency feature experiment (Team EPA + PFR + Snap)."""
    run_comprehensive_efficiency_experiment()


@cli.command()
def audit_artifacts_cmd():
    """Validate benchmark registry, predictions, and report consistency."""
    issues = run_audit()
    if issues:
        raise SystemExit(f"Audit failed: {len(issues)} issue(s) found")
    click.echo("✅ Artifact audit passed — all checks OK.")


@cli.command()
def build_dashboard_cmd():
    """Build static GitHub Pages dashboard from benchmark registry and prediction artifacts."""
    build_all()


@cli.command()
def backtest_2025_cmd():
    """Run comprehensive 2025 backtest analysis for the incumbent model."""
    run_backtest_2025()


@cli.command()
@click.argument("seasons", nargs=-1, type=int)
def backtest(seasons):
    """Run backtest for one or more seasons (e.g. 'sportslab backtest 2022 2023')."""
    if not seasons:
        click.echo("Usage: sportslab backtest <season> [season ...]")
        return
    results = run_backtest(list(seasons))
    click.echo(f"Report: {results['report']}")


@cli.command()
def qb_magnitude():
    """Run QB change magnitude experiment (continuous QB quality dropoff features)."""
    run_qb_magnitude_experiment()


@cli.command(name="calibration-improvements")
def calibration_improvements():
    """Run calibration improvements experiment (era split + high-confidence shrinkage)."""
    run_calibration_experiment()


@cli.command(name="adaptive-k")
def adaptive_k():
    """Run adaptive Elo K by week experiment."""
    run_adaptive_k_experiment()


@cli.command(name="coach-qb-tenure")
def coach_qb_tenure():
    """Run coach-QB tenure experiment."""
    run_coach_qb_tenure_experiment()


@cli.command(name="simulate-2025")
@click.option("--qb-input", type=str, default=None,
              help="CSV with game_id,home_qb_id,away_qb_id for live-safe QB starters")
@click.option("--output", type=str, default=None, help="Output CSV path")
@click.option("--report", type=str, default=None, help="Report output path")
def simulate_2025_cmd(qb_input, output, report):
    """Run week-by-week as-if-future simulation for 2025.

    Simulates what the incumbent would have predicted if used
    in real time, fitting Elo on all available data before each week.

    Uses oracle QB data by default; provide --qb-input for live-safe mode.
    """
    from sportslab.evaluation.simulate_2025 import simulate_2025
    simulate_2025(
        qb_input_path=qb_input,
        output_path=output or "reports/simulations/simulate_2025_results.csv",
        report_path=report or "reports/simulations/simulate_2025_report.md",
    )


@cli.command(name="predict-future")
@click.option("--input", type=str, default=None, help="CSV with future games to predict")
@click.option("--output", type=str, default=None, help="Output CSV path")
@click.option("--qb-input", type=str, default=None,
              help="CSV with game_id,home_qb_id,away_qb_id for live-safe QB starters")
@click.option("--season", type=int, default=None,
              help="Season year to predict (default: all future)")
@click.option("--week", type=int, default=None,
              help="Week number to predict (default: all future)")
def predict_future_cmd(input, output, qb_input, season, week):
    """Generate pregame predictions for future games without requiring scores.

    Fits Elo on all available historical data (2021-current),
    then emits pregame-safe predictions using the incumbent
    (Elo + qb_changed + rolling_mov_3 + Platt) without updating
    ratings on the predicted games.

    By default uses oracle QB data from nflreadpy schedules.
    Provide --qb-input to override with live-safe pregame-announced starters.
    """
    run_predict_future(input_path=input, output=output, qb_input=qb_input,
                       season=season, week=week)


@cli.command(name="predict-week")
@click.option("--season", type=int, required=True, help="Season year (e.g. 2026)")
@click.option("--week", type=int, required=True, help="Week number (1-22)")
@click.option("--qb-input", type=str, default=None,
              help="CSV with game_id,home_qb_id,away_qb_id for live-safe QB starters")
@click.option("--output", type=str, default=None, help="Override snapshot output path")
def predict_week_cmd(season, week, qb_input, output):
    """Generate predictions + snapshot + report for a single week.

    Fits Elo on all historical data (2021+), predicts the specified
    week, saves a timestamped snapshot and generates a weekly report.
    """
    predict_week(season=season, week=week, qb_input=qb_input, snapshot_path=output)


@cli.command(name="grade-week")
@click.option("--season", type=int, required=True, help="Season year")
@click.option("--week", type=int, required=True, help="Week number (1-22)")
@click.option("--snapshot", type=str, default=None,
              help="Snapshot CSV path (auto-detected if not provided)")
def grade_week_cmd(season, week, snapshot):
    """Grade a completed week's predictions against actual results.

    Loads the prediction snapshot, merges actual results from the
    feature table, computes metrics, and appends to prediction history.
    """
    grade_week(season=season, week=week, snapshot=snapshot)


@cli.command(name="season-report")
@click.option("--season", type=int, required=True, help="Season year")
def season_report_cmd(season):
    """Generate cumulative season dashboard from prediction history.

    Reads all graded weeks and produces a dashboard with per-week
    metrics, cumulative totals, and model metadata.
    """
    season_report(season=season)
