import click

from sportslab.data.ingest_nfl import ingest_nfl
from sportslab.evaluation.adaptive_k_experiment import run_adaptive_k_experiment
from sportslab.evaluation.audit_artifacts import run_audit
from sportslab.evaluation.autogluon_experiment import run_autogluon_experiment
from sportslab.evaluation.backtest_2025 import run_backtest, run_backtest_2025
from sportslab.evaluation.build_dashboard import build_all_docs
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
from sportslab.evaluation.data_audit import run_data_audit
from sportslab.evaluation.decayed_elo_experiment import run_decayed_elo_experiment
from sportslab.evaluation.elo_tuning import run_elo_tuning
from sportslab.evaluation.epa_features_experiment import run_epa_features_experiment
from sportslab.evaluation.expanded_elo_spine_experiment import run_expanded_elo_spine
from sportslab.evaluation.expressive_models_experiment import run_expressive_models_experiment
from sportslab.evaluation.feature_selection_experiment import run_feature_selection_experiment
from sportslab.evaluation.glicko_experiment import run_glicko_experiment
from sportslab.evaluation.injury_features_experiment import (
    run_injury_features_experiment,
)
from sportslab.evaluation.live_preflight import run_live_preflight
from sportslab.evaluation.margin_aware_elo import run_margin_aware_experiment
from sportslab.evaluation.market_baseline import run_market_baseline
from sportslab.evaluation.market_benchmark import run_market_benchmark
from sportslab.evaluation.no_qb_baseline import run_no_qb_baseline
from sportslab.evaluation.optuna_elo_search import run_optuna_search
from sportslab.evaluation.optuna_feature_selection_experiment import run_optuna_feature_selection
from sportslab.evaluation.predict_future import run_predict_future
from sportslab.evaluation.predict_incumbent import generate_incumbent_predictions
from sportslab.evaluation.prediction_audit import build_prediction_index, run_prediction_audit
from sportslab.evaluation.qb_ablation import run_qb_ablation
from sportslab.evaluation.qb_adjusted_elo_experiment import run_qb_adjusted_elo_experiment
from sportslab.evaluation.qb_continuity import run_qb_continuity
from sportslab.evaluation.qb_depth_experiment import run_qb_depth_experiment
from sportslab.evaluation.qb_features_experiment import run_qb_features_experiment
from sportslab.evaluation.qb_gated_experience import run_qb_gated_experience
from sportslab.evaluation.qb_injury_experiment import run_qb_injury_experiment
from sportslab.evaluation.qb_magnitude_experiment import run_qb_magnitude_experiment
from sportslab.evaluation.qb_market_delta import run_qb_market_delta_experiment
from sportslab.evaluation.qb_roster_interaction_experiment import run_qb_roster_interaction
from sportslab.evaluation.gam_logistic_experiment import run_gam_logistic_experiment
from sportslab.evaluation.gradient_boosting_diagnostic import (
    run_gradient_boosting_diagnostic,
)
from sportslab.evaluation.learned_overlay_experiment import (
    run_learned_overlay_experiment,
)
from sportslab.evaluation.regularized_logistic_experiment import (
    run_regularized_logistic_experiment,
)
from sportslab.evaluation.rehearsal_season import rehearse_season
from sportslab.evaluation.residual_blending_experiment import run_residual_blending_experiment
from sportslab.evaluation.residual_diagnostics import run_residual_diagnostics
from sportslab.evaluation.rolling_origin_elo_validation import (
    run_rolling_origin_validation,
)
from sportslab.evaluation.roster_overlay_foldsafe_experiment import run_roster_overlay_foldsafe
from sportslab.evaluation.schedule_rest_experiment import run_schedule_rest_experiment
from sportslab.evaluation.season_regression_experiment import run_season_regression_experiment
from sportslab.evaluation.situational_micro_experiment import run_situational_micro_experiment
from sportslab.evaluation.team_hfa_experiment import run_team_hfa_experiment
from sportslab.evaluation.train_baseline import train_baseline
from sportslab.evaluation.turnover_experiment import run_turnover_experiment
from sportslab.evaluation.weather_features_experiment import run_weather_features_experiment
from sportslab.evaluation.weekly_qb_audit import run_weekly_qb_audit
from sportslab.evaluation.weekly_pipeline import grade_week, predict_week, season_report
from sportslab.evaluation.weekly_report import generate_weekly_report
from sportslab.features.build_features import build_feature_table
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.ratings.roster_strength import compute_roster_strength


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
    build_all_docs()


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
@click.option("--auto-qb", is_flag=True, default=False,
              help="Auto-source QB starters from nflreadpy depth charts (preseason snapshot)")
@click.option("--weekly-qb", is_flag=True, default=False,
              help="Auto-source QB starters using week-over-week tracking (more accurate mid-season)")
@click.option("--output", type=str, default=None, help="Override snapshot output path")
@click.option("--mode", type=click.Choice(["live", "dry_run", "rehearsal"]),
              default="live", help="Snapshot mode (default: live)")
def predict_week_cmd(season, week, qb_input, auto_qb, weekly_qb, output, mode):
    """Generate predictions + snapshot + report for a single week.

    Fits Elo on all historical data (2021+), predicts the specified
    week, saves a timestamped snapshot and generates a weekly report.

    In live mode (default), oracle QB data is blocked. Provide --qb-input
    for a manual CSV, --auto-qb for depth chart snapshot (67% accurate),
    or --weekly-qb for week-over-week tracking (88% accurate).

    --weekly-qb is recommended over --auto-qb for week 2+ predictions
    because it catches mid-season QB changes missed by the preseason
    snapshot.
    """
    if qb_input and (auto_qb or weekly_qb):
        print("WARNING: Both --qb-input and --auto-qb/--weekly-qb provided. Using --qb-input.")
        auto_qb = False
        weekly_qb = False
    predict_week(season=season, week=week, qb_input=qb_input,
                 snapshot_path=output, mode=mode, auto_qb=auto_qb,
                 weekly_qb=weekly_qb)


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


@cli.command(name="data-audit")
@click.option("--seasons", type=str, default=None,
              help="Comma-separated season list to check (default: all)")
def data_audit_cmd(seasons):
    """Validate schedule and feature table health.

    Checks season coverage, row counts, column presence,
    duplicates, score consistency, and incumbent feature safety.
    """
    season_list = [int(s.strip()) for s in seasons.split(",")] if seasons else None
    run_data_audit(seasons=season_list)


@cli.command(name="live-preflight")
@click.option("--qb-input", type=str, default=None,
              help="Path to live QB starter CSV for validation")
@click.option("--seasons", type=str, default=None,
              help="Comma-separated season list to check (default: all)")
def live_preflight_cmd(qb_input, seasons):
    """Run pre-week checklist: data audit, QB validation, dry-run smoke test.

    Runs the full preflight checklist before a live prediction week:
    1. Data audit (structure, staleness, partial ingest)
    2. QB input CSV validation (if provided)
    3. Dry-run predict smoke test

    Reports pass/fail for each step.
    """
    season_list = [int(s.strip()) for s in seasons.split(",")] if seasons else None
    run_live_preflight(qb_input_path=qb_input, seasons=season_list)


@cli.command(name="prediction-audit")
@click.option("--season", type=int, required=True, help="Season year")
def prediction_audit_cmd(season):
    """Generate prediction audit report for a season.

    Reads graded snapshots from manifest, produces calibration buckets,
    confidence buckets, worst-prediction ledger, QB-source breakdown,
    week-by-week table, and history validation.
    """
    run_prediction_audit(season=season, mode="live")


@cli.command(name="build-prediction-index")
def build_prediction_index_cmd():
    """Generate docs/predictions/index.md from manifest state."""
    build_prediction_index()


@cli.command(name="rehearsal-season")
@click.option("--season", type=int, default=2025, help="Season year to rehearse")
@click.option(
    "--qb-input", type=str, default=None,
    help="CSV with game_id,home_qb_id,away_qb_id for live-safe QB starters",
)
def rehearsal_season_cmd(season, qb_input):
    """Replay a completed season through the weekly prediction pipeline.

    Generates week-by-week prediction snapshots, grades each week,
    and produces a season report and prediction audit — all isolated
    to ``reports/predictions/rehearsal/``.

    Does NOT modify live prediction artifacts or the incumbent model.
    """
    rehearse_season(season=season, qb_input_path=qb_input)


@cli.command(name="build-qb-adjustments")
@click.option("--output", type=str, default="data/features/nfl/qb_adjustments.parquet",
              help="Output path for QB adjustments parquet")
def build_qb_adjustments_cmd(output):
    """Build QB adjustment features and export as parquet.

    Computes per-game QB strength adjustments (in Elo points) with
    Bayesian shrinkage toward replacement level.  Exports all games
    with home_qb_adj, away_qb_adj, home_qb_starts, away_qb_starts.
    """
    import pandas as pd

    from sportslab.evaluation.season_regression_experiment import (
        build_team_regression_overrides,
    )
    from sportslab.features.ratings import compute_elo_features

    fp = "data/features/nfl/feature_table.parquet"
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=0.1, qb_change_bonus=0.2,
    )
    df = compute_elo_features(
        df_raw, k_factor=36, home_advantage=40,
        preseason_regression=0.1, team_regression_overrides=overrides,
        decay_half_life=32,
    )
    df = compute_qb_adjustments(df)

    out_cols = [
        "game_id", "season", "week", "gameday",
        "away_team", "home_team", "home_qb_id", "away_qb_id",
        "home_qb_name", "away_qb_name",
        "home_elo_pre", "away_elo_pre",
        "home_qb_adj", "away_qb_adj",
        "home_qb_starts", "away_qb_starts",
    ]
    avail = [c for c in out_cols if c in df.columns]
    out_path = str(output)
    df[avail].to_parquet(out_path, index=False)
    click.echo(f"QB adjustments saved: {out_path} ({len(df)} games, {len(avail)} columns)")


@cli.command(name="qb-adjusted-elo")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def qb_adjusted_elo_cmd(output):
    """Run QB-adjusted Elo experiment V0: 5 models, rolling validation, holdout."""
    run_qb_adjusted_elo_experiment(output_csv=output)


@cli.command(name="frozen-qb-overlay")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def frozen_qb_overlay_cmd(output):
    """Run frozen-incumbent QB overlay experiment: final QB promotion test.

    Applies QB adjustment ONLY in logit space on top of frozen incumbent
    probabilities.  Non-gated games are identical to the incumbent.
    No recalibration after gating.
    """
    from sportslab.evaluation.frozen_qb_overlay_experiment import (
        run_frozen_qb_overlay_experiment,
    )
    run_frozen_qb_overlay_experiment(output_csv=output)


@cli.command(name="frozen-qb-overlay-foldsafe")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def frozen_qb_overlay_foldsafe_cmd(output):
    """Run fold-safe frozen-incumbent QB overlay experiment.

    Fits Platt calibration per rolling-origin fold to avoid future-data
    leakage in validation.  Selects best variant by average val LL.
    Evaluates selected variant once on 2025 holdout.
    """
    from sportslab.evaluation.frozen_qb_overlay_foldsafe_experiment import (
        run_frozen_qb_overlay_foldsafe,
    )
    run_frozen_qb_overlay_foldsafe(output_csv=output)


@cli.command(name="learned-overlay")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def learned_overlay_cmd(output):
    """Run learned overlay experiment.

    Tests whether a single regularized logistic model can learn better
    weights for QB adjustment signals than the hand-tuned v3.0.0 overlay.
    """
    run_learned_overlay_experiment(output_csv=output)


@cli.command(name="gradient-boosting")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def gradient_boosting_cmd(output):
    """Run gradient boosting diagnostic experiment.

    Tests whether a strictly regularized HistGradientBoostingClassifier
    can match the v3.0.0 Frozen QB Overlay. Diagnostic only.
    """
    run_gradient_boosting_diagnostic(output_csv=output)


@cli.command(name="gam-logistic")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def gam_logistic_cmd(output):
    """Run GAM/spline logistic experiment.

    Tests whether a nonlinear (spline) transformation of elo_prob improves
    the v3.0.0 Frozen QB Overlay champion.
    """
    run_gam_logistic_experiment(output_csv=output)


@cli.command(name="regularized-logistic")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def regularized_logistic_cmd(output):
    """Run regularized logistic meta-model experiment.

    Tests whether a regularized logistic regression on the v3.0.0 incumbent
    logit plus additional pregame features (rest_diff, is_dome, div_game,
    week, game_type) improves on the v3.0.0 Frozen QB Overlay champion.
    """
    run_regularized_logistic_experiment(output_csv=output)


@cli.command(name="roster-overlay")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def roster_overlay_cmd(output):
    """Run fold-safe roster overlay experiment.

    Tests position-group availability overlays (OL, skill, front, LB,
    coverage) on top of the frozen v3.0.0 incumbent. Selects best variant
    by average validation log loss.
    """
    run_roster_overlay_foldsafe(output_csv=output)


@cli.command(name="qb-roster-interaction")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def qb_roster_cmd(output):
    """Run QB × roster interaction overlay experiment.

    Tests whether position-group availability overlays improve prediction
    when applied only on top of games where the QB overlay gate is active.
    Layer 1: frozen QB overlay champion (H.changed OR starts<17, cap=40,
    gamma=1.0). Layer 2: position-group overlay swept over gamma/threshold/cap.
    """
    run_qb_roster_interaction(output_csv=output)


@cli.command(name="expanded-elo-spine")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def expanded_elo_spine_cmd(output):
    """Run expanded Elo spine + frozen QB overlay experiment.

    Tests whether a better base Elo probability (wider K/HFA/regression/decay
    grid) improves the v3.0.0 Frozen QB Overlay champion.
    """
    run_expanded_elo_spine(output_csv=output)


@cli.command(name="gated-qb-elo")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path for holdout predictions")
def gated_qb_elo_cmd(output):
    """Run gated QB-adjusted Elo experiment V1: 7+ gating variants, sweep."""
    from sportslab.evaluation.gated_qb_adjusted_elo_experiment import (
        run_gated_qb_adjusted_elo_experiment,
    )
    run_gated_qb_adjusted_elo_experiment(output_csv=output)


@cli.command(name="roster-strength")
@click.option("--output", type=str, default="data/features/nfl/roster_strength.parquet",
              help="Output path for roster strength parquet")
def roster_strength_cmd(output):
    """Build V1 roster-strength features (only QB points populated in V0).

    Creates the full roster-strength feature table with position-group
    points.  Only QB points are populated (via qb_adjustment); other
    position groups return 0 in V0.  Ready for V1+ expansion.
    """
    import pandas as pd

    from sportslab.evaluation.season_regression_experiment import (
        build_team_regression_overrides,
    )
    from sportslab.features.ratings import compute_elo_features

    fp = "data/features/nfl/feature_table.parquet"
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=0.1, qb_change_bonus=0.2,
    )
    df = compute_elo_features(
        df_raw, k_factor=36, home_advantage=40,
        preseason_regression=0.1, team_regression_overrides=overrides,
        decay_half_life=32,
    )
    df = compute_roster_strength(df)

    from sportslab.ratings.roster_strength import ALL_ROSTER_COLUMNS
    out_cols = [
        "game_id", "season", "week", "gameday",
        "away_team", "home_team",
        "home_elo_pre", "away_elo_pre",
    ] + ALL_ROSTER_COLUMNS
    avail = [c for c in out_cols if c in df.columns]
    out_path = str(output)
    df[avail].to_parquet(out_path, index=False)
    click.echo(f"Roster strength saved: {out_path} ({len(df)} games, {len(avail)} columns)")
    click.echo("  Note: V0 only populates QB points. OL/skill/DL/LB/coverage/ST are 0.")


@cli.command(name="weekly-qb-audit")
@click.option("--season", type=int, required=True, help="Season year")
@click.option("--week", type=int, required=True, help="Week number (1-22)")
@click.option("--output", type=str, default=None,
              help="Optional CSV output path")
def weekly_qb_audit_cmd(season, week, output):
    """Audit QB sourcing strategies for a single week.

    Compares oracle, depth chart snapshot, and weekly tracker QB
    sources — shows per-game QB identity, overlay gate status,
    overlay delta, and final win probability under each source.
    """
    run_weekly_qb_audit(season=season, week=week, output_path=output)
