import click

from sportslab.data.ingest_nfl import ingest_nfl
from sportslab.evaluation.train_baseline import train_baseline
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
def train_baseline_cmd():
    """Train the first pure non-market logistic regression baseline."""
    train_baseline()
