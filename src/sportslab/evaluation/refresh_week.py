"""Weekly refresh pipeline: ingest scores → rebuild features → repredict → rebuild site.

Usage:
    sportslab refresh-week [--week WEEK]
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
PREDICTIONS_CSV = BASE / "reports" / "predictions" / "2026_season_predictions.csv"
FEATURE_TABLE = BASE / "data" / "features" / "nfl" / "feature_table.parquet"


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  $ {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=BASE)
    if result.returncode != 0:
        print(f"  ERROR: command failed with code {result.returncode}")
        sys.exit(result.returncode)
    print("  Done\n")


def load_predictions():
    if PREDICTIONS_CSV.exists():
        return pd.read_csv(PREDICTIONS_CSV)
    return None


def report_graded_week(preds_before, week):
    if preds_before is None:
        return
    ft = pd.read_parquet(FEATURE_TABLE)
    wk_games = ft[(ft["season"] == 2026) & (ft["week"] == week) & ft["home_score"].notna()]
    if len(wk_games) == 0:
        return
    wk_games = wk_games.copy()
    wk_games["actual_home_win"] = (wk_games["home_score"] > wk_games["away_score"]).astype(int)
    wk_games = wk_games[~wk_games["is_tie"]]

    wk_preds = preds_before[preds_before["week"] == week].copy()
    merged = wk_games.merge(wk_preds, on="game_id", suffixes=("_actual", "_pred"), how="inner")
    if len(merged) == 0:
        return

    from sklearn.metrics import log_loss
    y_true = merged["actual_home_win"]
    y_prob = merged["incumbent_home_win_prob"]
    ll = log_loss(y_true, y_prob, labels=[0, 1])
    acc = ((y_prob > 0.5) == y_true.astype(bool)).mean()
    print(f"\n  Week {week} Results:")
    print(f"     Games: {len(merged)}")
    print(f"     Log Loss: {ll:.4f}")
    print(f"     Accuracy: {acc:.1%}")

    winners = merged[merged["actual_home_win"] == 1]["home_team"].tolist()
    print(f"     Home winners: {', '.join(winners[:5])}{'...' if len(winners) > 5 else ''}")

    return ll, acc


def report_future_predictions():
    preds = load_predictions()
    if preds is None:
        return
    future = preds[preds["incumbent_home_win_prob"].notna()]
    if len(future):
        weeks = sorted(future["week"].unique())
        total = len(future)
        print(f"\n  Future predictions: {total} games across weeks {weeks[0]}-{weeks[-1]}")
        for w in weeks:
            wk = future[future["week"] == w]
            home_fav = (wk["incumbent_home_win_prob"] > 0.5).sum()
            print(f"     Week {w}: {len(wk)} games ({home_fav} home favorites)")


def refresh_week(week=None):
    print(f"\n{'#'*60}")
    print("  Sports ML Lab — Weekly Refresh Pipeline")
    if week:
        print(f"  Grading week {week}, predicting weeks {week+1}-18")
    else:
        print("  Full refresh (all seasons)")
    print(f"{'#'*60}\n")

    preds_before = load_predictions()

    run("sportslab ingest-nfl 2026", "Step 1: Ingest updated schedule (with actual scores)")

    run("sportslab build-features",
        "Step 2: Rebuild feature table (recompute home_win, model_eligible)")

    if week and preds_before is not None:
        report_graded_week(preds_before, week)

    outpath = "reports/predictions/2026_season_predictions.csv"
    run(f"sportslab predict-future --season 2026 --output {outpath}",
        "Step 3: Regenerate predictions for remaining weeks")

    report_future_predictions()

    run("sportslab build-team-site", "Step 4: Rebuild team site")

    print(f"\n{'#'*60}")
    print("  Weekly refresh complete!")
    print("  Site: site/index.html")
    print("  Predictions: reports/predictions/2026_season_predictions.csv")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    week = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--week"):
        week = int(sys.argv[1].split("=")[-1] if "=" in sys.argv[1] else sys.argv[2])
    refresh_week(week)
