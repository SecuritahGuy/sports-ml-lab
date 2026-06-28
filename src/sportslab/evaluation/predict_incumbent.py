"""Incumbent prediction artifact generation and registry validation.

Produces reproducible prediction CSVs for all model-eligible games using
the current clean football-only incumbent model:
  Standard Elo (K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2)
  + qb_changed + rolling_mov_3 + Platt scaling
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.coach import compute_coach_features
from sportslab.features.market import compute_market_features
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

INCUMBENT_VERSION = "v2.0.0"
INCUMBENT_DATE = "2026-06-23"
INCUMBENT_VAL_LL = 0.6334
INCUMBENT_HOLDOUT_LL = 0.6262
INCUMBENT_FEATURE_SET = "qb_changed + rolling_mov_3"
INCUMBENT_CALIBRATION = "Platt (logistic on Elo prob + features)"
INCUMBENT_REPORT = "reports/experiments/combined_features.md"
INCUMBENT_REGISTRY = "reports/benchmarks/nfl_research_incumbent.md"
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
HOLDOUT_SEASON = 2025
FEATURE_COLS = [
    "home_qb_changed",
    "away_qb_changed",
    "home_rolling_mov_3",
    "away_rolling_mov_3",
]
TRAIN_SEASONS = [2021, 2022, 2023, 2024]

CONFIDENCE_BINS = [
    (0.50, 0.55, "50-55"),
    (0.55, 0.60, "55-60"),
    (0.60, 0.65, "60-65"),
    (0.65, 0.70, "65-70"),
    (0.70, 0.80, "70-80"),
    (0.80, 1.01, "80+"),
]


def _assign_confidence_bucket(prob: float) -> str:
    for lo, hi, label in CONFIDENCE_BINS:
        if lo <= prob < hi:
            return label
    return "80+" if prob >= 0.80 else "50-55"


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def _build_feature_pipeline() -> pd.DataFrame:
    fp = Path("data/features/nfl/feature_table.parquet")
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=BEST_REG,
        qb_change_bonus=BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)
    df = compute_coach_features(df)
    df = compute_market_features(df)
    return df


def _fit_incumbent(df: pd.DataFrame) -> Pipeline:
    is_train = df["season"].isin(TRAIN_SEASONS).values
    train_elo = df.loc[is_train, "elo_prob"].values
    train_y = df.loc[is_train, TARGET_COLUMN].astype(int).values
    train_feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    train_feat = df.loc[is_train, train_feat_cols].values
    x_train = np.column_stack([train_elo, train_feat])
    pipe = _build_pipeline()
    pipe.fit(x_train, train_y)
    return pipe


def _add_caution_flags(
    df_out: pd.DataFrame,
    df: pd.DataFrame,
    prob: np.ndarray,
) -> pd.DataFrame:
    out = df_out.copy()
    qb_flag_cols = [c for c in ["home_qb_changed", "away_qb_changed"] if c in df.columns]
    if qb_flag_cols:
        out["caution_qb_change"] = df[qb_flag_cols].any(axis=1).astype(int)
    else:
        out["caution_qb_change"] = 0

    out["caution_neutral"] = df.get(NEUTRAL_COLUMN, pd.Series(False)).astype(int)

    week_col = df.get("week", pd.Series(1))
    out["caution_early_season"] = (week_col <= 4).astype(int)

    missing_feat_cols = [c for c in FEATURE_COLS if c not in df.columns]
    out["caution_missing_features"] = int(len(missing_feat_cols) > 0)

    if "market_home_prob_novig" in df.columns:
        market_prob = df["market_home_prob_novig"].values
        disagreement = np.abs(prob - market_prob)
        out["caution_model_market_disagreement"] = (disagreement > 0.15).astype(int)
        out["market_model_diff"] = (market_prob - prob).round(4)
    else:
        out["caution_model_market_disagreement"] = 0
        out["market_model_diff"] = np.nan

    return out


def generate_incumbent_predictions() -> Dict[str, str]:
    print("=== Building feature table ===")
    df = _build_feature_pipeline()

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)

    print(f"  Eligible games: {len(df)}")

    print("=== Fitting incumbent model on 2021-2024 ===")
    pipe = _fit_incumbent(df)
    print("  Fit complete")

    elo_prob = df["elo_prob"].values
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    feat_vals = df[feat_cols].values if feat_cols else np.empty((len(df), 0))
    x_all = np.column_stack([elo_prob, feat_vals]) if feat_vals.size else elo_prob.reshape(-1, 1)
    prob = pipe.predict_proba(x_all)[:, 1]
    pred_winner = np.where(prob >= 0.5, df["home_team"], df["away_team"])

    df_out = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "season": df["season"],
            "week": df["week"],
            "gameday": df["gameday"],
            "away_team": df["away_team"],
            "home_team": df["home_team"],
            "home_score": df.get("home_score", pd.Series([np.nan] * len(df))).astype("Int64"),
            "away_score": df.get("away_score", pd.Series([np.nan] * len(df))).astype("Int64"),
            "result": df.get("result", pd.Series([np.nan] * len(df))),
            "home_win_actual": df.get(TARGET_COLUMN, pd.Series([np.nan] * len(df))),
            "incumbent_home_win_prob": prob,
            "predicted_winner": pred_winner,
            "confidence_bucket": [_assign_confidence_bucket(p) for p in prob],
            "model_version": INCUMBENT_VERSION,
            "feature_set": INCUMBENT_FEATURE_SET,
            "calibration_method": INCUMBENT_CALIBRATION,
        }
    )

    df_out = _add_caution_flags(df_out, df, prob)

    if "market_home_prob_novig" in df.columns:
        df_out["market_prob_diagnostic"] = df["market_home_prob_novig"].round(4)
        df_out["market_minus_model_diagnostic"] = (df["market_home_prob_novig"] - prob).round(4)
    else:
        df_out["market_prob_diagnostic"] = np.nan
        df_out["market_minus_model_diagnostic"] = np.nan

    if "home_qb_changed" in df.columns:
        qb_home = df["home_qb_changed"].astype(int)
        qb_away = df["away_qb_changed"].astype(int)
        df_out["qb_change_flag"] = (qb_home | qb_away).astype(int)
    else:
        df_out["qb_change_flag"] = 0

    # Generate full predictions CSV
    out_dir = Path("reports/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "incumbent_predictions.csv"
    df_out.to_csv(full_path, index=False)
    print(f"\nFull predictions: {full_path} ({len(df_out)} games)")

    # Generate holdout-only CSV
    is_hold = df_out["season"] == HOLDOUT_SEASON
    hold_path = out_dir / "incumbent_predictions_2025_holdout.csv"
    df_out[is_hold].to_csv(hold_path, index=False)
    hold_count = is_hold.sum()
    print(f"Holdout predictions: {hold_path} ({hold_count} games)")

    # Holdout metrics
    hold_y = df_out.loc[is_hold, "home_win_actual"].values
    hold_prob = df_out.loc[is_hold, "incumbent_home_win_prob"].values
    valid = ~np.isnan(hold_y)
    if valid.any():
        metrics = compute_classification_metrics(hold_y[valid], hold_prob[valid])
        print("\nHoldout metrics:")
        print(f"  Log loss: {metrics['log_loss']:.4f}")
        print(f"  Brier:    {metrics['brier_score']:.4f}")
        print(f"  AUC:      {metrics['roc_auc']:.4f}")
        print(f"  Acc:      {metrics['accuracy']:.4f}")

    # Prediction cards markdown
    cards_path = out_dir / "incumbent_prediction_cards.md"
    _write_prediction_cards(df_out[is_hold], cards_path)
    print(f"Prediction cards: {cards_path}")

    print("\n=== Incumbent metadata ===")
    print(f"  Model:                 {INCUMBENT_VERSION}")
    print(f"  Feature set:           {INCUMBENT_FEATURE_SET}")
    print(f"  Calibration:           {INCUMBENT_CALIBRATION}")
    print(f"  Validation LL:         {INCUMBENT_VAL_LL}")
    print(f"  Holdout LL:            {INCUMBENT_HOLDOUT_LL}")
    print(f"  Report:                {INCUMBENT_REPORT}")
    print(f"  Registry:              {INCUMBENT_REGISTRY}")

    return {
        "full": str(full_path),
        "holdout": str(hold_path),
        "cards": str(cards_path),
    }


def _write_prediction_cards(df_hold: pd.DataFrame, path: Path) -> None:
    with open(path, "w") as f:
        f.write("# Incumbent Prediction Cards — 2025 Holdout\n\n")
        f.write(f"*Generated by `predict-incumbent` ({INCUMBENT_VERSION}, {INCUMBENT_DATE})*\n\n")
        f.write("| Game | Away | Home | Prob | Winner | Actual | Bucket | QB Chg | Mkt-Model |\n")
        f.write("|------|------|------|------|--------|--------|--------|--------|-----------|\n")
        cards = []
        for _, row in df_hold.iterrows():
            corr = (row["home_win_actual"] == 1 and row["incumbent_home_win_prob"] >= 0.5) or (
                row["home_win_actual"] == 0 and row["incumbent_home_win_prob"] < 0.5
            )
            mark = " ✅ " if corr else " ❌ "
            cards.append(
                f"| {row['season']} W{row['week']} "
                f"({row['gameday']}) "
                f"| {row['away_team']} "
                f"| {row['home_team']} "
                f"| {row['incumbent_home_win_prob']:.3f} "
                f"| {row['predicted_winner']} "
                f"| {'H' if row['home_win_actual'] == 1 else 'A'}"
                f"| {mark}"
                f"| {row.get('confidence_bucket', '')} "
                f"| {'⚠' if row.get('caution_qb_change', 0) else ''} "
                f"| {row.get('market_model_diff', '')} |\n"
            )
        f.writelines(cards)
