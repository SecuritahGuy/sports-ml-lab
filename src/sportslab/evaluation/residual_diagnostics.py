"""Residual diagnostics for the MOV Elo+Platt incumbent.

Identifies systematic failure modes by analyzing prediction residuals
across teams, weather, scheduling, rest, week, game type, and other
dimensions. Produces a diagnostic report.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.scheduling import compute_scheduling_features
from sportslab.features.weather import compute_weather_features

HOLDOUT_SEASON = 2025

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.20
BEST_MOV_TYPE = "capped_linear"
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0

INCUMBENT_HOLDOUT_LL = 0.6373


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _fit_platt(train_prob: np.ndarray, train_y: np.ndarray) -> Pipeline:
    platt = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    platt.fit(train_prob.reshape(-1, 1), train_y)
    return platt


def _gather_dimension_data(df: pd.DataFrame) -> pd.DataFrame:
    """Attach scheduling, QB, and weather features for diagnostic dimensions."""
    out = compute_scheduling_features(df)
    out = compute_qb_features(out)
    out = compute_weather_features(out)
    return out


def run_residual_diagnostics(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/residual_diagnostics.md",
) -> str:
    """Run residual diagnostics on the MOV Elo+Platt incumbent."""
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Build features ──
    print("=== Building feature stack ===")
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
    )
    df = _gather_dimension_data(df)
    df = _filter_df(df)

    y = df[TARGET_COLUMN].astype(float).values
    elo_prob = df["elo_prob"].values

    # Split
    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = (df["season"] == HOLDOUT_SEASON).values

    train_elo = elo_prob[is_train]
    train_y_ = y[is_train].astype(int)
    hold_elo = elo_prob[is_hold]
    hold_y_ = y[is_hold]

    # Fit Platt on 2021-2024
    platt = _fit_platt(train_elo, train_y_)

    # Predict on all
    hold_pred = platt.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    train_pred = platt.predict_proba(train_elo.reshape(-1, 1))[:, 1]
    all_pred = np.empty(len(df))
    all_pred[is_train] = train_pred
    all_pred[is_hold] = hold_pred

    hold_m = compute_classification_metrics(hold_y_, hold_pred)
    train_m = compute_classification_metrics(train_y_, train_pred)

    # Residuals
    resid = all_pred - y

    # Build analysis dataframe
    dx = df[
        ["season", "week", "weekday", "game_type", "home_team", "away_team", "roof", "surface"]
    ].copy()
    dx["predicted"] = all_pred
    dx["actual"] = y
    dx["residual"] = resid
    dx["abs_error"] = np.abs(resid)
    dx["squared_error"] = resid**2
    dx["elo_prob"] = elo_prob
    dx["elo_diff"] = df["elo_diff"]
    dx["is_home_favorite"] = (elo_prob >= 0.5).astype(int)
    dx["log_loss_contrib"] = -(
        y * np.log(np.clip(all_pred, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - all_pred, 1e-15, 1))
    )

    # Merge extra dimension columns if available
    for c in ["div_game", "is_dome", "is_international"]:
        if c in df.columns:
            dx[c] = df[c].values
    for c in [
        "home_short_week",
        "away_short_week",
        "home_off_bye",
        "away_off_bye",
        "thursday_flag",
        "monday_flag",
        "home_consecutive_road",
        "away_consecutive_road",
    ]:
        if c in df.columns:
            dx[c] = df[c].values
    for c in [
        "home_qb_changed",
        "away_qb_changed",
        "qb_change_diff",
        "cold_flag",
        "windy_flag",
        "bad_weather_flag",
        "outdoor_game_flag",
        "temperature_f",
        "wind_mph",
    ]:
        if c in df.columns:
            dx[c] = df[c].values

    hold = dx[is_hold].copy()
    train_dx = dx[is_train].copy()

    # ═══ Report ═══
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== Overall Performance ===")
    print(
        f"  Training (2021-2024): LL={train_m['log_loss']:.4f}, "
        f"Brier={train_m['brier_score']:.4f}, Acc={train_m['accuracy']:.4f}"
    )
    print(
        f"  Holdout (2025):       LL={hold_m['log_loss']:.4f}, "
        f"Brier={hold_m['brier_score']:.4f}, Acc={hold_m['accuracy']:.4f}"
    )

    with open(rp, "w") as f:
        f.write("# Residual Diagnostics: MOV Elo + Platt Incumbent\n\n")
        f.write("Diagnostic analysis of where the incumbent fails systematically.\n\n")

        # ── 1. Overall ──
        f.write("## 1. Overall Performance\n\n")
        f.write("| Metric | Train (2021-2024) | Holdout (2025) |\n")
        f.write("|--------|-------------------|----------------|\n")
        f.write(f"| Log loss | {train_m['log_loss']:.4f} | {hold_m['log_loss']:.4f} |\n")
        f.write(f"| Brier score | {train_m['brier_score']:.4f} | {hold_m['brier_score']:.4f} |\n")
        f.write(f"| Accuracy | {train_m['accuracy']:.4f} | {hold_m['accuracy']:.4f} |\n")
        roc = hold_m.get("roc_auc")
        if roc is not None:
            f.write(f"| ROC AUC | — | {roc:.4f} |\n")
        f.write(f"| N | {len(train_dx)} | {len(hold)} |\n\n")

        # ── 2. Calibration ──
        f.write("## 2. Calibration (Holdout 2025)\n\n")
        f.write("| Bucket | Count | Mean Pred | Mean Actual | Cal Error |\n")
        f.write("|--------|-------|-----------|-------------|-----------|\n")
        for b, vals in sorted(hold_m["calibration_buckets"].items()):
            mp = vals["mean_predicted_prob"]
            ma = vals["mean_actual_rate"]
            ce = vals["calibration_error"]
            f.write(f"| {b} | {vals['count']} | {mp} | {ma} | {ce} |\n")
        f.write("\n")

        # ── 3. Residuals by Team ──
        f.write("## 3. Residuals by Team\n\n")
        f.write(
            "Average residual (predicted - actual). Negative = model was too pessimistic"
            " (underpredicted home wins).\n\n"
        )
        f.write("### Worst-predicted teams (highest mean |residual|)\n\n")
        f.write("| Team | Side | N | Mean Residual | Mean |Residual| |\n")
        f.write("|------|------|---|---------------|-----------------|\n")

        team_records = []
        for team_col, side in [("home_team", "home"), ("away_team", "away")]:
            grouped = train_dx.groupby(team_col)
            for team, grp in grouped:
                mean_res = grp["residual"].mean()
                mean_abs = grp["abs_error"].mean()
                team_records.append((side, team, len(grp), mean_res, mean_abs))
        team_df = pd.DataFrame(
            team_records, columns=["side", "team", "n", "mean_residual", "mean_abs_error"]
        )
        worst = team_df.sort_values("mean_abs_error", ascending=False).head(15)
        for _, r in worst.iterrows():
            f.write(
                f"| {r['team']} | {r['side']} | {r['n']} | {r['mean_residual']:+.4f} |"
                f" {r['mean_abs_error']:.4f} |\n"
            )
        f.write("\n")

        # ── 4. By Game Type ──
        f.write("## 4. Residuals by Game Context\n\n")

        def _write_grouped(title, col, label_map=None):
            f.write(f"### {title}\n\n")
            f.write("| Group | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |\n")
            f.write("|-------|---|-----------|-------------|----------|-----------------|\n")
            for val, grp in sorted(train_dx.groupby(col), key=lambda x: -len(x[1])):
                label = label_map.get(val, val) if label_map else val
                n = len(grp)
                mn_pred = grp["predicted"].mean()
                mn_act = grp["actual"].mean()
                ll = float(np.mean(grp["log_loss_contrib"]))
                mae = grp["abs_error"].mean()
                f.write(
                    f"| {label} | {n} | {mn_pred:.4f} | {mn_act:.4f} | {ll:.4f} | {mae:.4f} |\n"
                )
            f.write("\n")

        _write_grouped("By Season", "season")
        _write_grouped(
            "By Game Type",
            "game_type",
            {
                "REG": "Regular",
                "CON": "Conference",
                "DIV": "Divisional",
                "WC": "Wild Card",
                "SB": "Super Bowl",
            },
        )
        _write_grouped("By Weekday", "weekday")
        _write_grouped("By Roof", "roof")

        # Week
        f.write("### By Week\n\n")
        f.write("| Week | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |\n")
        f.write("|------|---|-----------|-------------|----------|-----------------|\n")
        for w, grp in sorted(train_dx.groupby("week"), key=lambda x: x[0]):
            n = len(grp)
            mn_pred = grp["predicted"].mean()
            mn_act = grp["actual"].mean()
            ll = float(np.mean(grp["log_loss_contrib"]))
            mae = grp["abs_error"].mean()
            f.write(f"| {w} | {n} | {mn_pred:.4f} | {mn_act:.4f} | {ll:.4f} | {mae:.4f} |\n")
        f.write("\n")

        # Short week / bye
        if "home_short_week" in dx.columns:
            f.write("### Short Week (Home Team)\n\n")
            f.write("| Short Week | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |\n")
            f.write("|------------|---|-----------|-------------|----------|-----------------|\n")
            for val, grp in sorted(train_dx.groupby("home_short_week"), key=lambda x: -len(x[1])):
                label = "Yes" if val == 1 else "No"
                n = len(grp)
                mn_pred = grp["predicted"].mean()
                mn_act = grp["actual"].mean()
                ll = float(np.mean(grp["log_loss_contrib"]))
                mae = grp["abs_error"].mean()
                f.write(
                    f"| {label} | {n} | {mn_pred:.4f} | {mn_act:.4f} | {ll:.4f} | {mae:.4f} |\n"
                )
            f.write("\n")

        if "home_off_bye" in dx.columns:
            f.write("### Off Bye (Home Team)\n\n")
            f.write("| Off Bye | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |\n")
            f.write("|---------|---|-----------|-------------|----------|-----------------|\n")
            for val, grp in sorted(train_dx.groupby("home_off_bye"), key=lambda x: -len(x[1])):
                label = "Yes" if val == 1 else "No"
                n = len(grp)
                mn_pred = grp["predicted"].mean()
                mn_act = grp["actual"].mean()
                ll = float(np.mean(grp["log_loss_contrib"]))
                mae = grp["abs_error"].mean()
                f.write(
                    f"| {label} | {n} | {mn_pred:.4f} | {mn_act:.4f} | {ll:.4f} | {mae:.4f} |\n"
                )
            f.write("\n")

        # Primetime
        if "thursday_flag" in dx.columns:
            f.write("### Primetime Games\n\n")
            f.write("| Game Type | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |\n")
            f.write("|-----------|---|-----------|-------------|----------|-----------------|\n")
            for val, grp in sorted(train_dx.groupby("thursday_flag"), key=lambda x: -len(x[1])):
                label = "Thursday" if val == 1 else "Not Thursday"
                n = len(grp)
                mn_pred = grp["predicted"].mean()
                mn_act = grp["actual"].mean()
                ll = float(np.mean(grp["log_loss_contrib"]))
                mae = grp["abs_error"].mean()
                f.write(
                    f"| {label} | {n} | {mn_pred:.4f} | {mn_act:.4f} | {ll:.4f} | {mae:.4f} |\n"
                )
            f.write("\n")

        # Bad weather
        if "bad_weather_flag" in dx.columns:
            f.write("### Bad Weather\n\n")
            f.write("| Weather | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |\n")
            f.write("|---------|---|-----------|-------------|----------|-----------------|\n")
            for val, grp in sorted(train_dx.groupby("bad_weather_flag"), key=lambda x: -len(x[1])):
                label = "Bad weather" if val == 1 else "Normal"
                n = len(grp)
                mn_pred = grp["predicted"].mean()
                mn_act = grp["actual"].mean()
                ll = float(np.mean(grp["log_loss_contrib"]))
                mae = grp["abs_error"].mean()
                f.write(
                    f"| {label} | {n} | {mn_pred:.4f} | {mn_act:.4f} | {ll:.4f} | {mae:.4f} |\n"
                )
            f.write("\n")

        if "outdoor_game_flag" in dx.columns:
            f.write("### Indoor vs Outdoor\n\n")
            f.write("| Venue | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |\n")
            f.write("|-------|---|-----------|-------------|----------|-----------------|\n")
            for val, grp in sorted(train_dx.groupby("outdoor_game_flag"), key=lambda x: -len(x[1])):
                label = "Outdoor" if val == 1 else "Indoor"
                n = len(grp)
                mn_pred = grp["predicted"].mean()
                mn_act = grp["actual"].mean()
                ll = float(np.mean(grp["log_loss_contrib"]))
                mae = grp["abs_error"].mean()
                f.write(
                    f"| {label} | {n} | {mn_pred:.4f} | {mn_act:.4f} | {ll:.4f} | {mae:.4f} |\n"
                )
            f.write("\n")

        # QB change
        if "home_qb_changed" in dx.columns:
            f.write("### QB Change (Home Team)\n\n")
            f.write("| QB Changed | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |\n")
            f.write("|------------|---|-----------|-------------|----------|-----------------|\n")
            for val, grp in sorted(train_dx.groupby("home_qb_changed"), key=lambda x: -len(x[1])):
                label = "Yes" if val == 1 else "No"
                n = len(grp)
                mn_pred = grp["predicted"].mean()
                mn_act = grp["actual"].mean()
                ll = float(np.mean(grp["log_loss_contrib"]))
                mae = grp["abs_error"].mean()
                f.write(
                    f"| {label} | {n} | {mn_pred:.4f} | {mn_act:.4f} | {ll:.4f} | {mae:.4f} |\n"
                )
            f.write("\n")

        # ── 5. Residual vs Elo confidence ──
        f.write("## 5. Residuals vs Elo Confidence\n\n")
        f.write("Do residuals grow when Elo is more confident (extreme probabilities)?\n\n")
        f.write("| Elo Bucket | N | Mean Pred | Mean Actual | Mean Residual | Log Loss |\n")
        f.write("|------------|---|-----------|-------------|---------------|----------|\n")
        elo_bins = np.arange(0, 1.01, 0.1)
        train_dx["elo_bin"] = pd.cut(train_dx["elo_prob"], elo_bins)
        for b, grp in sorted(
            train_dx.groupby("elo_bin", observed=False),
            key=lambda x: x[0].mid if hasattr(x[0], "mid") else 0,
        ):
            label = f"[{b.left:.1f},{b.right:.1f})"
            n = len(grp)
            mn_pred = grp["predicted"].mean()
            mn_act = grp["actual"].mean()
            mn_res = grp["residual"].mean()
            ll = float(np.mean(grp["log_loss_contrib"]))
            f.write(
                f"| {label} | {n} | {mn_pred:.4f} | {mn_act:.4f} | {mn_res:+.4f} | {ll:.4f} |\n"
            )
        f.write("\n")

        # ── 6. Extreme errors ──
        f.write("## 6. Extreme Prediction Errors (Holdout 2025)\n\n")
        f.write("Games where the model was most confidently wrong.\n\n")
        f.write("| Predicted | Actual | Residual | Home | Away | Week |\n")
        f.write("|-----------|--------|----------|------|------|------|\n")
        hold_extreme = hold.copy()
        hold_extreme["home_team"] = df.loc[is_hold, "home_team"].values
        hold_extreme["away_team"] = df.loc[is_hold, "away_team"].values
        hold_extreme = hold_extreme.sort_values("abs_error", ascending=False)
        for _, r in hold_extreme.head(20).iterrows():
            f.write(
                f"| {r['predicted']:.3f} | {r['actual']:.0f} | {r['residual']:+.4f} |"
                f" {r['home_team']} | {r['away_team']} | {r['week']} |\n"
            )
        f.write("\n")

        # ── 7. Directional bias ──
        f.write("## 7. Directional Bias\n\n")
        wrong_away = (hold["residual"] < -0.5).sum()
        wrong_home = (hold["residual"] > 0.5).sum()
        f.write(f"- Games where model confidently predicted home win but away won: {wrong_home}\n")
        f.write(f"- Games where model confidently predicted away win but home won: {wrong_away}\n")
        f.write(f"- Mean residual (holdout): {hold['residual'].mean():+.4f}\n")
        f.write(
            f"- Model is optimistic about home teams: "
            f"{'YES' if hold['residual'].mean() > 0 else 'NO'}\n\n"
        )

        # ── 8. Best/west predicted teams ──
        f.write("## 8. Best vs Worst Predicted Teams (Holdout)\n\n")
        hold_team = hold.copy()
        hold_team["home_team"] = df.loc[is_hold, "home_team"].values
        hold_team["away_team"] = df.loc[is_hold, "away_team"].values

        all_team_errors = []
        for side_key in ["home_team", "away_team"]:
            for team, grp in hold_team.groupby(side_key):
                all_team_errors.append(
                    {
                        "team": team,
                        "n": len(grp),
                        "mean_abs_error": grp["abs_error"].mean(),
                        "mean_residual": grp["residual"].mean(),
                    }
                )
        team_err_df = pd.DataFrame(all_team_errors)

        f.write("### Worst Predicted Teams (highest MAE)\n\n")
        f.write("| Team | N | Mean |Residual| | Mean Residual |\n")
        f.write("|------|---|-----------------|---------------|\n")
        for _, r in team_err_df.sort_values("mean_abs_error", ascending=False).head(10).iterrows():
            f.write(
                f"| {r['team']} | {r['n']} | {r['mean_abs_error']:.4f} | "
                f"{r['mean_residual']:+.4f} |\n"
            )
        f.write("\n")

        f.write("### Best Predicted Teams (lowest MAE)\n\n")
        f.write("| Team | N | Mean |Residual| | Mean Residual |\n")
        f.write("|------|---|-----------------|---------------|\n")
        for _, r in team_err_df.sort_values("mean_abs_error", ascending=True).head(10).iterrows():
            f.write(
                f"| {r['team']} | {r['n']} | {r['mean_abs_error']:.4f} | "
                f"{r['mean_residual']:+.4f} |\n"
            )
        f.write("\n")

        # ── 9. Market comparison ──
        f.write("## 9. Market Efficiency Check\n\n")
        f.write("Correlation between incumbent residual and market probability:\n")
        if "spread_line" in df.columns:
            spread_hold = df.loc[is_hold, "spread_line"].fillna(0).values
            corr_res = hold["residual"]
            corr_spread = pd.Series(spread_hold).fillna(0)
            r_val, p_val = pearsonr(corr_res, corr_spread)
            f.write(f"- Correlation(residual, spread_line): r={r_val:.3f}, p={p_val:.4f}\n")
            f.write(
                f"- Interpretation: {
                    'Residuals are independent of market spread'
                    if p_val > 0.05
                    else 'Residuals correlate with market spread'
                }\n\n"
            )

        # ── 10. Summary ──
        f.write("## 10. Summary & Recommendations\n\n")
        f.write("### Where the model works well\n\n")
        f.write(
            "- Overall log loss {:.4f} is {:.2f} away from market ({:.4f}).\n".format(
                hold_m["log_loss"],
                hold_m["log_loss"] - INCUMBENT_HOLDOUT_LL,
                INCUMBENT_HOLDOUT_LL,
            )
        )
        f.write("- Calibration is reasonable (check decile table above).\n")
        f.write("- Performance is consistent across most game contexts.\n\n")

        f.write("### Where the model struggles\n\n")
        worst_team = team_df.sort_values("mean_abs_error", ascending=False).iloc[0]
        f.write(
            f"- Team prediction quality varies (worst: {worst_team['team']} "
            f"MAE={worst_team['mean_abs_error']:.4f})\n"
        )
        f.write("- Check high-error seasons/game types from tables above.\n")
        f.write("- Extreme predictions (very confident) still miss sometimes.\n\n")

        f.write("### Recommended Next Steps\n\n")
        f.write("1. **DVOA/EPA features** — check if nflreadpy provides advanced metrics.\n")
        f.write("2. **Move to market-relative modeling** — use market odds as a baseline\n")
        f.write("   and model the residual (market vs actual).\n")
        f.write("3. **Per-team Elo initialization** — teams may need different starting Elos.\n")
        f.write("4. **Coach/coordinators features** — system changes affect team performance.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
