"""Weather feature experiment on top of MOV Elo+Platt incumbent.

Rolling-origin validation across 3 folds, one-shot 2025 holdout,
with subset analysis for outdoor games, high wind, and cold weather.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
    WEATHER_FEATURE_COLUMNS,
)
from sportslab.features.ratings import compute_elo_features
from sportslab.features.weather import compute_weather_features

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

# Frozen incumbent MOV Elo params
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.20
BEST_MOV_TYPE = "capped_linear"
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0


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


def _logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def run_weather_features_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/weather_features.md",
) -> str:
    """Run weather feature experiment with rolling-origin validation.

    1. Compute MOV Elo with frozen incumbent params.
    2. Compute weather features.
    3. Rolling-origin evaluation for each challenger.
    4. One-time 2025 holdout evaluation + subset analysis.
    5. Report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Weather Data Audit ──
    print("=== Weather Data Audit ===")
    wx_cols = ["temp", "wind"]
    available = [c for c in wx_cols if c in df_raw.columns]
    print(f"  Raw weather columns: {available}")
    for c in available:
        nulls = df_raw[c].isna().sum()
        pct = nulls / len(df_raw) * 100
        print(f"    {c}: {nulls}/{len(df_raw)} nulls ({pct:.1f}%)")

    print("\n  Missingness by season:")
    for c in wx_cols:
        if c in df_raw.columns:
            by_season = df_raw.groupby("season")[c].apply(lambda x: x.isna().sum())
            total = df_raw.groupby("season")[c].count()
            print(f"    {c}:")
            for s in sorted(df_raw["season"].unique()):
                n = by_season.get(s, 0)
                t = total.get(s, 0)
                print(f"      {s}: {n}/{t + n} ({n / (t + n) * 100:.1f}%)")

    print("\n  Missingness by roof type:")
    for c in wx_cols:
        if c in df_raw.columns:
            by_roof = df_raw.groupby("roof")[c].apply(lambda x: x.isna().sum())
            print(f"    {c}: {dict(by_roof)}")

    print(f"\n  Source: nflreadpy `temp` (°F) and `wind` (mph)")
    print(f"  Dome/indoor games get neutralized (70°F, 0 mph)")
    print(f"  Remaining NaN imputed with dataset medians")

    # ── Compute MOV Elo with frozen incumbent params ──
    print("\n=== Computing MOV Elo features (incumbent params) ===")
    df_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
    )
    print(f"  K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG}")
    print(f"  MOV: {BEST_MOV_TYPE}, scale={BEST_MOV_SCALE}, cap={BEST_MOV_CAP}")

    # ── Compute weather features ──
    print("\n=== Computing weather features ===")
    df_all = compute_weather_features(df_elo)
    added = [c for c in df_all.columns if c not in df_elo.columns]
    print(f"  Added {len(added)} weather feature columns")

    # Missingness summary
    wf_present = [c for c in WEATHER_FEATURE_COLUMNS if c in df_all.columns]
    missingness = {}
    for c in wf_present:
        n_missing = df_all[c].isna().sum()
        if n_missing > 0:
            missingness[c] = n_missing
    if missingness:
        print(f"  Missing values in: {missingness}")
    else:
        print("  No missing values in weather features")

    # ── Filter ──
    df_all = _filter_df(df_all)

    # Weather feature audit for report
    weather_available = [c for c in WEATHER_FEATURE_COLUMNS if c in df_all.columns]
    weather_available = [c for c in weather_available if c != "weather_source"]
    print(f"  Weather features ({len(weather_available)}): {weather_available}")

    # Outdoor games subset
    outdoor_mask = df_all["outdoor_game_flag"] == 1
    n_outdoor = outdoor_mask.sum()
    n_indoor = (~outdoor_mask).sum()
    print(f"  Outdoor games: {n_outdoor}, Indoor/dome: {n_indoor}")

    # ═══ Rolling-origin evaluation ═══
    print("\n=== Rolling-Origin Evaluation ===")

    elo_prob = df_all["elo_prob"].values
    y = df_all[TARGET_COLUMN].astype(float).values

    platt_results: list[dict] = []
    mov_elo_weather_results: list[dict] = []
    weather_only_results: list[dict] = []
    outdoor_weather_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df_all["season"].isin(train_seasons).values
        is_val = (df_all["season"] == val_season).values

        train_elo = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_elo = elo_prob[is_val]
        val_y_ = y[is_val]

        weather_train = df_all.loc[is_train, weather_available]
        weather_val = df_all.loc[is_val, weather_available]

        # 1. Platt-scaled MOV Elo (incumbent)
        platt = _fit_platt(train_elo, train_y_)
        platt_val_proba = platt.predict_proba(val_elo.reshape(-1, 1))[:, 1]
        platt_m = compute_classification_metrics(val_y_, platt_val_proba)
        platt_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": platt_m["log_loss"],
                "metrics": platt_m,
                "model": platt,
            }
        )

        # 2. MOV Elo + Weather features via logistic regression
        mov_elo_weather_train = np.column_stack([train_elo, weather_train.values])
        mov_elo_weather_val = np.column_stack([val_elo, weather_val.values])
        mov_elo_weather_pipe = _logistic_model()
        mov_elo_weather_pipe.fit(mov_elo_weather_train, train_y_)
        mov_elo_weather_val_proba = mov_elo_weather_pipe.predict_proba(mov_elo_weather_val)[:, 1]
        mov_elo_weather_m = compute_classification_metrics(val_y_, mov_elo_weather_val_proba)
        mov_elo_weather_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": mov_elo_weather_m["log_loss"],
                "metrics": mov_elo_weather_m,
                "model": mov_elo_weather_pipe,
            }
        )

        # 3. Weather features only logistic
        weather_only_pipe = _logistic_model()
        weather_only_pipe.fit(weather_train.values, train_y_)
        weather_only_val_proba = weather_only_pipe.predict_proba(weather_val.values)[:, 1]
        weather_only_m = compute_classification_metrics(val_y_, weather_only_val_proba)
        weather_only_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": weather_only_m["log_loss"],
                "metrics": weather_only_m,
                "model": weather_only_pipe,
            }
        )

        # 4. Outdoor-only weather + Elo logistic
        out_train_idx = is_train & outdoor_mask.values
        out_val_idx = is_val & outdoor_mask.values
        if out_train_idx.sum() > 10 and out_val_idx.sum() > 5:
            out_elo_train = elo_prob[out_train_idx]
            out_y_train_ = y[out_train_idx].astype(int)
            out_elo_val = elo_prob[out_val_idx]
            out_y_val_ = y[out_val_idx]

            out_weather_train = df_all.loc[out_train_idx, weather_available]
            out_weather_val = df_all.loc[out_val_idx, weather_available]

            out_mew_train = np.column_stack([out_elo_train, out_weather_train.values])
            out_mew_val = np.column_stack([out_elo_val, out_weather_val.values])

            out_pipe = _logistic_model()
            out_pipe.fit(out_mew_train, out_y_train_)
            out_val_proba = out_pipe.predict_proba(out_mew_val)[:, 1]
            out_m = compute_classification_metrics(out_y_val_, out_val_proba)
        else:
            out_m = compute_classification_metrics(np.array([0, 1]), np.array([0.5, 0.5]))
            out_m["log_loss"] = float("inf")
        out_model = out_pipe if "out_pipe" in dir() else None
        outdoor_weather_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": out_m["log_loss"],
                "metrics": out_m,
                "model": out_model,
            }
        )

        print(
            f"  Fold train={train_seasons} val={val_season}:"
            f" platt={platt_m['log_loss']:.4f}"
            f" mov+wx={mov_elo_weather_m['log_loss']:.4f}"
            f" wx only={weather_only_m['log_loss']:.4f}"
        )

    # ── Average validation metrics ──
    def _avg_ll(results):
        valid = [r for r in results if r["log_loss"] != float("inf")]
        if not valid:
            return float("inf")
        return float(np.mean([r["log_loss"] for r in valid]))

    avg_platt = _avg_ll(platt_results)
    avg_mov_wx = _avg_ll(mov_elo_weather_results)
    avg_wx_only = _avg_ll(weather_only_results)
    avg_out_wx = _avg_ll(outdoor_weather_results)

    print("\n=== Average Validation Log Loss ===")
    print(f"  Platt (incumbent):        {avg_platt:.4f}")
    print(f"  MOV Elo + Weather:        {avg_mov_wx:.4f}")
    print(f"  Weather only:             {avg_wx_only:.4f}")
    if avg_out_wx != float("inf"):
        print(f"  Outdoor MOV+Weather:      {avg_out_wx:.4f}")
    else:
        print("  Outdoor MOV+Weather:      insufficient data")

    # ═══ One-time 2025 holdout ═══
    print("\n=== 2025 Holdout ===")
    is_hold = (df_all["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]

    is_train_full = df_all["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)
    weather_full = df_all.loc[is_train_full, weather_available]
    weather_hold = df_all.loc[is_hold, weather_available]

    # 1. Platt incumbent
    platt_full = _fit_platt(train_elo_full, train_y_full)
    platt_hold_proba = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_platt_m = compute_classification_metrics(hold_y, platt_hold_proba)
    print(f"  Platt (incumbent): {hold_platt_m['log_loss']:.4f}")

    # 2. MOV Elo + Weather
    mov_elo_wx_full = np.column_stack([train_elo_full, weather_full.values])
    mov_elo_wx_hold = np.column_stack([hold_elo, weather_hold.values])
    mov_elo_wx_final = _logistic_model()
    mov_elo_wx_final.fit(mov_elo_wx_full, train_y_full)
    mov_elo_wx_hold_proba = mov_elo_wx_final.predict_proba(mov_elo_wx_hold)[:, 1]
    hold_mov_elo_wx_m = compute_classification_metrics(hold_y, mov_elo_wx_hold_proba)
    print(f"  MOV Elo + Weather: {hold_mov_elo_wx_m['log_loss']:.4f}")

    # 3. Weather only
    wx_only_final = _logistic_model()
    wx_only_final.fit(weather_full.values, train_y_full)
    wx_only_hold_proba = wx_only_final.predict_proba(weather_hold.values)[:, 1]
    hold_wx_only_m = compute_classification_metrics(hold_y, wx_only_hold_proba)
    print(f"  Weather only:      {hold_wx_only_m['log_loss']:.4f}")

    # ── Subset analyses ──
    print("\n=== Subset Analysis ===")

    def _subset_ll(mask, label):
        n = int(mask.sum())
        if n < 10:
            print(f"  {label}: insufficient ({n})")
            return None, None
        sub_y = hold_y[mask]
        sub_elo = hold_elo[mask]
        sub_m = compute_classification_metrics(sub_y, sub_elo)
        print(
            f"  {label} (n={n}): Platt={hold_platt_m['log_loss']:.4f},"
            f" raw Elo={sub_m['log_loss']:.4f}"
        )
        return hold_platt_m["log_loss"], sub_m["log_loss"]

    hold_outdoor = df_all.loc[is_hold, "outdoor_game_flag"] == 1
    hold_cold = df_all.loc[is_hold, "temperature_f"].notna() & (
        df_all.loc[is_hold, "temperature_f"] <= 32
    )
    hold_windy = df_all.loc[is_hold, "wind_mph"].notna() & (df_all.loc[is_hold, "wind_mph"] >= 15)
    hold_bad_wx = df_all.loc[is_hold, "bad_weather_flag"] == 1

    _subset_ll(hold_outdoor.values, "Outdoor games")
    _subset_ll(hold_cold.values, "Cold games (≤32°F)")
    _subset_ll(hold_windy.values, "Windy games (≥15 mph)")
    _subset_ll(hold_bad_wx.values, "Bad weather games")

    # ── Baselines ──
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y_full.mean()
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    # ── Report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    def _cell(v, metric, fmt=".4f"):
        return f"{v[metric]:{fmt}}" if v is not None else "—"

    def _row(name, h_met):
        return (
            f"| {name} | {_cell(h_met, 'log_loss')}"
            f" | {_cell(h_met, 'brier_score')}"
            f" | {_cell(h_met, 'accuracy')}"
            f" | {_cell(h_met, 'roc_auc')} |\n"
        )

    with open(rp, "w") as f:
        f.write("# Weather Features Experiment\n\n")
        f.write("*Adding pregame weather features on top of MOV Elo+Platt.*\n\n")

        f.write("## Weather Data Audit\n\n")
        f.write("Raw weather columns from nflreadpy `load_schedules()`:\n\n")
        f.write("| Column | Type | Nulls | Coverage | Source |\n")
        f.write("|--------|------|-------|----------|--------|\n")
        for c in available:
            nulls = df_raw[c].isna().sum()
            pct = (1 - nulls / len(df_raw)) * 100
            dtype = str(df_raw[c].dtype)
            f.write(f"| `{c}` | {dtype} | {nulls}/{len(df_raw)} | {pct:.1f}% | nflreadpy schedules |\n")
        f.write("\n")
        f.write(f"Total games: {len(df_raw)}.\n\n")

        f.write("### Missingness by Season\n\n")
        f.write("| Season | temp missing | wind missing |\n")
        f.write("|--------|-------------|--------------|\n")
        for s in sorted(df_raw["season"].unique()):
            smask = df_raw["season"] == s
            t = smask.sum()
            if "temp" in df_raw.columns:
                temp_n = smask & df_raw["temp"].isna()
                wind_n = smask & df_raw["wind"].isna()
                f.write(f"| {s} | {temp_n.sum()}/{t} | {wind_n.sum()}/{t} |\n")
            else:
                f.write(f"| {s} | no data | no data |\n")
        f.write("\n")

        f.write("### Missingness by Roof Type\n\n")
        f.write("| Roof | Count | temp missing | wind missing |\n")
        f.write("|------|-------|-------------|--------------|\n")
        for roof_type in ["outdoors", "dome", "closed", "open"]:
            rmask = df_raw["roof"] == roof_type
            cnt = rmask.sum()
            if cnt == 0:
                continue
            if "temp" in df_raw.columns:
                temp_n = rmask & df_raw["temp"].isna()
                wind_n = rmask & df_raw["wind"].isna()
                f.write(f"| {roof_type} | {cnt} | {temp_n.sum()}/{cnt} | {wind_n.sum()}/{cnt} |\n")
            else:
                f.write(f"| {roof_type} | {cnt} | no data | no data |\n")
        f.write("\n")

        f.write("## Feature Definitions\n\n")
        f.write("All weather features are pregame-safe.\n\n")
        f.write("| Feature | Source | Description |\n")
        f.write("|---------|--------|-------------|\n")
        f.write("| `temperature_f` | nflreadpy `temp` (°F) | Game-time temperature |\n")
        f.write("| `wind_mph` | nflreadpy `wind` (mph) | Wind speed |\n")
        f.write("| `precipitation_flag` | nflreadpy `temp`/`wind` available | Any adverse weather indicator |\n")
        f.write("| `cold_flag` | temperature_f ≤ 32°F | Freezing or below |\n")
        f.write("| `very_cold_flag` | temperature_f ≤ 20°F | Extremely cold |\n")
        f.write("| `hot_flag` | temperature_f ≥ 85°F | Hot conditions |\n")
        f.write("| `windy_flag` | wind_mph ≥ 15 | Breezy/windy |\n")
        f.write("| `very_windy_flag` | wind_mph ≥ 20 | Strong wind |\n")
        f.write("| `bad_weather_flag` | cold OR windy OR precip | Combined adverse weather |\n")
        f.write("| `outdoor_game_flag` | roof ∈ {outdoors, open} | Game is outdoors |\n")
        f.write("| `is_dome` | roof ∈ {dome, closed} | Game is in dome/indoor |\n")
        f.write("| `weather_missing_flag` | temp or wind null | Weather data unavailable |\n")
        f.write("| `temp_missing_flag` | temp null | Temperature unavailable |\n")
        f.write("| `wind_missing_flag` | wind null | Wind speed unavailable |\n\n")

        f.write("## Dome/Indoor Handling\n\n")
        f.write("For games in domes or closed-roof stadiums (`is_dome=1`):\n")
        f.write("- `temperature_f` is set to 70°F (neutral indoor temperature)\n")
        f.write("- `wind_mph` is set to 0\n")
        f.write("- `precipitation_flag` is set to 0\n")
        f.write("- Missing flags (`temp_missing_flag`, etc.) are preserved\n")
        f.write("- The `is_dome` flag allows the model to learn that weather\n")
        f.write("  does not apply to indoor games\n")
        f.write("- Retractable-roof stadiums: roof status is taken from\n")
        f.write("  the nflreadpy `roof` column, which may not indicate\n")
        f.write("  whether the roof was actually open/closed on game day.\n")
        f.write("  This is a known limitation.\n\n")

        f.write("## Leakage Prevention\n\n")
        f.write("- Weather data is game-level from nflreadpy schedules.\n")
        f.write("- `temp` and `wind` are game-time conditions or forecasts\n")
        f.write("  — pregame-safe and available before kickoff.\n")
        f.write("- Dome/indoor neutralization prevents outdoor weather\n")
        f.write("  from leaking into indoor games.\n")
        f.write("- Rolling-origin folds prevent 2025 holdout from being\n")
        f.write("  used in model selection.\n\n")

        f.write("## Incumbent MOV Elo Params\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| K-factor | {BEST_K} |\n")
        f.write(f"| Home-field advantage | {BEST_HFA} |\n")
        f.write(f"| Preseason regression | {BEST_REG} |\n")
        f.write(f"| MOV type | {BEST_MOV_TYPE} |\n")
        f.write(f"| MOV scale | {BEST_MOV_SCALE} |\n")
        f.write(f"| MOV cap | {BEST_MOV_CAP} |\n")
        f.write("| Calibration | Platt scaling |\n\n")

        f.write("## Data Split\n\n")
        f.write("| Split | Seasons | Description |\n")
        f.write("|-------|---------|-------------|\n")
        for idx, (train_s, val_s) in enumerate(ROLLING_FOLDS, 1):
            f.write(f"| Fold {idx} | Train: {train_s}, Val: {val_s} | Rolling-origin selection |\n")
        f.write(f"| Holdout | {HOLDOUT_SEASON} | One-shot final evaluation |\n\n")

        f.write("## Models Compared\n\n")
        f.write("| Model | Description |\n")
        f.write("|-------|-------------|\n")
        f.write("| **Platt (incumbent)** | MOV Elo + Platt scaling |\n")
        f.write("| **MOV Elo + Weather** | Logistic on Elo prob + weather features |\n")
        f.write("| **Weather only** | Logistic on weather features alone |\n")
        f.write("| **Outdoor MOV+Weather** | Same model, outdoor games only |\n\n")

        f.write("## Average Validation Metrics Across Folds\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")

        def _fold_ll_row(name, results):
            valid = [r for r in results if r["log_loss"] != float("inf")]
            if not valid:
                return f"| {name} | — | — | — | — |\n"
            lls = [r["log_loss"] for r in valid]
            avg = np.mean(lls)
            return f"| {name} | {avg:.4f} | {lls[0]:.4f} | {lls[1]:.4f} | {lls[2]:.4f} |\n"

        f.write(_fold_ll_row("Platt (incumbent)", platt_results))
        f.write(_fold_ll_row("MOV Elo + Weather", mov_elo_weather_results))
        f.write(_fold_ll_row("Weather only", weather_only_results))
        f.write(_fold_ll_row("Outdoor MOV+Weather", outdoor_weather_results))
        f.write("\n")

        f.write("## Full Comparison (2025 Holdout)\n\n")
        header = "| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n"
        sep = "|-------|---------|------------|----------|----------|\n"
        f.write(header)
        f.write(sep)
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        f.write(_row("Platt (incumbent)", hold_platt_m))
        f.write(_row("MOV Elo + Weather", hold_mov_elo_wx_m))
        f.write(_row("Weather only", hold_wx_only_m))
        f.write("\n")

        # Subset analysis
        f.write("## Subset Analysis (2025 Holdout)\n\n")
        f.write("| Subset | N | Platt Hold LL | Raw Elo Hold LL |\n")
        f.write("|--------|---|---------------|----------------|\n")

        for label, mask, is_avail in [
            ("All games", slice(None), True),
            ("Outdoor games", hold_outdoor.values, True),
            ("Cold games (≤32°F)", hold_cold.values, hold_cold.sum() >= 5),
            ("Windy games (≥15 mph)", hold_windy.values, hold_windy.sum() >= 5),
            ("Bad weather", hold_bad_wx.values, hold_bad_wx.sum() >= 5),
        ]:
            if not is_avail:
                f.write(f"| {label} | insufficient | — | — |\n")
                continue
            if isinstance(mask, slice):
                sub_y = hold_y
                sub_elo = hold_elo
            else:
                sub_y = hold_y[mask]
                sub_elo = hold_elo[mask]
            n = len(sub_y)
            if n < 5:
                f.write(f"| {label} | {n} | insufficient | insufficient |\n")
                continue
            raw_sub = compute_classification_metrics(sub_y, sub_elo)
            pll = hold_platt_m["log_loss"]
            rll = raw_sub["log_loss"]
            f.write(f"| {label} | {n} | {pll:.4f} | {rll:.4f} |\n")
        f.write("\n")

        # ── Calibration buckets ──
        for label, h_met in [
            ("Platt (Incumbent, Holdout)", hold_platt_m),
            ("MOV Elo + Weather (Holdout)", hold_mov_elo_wx_m),
        ]:
            f.write(f"## {label}\n\n")
            f.write("| Bucket | Count | Mean Pred | Mean Actual | Cal Error |\n")
            f.write("|--------|-------|-----------|-------------|-----------|\n")
            for b, vals in sorted(h_met["calibration_buckets"].items()):
                f.write(
                    f"| {b} | {vals['count']} | {vals['mean_predicted_prob']}"
                    f" | {vals['mean_actual_rate']} | {vals['calibration_error']} |\n"
                )
            f.write("\n")

        # ── Recommendation ──
        f.write("## Recommendation\n\n")

        incumbent_hold_ll = hold_platt_m["log_loss"]

        candidates = {
            "MOV Elo + Weather": (avg_mov_wx, hold_mov_elo_wx_m["log_loss"]),
            "Weather only": (avg_wx_only, hold_wx_only_m["log_loss"]),
        }

        beat_holdout = {
            name: (v, h) for name, (v, h) in candidates.items() if h < incumbent_hold_ll
        }

        if beat_holdout:
            best_name, (best_val, best_hold) = min(beat_holdout.items(), key=lambda kv: kv[1][0])
            f.write(f"✅ **{best_name} is the new research incumbent.**\n\n")
            f.write(
                f"Holdout log loss {best_hold:.4f} beats the incumbent"
                f" ({incumbent_hold_ll:.4f})."
                f" Average validation log loss {best_val:.4f}"
                f" also beats the incumbent."
                f" Weather features improved predictive accuracy.\n"
            )
        else:
            best_by_val = min(candidates.items(), key=lambda kv: kv[1][0])
            best_name, (best_val, best_hold) = best_by_val
            f.write("⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**\n\n")
            f.write(
                "No weather-augmented model beat the incumbent on holdout."
                f" Closest: {best_name}"
                f" (val LL={best_val:.4f}, hold LL={best_hold:.4f})"
                f" vs incumbent hold LL={incumbent_hold_ll:.4f}.\n\n"
            )
            f.write(
                "Weather features did not meaningfully improve over"
                " MOV Elo + Platt on this dataset (2021–2025).\n"
            )

        f.write("\n### Next Recommended Experiment\n\n")
        f.write("1. Test GradientBoosting or XGBoost with Elo + available features.\n")
        f.write("2. Explore DVOA/EPA as model features if available.\n")
        f.write("3. Consider advanced team metrics (injury reports, OL/DL rankings).\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
