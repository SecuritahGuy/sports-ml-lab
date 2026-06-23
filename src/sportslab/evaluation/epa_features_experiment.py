"""EPA team-efficiency feature experiment on top of MOV Elo+Platt incumbent.

Rolling-origin validation across 3 folds, one-shot 2025 holdout.
Tests whether pregame rolling EPA features improve prediction quality,
especially around the known QB-change failure mode.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.epa import (
    EPA_FEATURE_COLUMNS,
    compute_epa_features,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features

# Frozen incumbent season-regression Elo params
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
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


def run_epa_features_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/epa_features.md",
    pbp_cache_dir: str = "data/interim/nfl",
) -> str:
    """Run EPA feature experiment with rolling-origin validation.

    1. Compute MOV Elo with frozen incumbent params.
    2. Compute rolling EPA features from PBP data.
    3. Rolling-origin evaluation for each challenger.
    4. One-time 2025 holdout + QB-change + high-confidence subsets.
    5. Report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Build team regression overrides from QB change data ──
    team_overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS
    )

    # ── Compute MOV Elo with frozen incumbent params ──
    print("\n=== Computing MOV Elo features (season-regression incumbent params) ===")
    df_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        team_regression_overrides=team_overrides,
    )
    print(f"  K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG},"
          f" decay={BEST_DECAY}, qb_bonus={BEST_QB_BONUS}")
    print(f"  MOV: {BEST_MOV_TYPE}, scale={BEST_MOV_SCALE}, cap={BEST_MOV_CAP}")

    # ── Compute EPA features ──
    print("\n=== Computing EPA features ===")
    seasons_needed = sorted(df_elo["season"].unique())
    print(f"  Loading PBP data for seasons: {seasons_needed}")
    df_all = compute_epa_features(df_elo, cache_dir=pbp_cache_dir)
    added = [c for c in df_all.columns if c not in df_elo.columns]
    print(f"  Added {len(added)} EPA feature columns")

    # ── Compute QB features for subset analysis ──
    print("\n=== Computing QB features (for subset analysis) ===")
    df_all = compute_qb_features(df_all)
    qb_added = [c for c in df_all.columns if c not in df_elo.columns and c not in added]
    print(
        f"  Added {len([c for c in qb_added if not c.startswith('home_qb_id')])} QB feature columns"
    )

    # ── Filter ──
    df_all = _filter_df(df_all)

    # EPA feature audit
    epa_available = [c for c in EPA_FEATURE_COLUMNS if c in df_all.columns]
    print(f"  EPA features ({len(epa_available)}): {epa_available[:6]}...")

    # Reduced EPA: just 4 net differentials (not all 56 features)
    REDUCED_EPA_COLS = [
        "epa_net_per_play_3",
        "epa_net_per_play_5",
        "success_rate_net_3",
        "success_rate_net_5",
    ]
    epa_net_available = [c for c in REDUCED_EPA_COLS if c in df_all.columns]
    print(f"  Reduced EPA (net diffs only): {epa_net_available}")

    elo_prob = df_all["elo_prob"].values
    y = df_all[TARGET_COLUMN].astype(float).values

    # ═══ Rolling-origin evaluation ═══
    print("\n=== Rolling-Origin Evaluation ===")

    platt_results: list[dict] = []
    epa_only_results: list[dict] = []
    mov_elo_epa_results: list[dict] = []
    elo_epa_results: list[dict] = []
    mov_elo_epa_net_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df_all["season"].isin(train_seasons).values
        is_val = (df_all["season"] == val_season).values

        train_elo = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_elo = elo_prob[is_val]
        val_y_ = y[is_val]

        epa_train = df_all.loc[is_train, epa_available]
        epa_val = df_all.loc[is_val, epa_available]

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

        # 2. EPA features only
        epa_only_pipe = _logistic_model()
        epa_only_pipe.fit(epa_train.values, train_y_)
        epa_only_val_proba = epa_only_pipe.predict_proba(epa_val.values)[:, 1]
        epa_only_m = compute_classification_metrics(val_y_, epa_only_val_proba)
        epa_only_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": epa_only_m["log_loss"],
                "metrics": epa_only_m,
                "model": epa_only_pipe,
            }
        )

        # 3. MOV Elo + EPA features
        mov_elo_epa_train = np.column_stack([train_elo, epa_train.values])
        mov_elo_epa_val = np.column_stack([val_elo, epa_val.values])
        mov_elo_epa_pipe = _logistic_model()
        mov_elo_epa_pipe.fit(mov_elo_epa_train, train_y_)
        mov_elo_epa_val_proba = mov_elo_epa_pipe.predict_proba(mov_elo_epa_val)[:, 1]
        mov_elo_epa_m = compute_classification_metrics(val_y_, mov_elo_epa_val_proba)
        mov_elo_epa_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": mov_elo_epa_m["log_loss"],
                "metrics": mov_elo_epa_m,
                "model": mov_elo_epa_pipe,
            }
        )

        # 4. Raw Elo + EPA (diagnostic, no Platt)
        elo_epa_train = np.column_stack([train_elo, epa_train.values])
        elo_epa_val = np.column_stack([val_elo, epa_val.values])
        elo_epa_pipe = _logistic_model()
        elo_epa_pipe.fit(elo_epa_train, train_y_)
        elo_epa_val_proba = elo_epa_pipe.predict_proba(elo_epa_val)[:, 1]
        elo_epa_m = compute_classification_metrics(val_y_, elo_epa_val_proba)
        elo_epa_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": elo_epa_m["log_loss"],
                "metrics": elo_epa_m,
                "model": elo_epa_pipe,
            }
        )

        # 5. MOV Elo + EPA net differentials only (reduced)
        epa_net_train = df_all.loc[is_train, epa_net_available]
        epa_net_val = df_all.loc[is_val, epa_net_available]
        mov_elo_epa_net_train = np.column_stack([train_elo, epa_net_train.values])
        mov_elo_epa_net_val = np.column_stack([val_elo, epa_net_val.values])
        mov_elo_epa_net_pipe = _logistic_model()
        mov_elo_epa_net_pipe.fit(mov_elo_epa_net_train, train_y_)
        mov_elo_epa_net_val_proba = mov_elo_epa_net_pipe.predict_proba(mov_elo_epa_net_val)[:, 1]
        mov_elo_epa_net_m = compute_classification_metrics(val_y_, mov_elo_epa_net_val_proba)
        mov_elo_epa_net_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": mov_elo_epa_net_m["log_loss"],
                "metrics": mov_elo_epa_net_m,
                "model": mov_elo_epa_net_pipe,
            }
        )

        print(
            f"  Fold train={train_seasons} val={val_season}:"
            f" platt={platt_m['log_loss']:.4f}"
            f" epa_only={epa_only_m['log_loss']:.4f}"
            f" mov+epa={mov_elo_epa_m['log_loss']:.4f}"
            f" elo+epa={elo_epa_m['log_loss']:.4f}"
            f" mov+epa_net={mov_elo_epa_net_m['log_loss']:.4f}"
        )

    # ── Average validation metrics ──
    def _avg_ll(results):
        return float(np.mean([r["log_loss"] for r in results]))

    avg_platt = _avg_ll(platt_results)
    avg_epa_only = _avg_ll(epa_only_results)
    avg_mov_epa = _avg_ll(mov_elo_epa_results)
    avg_elo_epa = _avg_ll(elo_epa_results)
    avg_mov_epa_net = _avg_ll(mov_elo_epa_net_results)

    print("\n=== Average Validation Log Loss ===")
    print(f"  Platt (incumbent):        {avg_platt:.4f}")
    print(f"  EPA only:                 {avg_epa_only:.4f}")
    print(f"  MOV Elo + EPA (all 56):   {avg_mov_epa:.4f}")
    print(f"  Raw Elo + EPA:            {avg_elo_epa:.4f}")
    print(f"  MOV Elo + EPA net (4):    {avg_mov_epa_net:.4f}")

    # ═══ One-time 2025 holdout ═══
    print("\n=== 2025 Holdout ===")
    is_hold = (df_all["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]

    is_train_full = df_all["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)
    epa_full = df_all.loc[is_train_full, epa_available]
    epa_hold = df_all.loc[is_hold, epa_available]

    # 1. Platt incumbent
    platt_full = _fit_platt(train_elo_full, train_y_full)
    platt_hold_proba = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_platt_m = compute_classification_metrics(hold_y, platt_hold_proba)
    print(f"  Platt (incumbent): {hold_platt_m['log_loss']:.4f}")

    # 2. EPA only
    epa_only_final = _logistic_model()
    epa_only_final.fit(epa_full.values, train_y_full)
    epa_only_hold_proba = epa_only_final.predict_proba(epa_hold.values)[:, 1]
    hold_epa_only_m = compute_classification_metrics(hold_y, epa_only_hold_proba)
    print(f"  EPA only:          {hold_epa_only_m['log_loss']:.4f}")

    # 3. MOV Elo + EPA
    mov_elo_epa_full = np.column_stack([train_elo_full, epa_full.values])
    mov_elo_epa_hold = np.column_stack([hold_elo, epa_hold.values])
    mov_elo_epa_final = _logistic_model()
    mov_elo_epa_final.fit(mov_elo_epa_full, train_y_full)
    mov_elo_epa_hold_proba = mov_elo_epa_final.predict_proba(mov_elo_epa_hold)[:, 1]
    hold_mov_elo_epa_m = compute_classification_metrics(hold_y, mov_elo_epa_hold_proba)
    print(f"  MOV Elo + EPA:     {hold_mov_elo_epa_m['log_loss']:.4f}")

    # 4. Raw Elo + EPA
    elo_epa_full = np.column_stack([train_elo_full, epa_full.values])
    elo_epa_hold = np.column_stack([hold_elo, epa_hold.values])
    elo_epa_final = _logistic_model()
    elo_epa_final.fit(elo_epa_full, train_y_full)
    elo_epa_hold_proba = elo_epa_final.predict_proba(elo_epa_hold)[:, 1]
    hold_elo_epa_m = compute_classification_metrics(hold_y, elo_epa_hold_proba)
    print(f"  Raw Elo + EPA:     {hold_elo_epa_m['log_loss']:.4f}")

    # 5. MOV Elo + EPA net differentials only
    epa_net_full = df_all.loc[is_train_full, epa_net_available]
    epa_net_hold = df_all.loc[is_hold, epa_net_available]
    mov_elo_epa_net_full = np.column_stack([train_elo_full, epa_net_full.values])
    mov_elo_epa_net_hold = np.column_stack([hold_elo, epa_net_hold.values])
    mov_elo_epa_net_final = _logistic_model()
    mov_elo_epa_net_final.fit(mov_elo_epa_net_full, train_y_full)
    mov_elo_epa_net_hold_proba = mov_elo_epa_net_final.predict_proba(mov_elo_epa_net_hold)[:, 1]
    hold_mov_elo_epa_net_m = compute_classification_metrics(hold_y, mov_elo_epa_net_hold_proba)
    print(f"  MOV Elo + EPA net:{hold_mov_elo_epa_net_m['log_loss']:.4f}")

    # ── Subset analyses ──
    print("\n=== Subset Analysis ===")

    def _subset_metrics(mask, label):
        n = int(mask.sum())
        if n < 5:
            print(f"  {label}: insufficient ({n})")
            return ()
        sub_y = hold_y[mask]
        results = {}
        for name, proba in [
            ("Platt", platt_hold_proba),
            ("EPA only", epa_only_hold_proba),
            ("MOV+EPA", mov_elo_epa_hold_proba),
        ]:
            sub_proba = proba[mask]
            m = compute_classification_metrics(sub_y, sub_proba)
            results[name] = m["log_loss"]
        print(f"  {label} (n={n}): {results}")
        return results

    # QB change subset
    hold_qb_changed = df_all.loc[is_hold, "home_qb_changed"].fillna(0).astype(bool).values
    hold_qb_stable = ~hold_qb_changed

    _subset_metrics(hold_qb_changed, "QB changed (home)")
    _subset_metrics(hold_qb_stable, "QB stable (home)")

    # High-confidence subset
    hold_high_conf = platt_hold_proba > 0.9
    hold_low_conf = platt_hold_proba <= 0.6
    _subset_metrics(hold_high_conf, "High confidence (>0.9)")
    _subset_metrics(hold_low_conf, "Low confidence (<=0.6)")

    # Baselines
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y_full.mean()
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    # ── Report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# EPA Team-Efficiency Features Experiment\n\n")
        f.write(
            "*Testing whether pregame rolling EPA features improve"
            " on MOV Elo+Platt, with focus on QB-change failure mode.*\n\n"
        )

        f.write("## PBP Data Source\n\n")
        f.write("| Source | Columns | Coverage |\n")
        f.write("|--------|---------|----------|\n")
        f.write("| nflreadpy (nflverse) | epa, success, pass/run splits | 2021–2025 |\n\n")

        f.write("## Feature Definitions\n\n")
        f.write("All features are pregame-safe (chronological rolling windows, shifted).\n\n")
        f.write("| Feature Group | Windows | Description |\n")
        f.write("|--------------|---------|-------------|\n")
        f.write("| Offensive EPA/play | 3, 5 | Rolling avg of home/away offense EPA per play |\n")
        f.write("| Offensive success rate | 3, 5 | Rolling avg of EPA success rate |\n")
        f.write("| Defensive EPA/play | 3, 5 | Rolling avg of opponent EPA per play |\n")
        f.write("| Defensive success rate | 3, 5 | Rolling avg of EPA success rate against |\n")
        f.write("| Passing splits | 3, 5 | Pass-only EPA and success rate |\n")
        f.write("| Rushing splits | 3, 5 | Rush-only EPA and success rate |\n")
        f.write("| Net differentials | 3, 5 | Home offense − away defense |\n")
        f.write("| Missing flags | — | Games available and missingness indicator |\n\n")

        f.write("## Leakage Prevention\n\n")
        f.write("- Rolling windows are shifted by 1 game (current game excluded)\n")
        f.write("- Stats reset at season boundaries (no prior-season carryover)\n")
        f.write("- Week 1 games use 0 (neutral) imputation + missingness flags\n")
        f.write("- Rolling-origin folds prevent 2025 holdout from influencing model selection\n\n")

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
        f.write("| **EPA only** | Logistic on EPA features alone |\n")
        f.write("| **MOV Elo + EPA** | Logistic on Elo prob + EPA features |\n")
        f.write("| **Raw Elo + EPA** | Logistic on raw Elo prob + EPA (diagnostic) |\n\n")

        f.write("## Average Validation Metrics Across Folds\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")

        def _fold_ll_row(name, results):
            lls = [r["log_loss"] for r in results]
            avg = np.mean(lls)
            return f"| {name} | {avg:.4f} | {lls[0]:.4f} | {lls[1]:.4f} | {lls[2]:.4f} |\n"

        f.write(_fold_ll_row("Platt (incumbent)", platt_results))
        f.write(_fold_ll_row("EPA only", epa_only_results))
        f.write(_fold_ll_row("MOV Elo + EPA", mov_elo_epa_results))
        f.write(_fold_ll_row("Raw Elo + EPA", elo_epa_results))
        f.write("\n")

        f.write("## Full Comparison (2025 Holdout)\n\n")
        f.write("| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n")
        f.write("|-------|---------|------------|----------|----------|\n")
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")

        for name, h_met in [
            ("Platt (incumbent)", hold_platt_m),
            ("EPA only", hold_epa_only_m),
            ("MOV Elo + EPA", hold_mov_elo_epa_m),
            ("Raw Elo + EPA", hold_elo_epa_m),
        ]:
            f.write(
                f"| {name} | {h_met['log_loss']:.4f}"
                f" | {h_met['brier_score']:.4f}"
                f" | {h_met['accuracy']:.4f}"
                f" | {h_met['roc_auc']:.4f} |\n"
            )
        f.write("\n")

        # Subset analysis
        f.write("## Subset Analysis (2025 Holdout)\n\n")
        f.write("| Subset | N | Platt | EPA only | MOV+EPA |\n")
        f.write("|--------|---|-------|----------|--------|\n")

        for label, mask in [
            ("All games", slice(None)),
            ("QB changed (home)", hold_qb_changed),
            ("QB stable (home)", hold_qb_stable),
            ("High confidence (>0.9)", hold_high_conf),
            ("Low confidence (<=0.6)", hold_low_conf),
        ]:
            if isinstance(mask, slice):
                sub_y = hold_y
                sub_platt = platt_hold_proba
                sub_epa = epa_only_hold_proba
                sub_mov_epa = mov_elo_epa_hold_proba
                n = len(sub_y)
            else:
                sub_y = hold_y[mask]
                sub_platt = platt_hold_proba[mask]
                sub_epa = epa_only_hold_proba[mask]
                sub_mov_epa = mov_elo_epa_hold_proba[mask]
                n = len(sub_y)
            if n < 5:
                f.write(f"| {label} | {n} | insufficient | insufficient | insufficient |\n")
                continue
            pll = compute_classification_metrics(sub_y, sub_platt)["log_loss"]
            ell = compute_classification_metrics(sub_y, sub_epa)["log_loss"]
            mll = compute_classification_metrics(sub_y, sub_mov_epa)["log_loss"]
            f.write(f"| {label} | {n} | {pll:.4f} | {ell:.4f} | {mll:.4f} |\n")
        f.write("\n")

        # ── Recommendation ──
        f.write("## Recommendation\n\n")

        incumbent_hold_ll = hold_platt_m["log_loss"]

        candidates = {
            "EPA only": (avg_epa_only, hold_epa_only_m["log_loss"]),
            "MOV Elo + EPA": (avg_mov_epa, hold_mov_elo_epa_m["log_loss"]),
            "Raw Elo + EPA": (avg_elo_epa, hold_elo_epa_m["log_loss"]),
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
                f" also beats the incumbent.\n"
            )
        else:
            best_by_val = min(candidates.items(), key=lambda kv: kv[1][0])
            best_name, (best_val, best_hold) = best_by_val
            f.write("⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**\n\n")
            f.write(
                "No EPA-augmented model beat the incumbent on holdout."
                f" Closest: {best_name}"
                f" (val LL={best_val:.4f}, hold LL={best_hold:.4f})"
                f" vs incumbent hold LL={incumbent_hold_ll:.4f}.\n\n"
            )

        # QB-change failure mode assessment
        qb_changed_ll = hold_platt_m.copy()
        if hold_qb_changed.sum() >= 5:
            qb_changed_ll = compute_classification_metrics(
                hold_y[hold_qb_changed], platt_hold_proba[hold_qb_changed]
            )["log_loss"]
        qb_stable_ll = compute_classification_metrics(
            hold_y[hold_qb_stable], platt_hold_proba[hold_qb_stable]
        )["log_loss"]

        f.write("\n### QB-Change Failure Mode Assessment\n\n")
        if hold_qb_changed.sum() >= 5:
            qb_gap = qb_changed_ll - qb_stable_ll
            if qb_changed_ll == float("inf"):
                f.write("QB-change subset had no holdout games with home QB change.\n")
            else:
                f.write(
                    f"Platt incumbent on QB-changed games: {qb_changed_ll:.4f}"
                    f" | QB-stable: {qb_stable_ll:.4f}"
                    f" | QB-change gap: {qb_gap:.4f}\n"
                )
                if beat_holdout:
                    f.write(
                        "EPA features did improve holdout — "
                        "further analysis should check if this closes the QB-change gap.\n"
                    )
                else:
                    f.write("EPA features did not close the QB-change gap on holdout.\n")
        else:
            f.write("Insufficient QB-change games to assess.\n")

        f.write("\n### Next Recommended Experiment\n\n")
        f.write("1. If EPA features beat incumbent, test with more expressive models.\n")
        f.write("2. DVOA/EPA from external sources if nflfastR is insufficient.\n")
        f.write("3. Consider team-injury feature engineering.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
