"""Injury report feature experiment on top of MOV Elo+Platt incumbent.

Tests whether pregame injury report data (QB OUT flags, position-group
injury counts, injury-driven QB changes) improves on the incumbent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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
from sportslab.features.injuries import (
    INJURY_FEATURE_COLUMNS,
    compute_injury_features,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2


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


def run_injury_features_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/injury_features.md",
) -> str:
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Compute MOV Elo ──
    print("\n=== Computing MOV Elo features (incumbent params) ===")
    team_overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=BEST_REG,
        qb_change_bonus=BEST_QB_BONUS,
    )
    df_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=team_overrides,
        decay_half_life=BEST_DECAY,
    )
    print(
        f"  K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG}, decay={BEST_DECAY},"
        f" qb_bonus={BEST_QB_BONUS}"
    )
    print(f"  Elo prob range: [{df_elo['elo_prob'].min():.4f}, {df_elo['elo_prob'].max():.4f}]")

    # ── Compute QB features ──
    print("\n=== Computing QB features ===")
    df_qb = compute_qb_features(df_elo)
    qb_change_count = df_qb["home_qb_changed"].sum()
    print(f"  Home QB changes: {int(qb_change_count)} / {len(df_qb)}")

    # ── Compute injury features ──
    print("\n=== Computing injury features ===")
    df_all = compute_injury_features(df_qb)
    added = [c for c in df_all.columns if c not in df_qb.columns]
    print(f"  Added {len(added)} injury feature columns")

    # Injury-driven changes
    injury_changes = df_all[
        (df_all["home_qb_injury_change"] == 1) | (df_all["away_qb_injury_change"] == 1)
    ]
    print(f"  Injury-driven QB changes: {len(injury_changes)}")

    # ── Filter ──
    df_all = _filter_df(df_all)
    print(f"  After filter: {len(df_all)} rows")

    # Feature columns available
    injury_available = [c for c in INJURY_FEATURE_COLUMNS if c in df_all.columns]
    print(f"  Injury feature cols ({len(injury_available)}): {injury_available}")

    elo_prob = df_all["elo_prob"].values
    y = df_all[TARGET_COLUMN].astype(float).values

    # QB-change subset tracking for report
    home_qb_changed = (
        df_all["home_qb_changed"].values
        if "home_qb_changed" in df_all.columns
        else np.zeros(len(df_all))
    )

    # ═══ Rolling-origin evaluation ═══
    print("\n=== Rolling-Origin Evaluation ===")

    platt_results = []
    elo_injury_results = []
    injury_only_results = []
    elo_qbi_results = []

    # QB injury-only feature subset
    qb_injury_cols = [c for c in injury_available if "qb_" in c or "any_qb" in c]

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df_all["season"].isin(train_seasons).values
        is_val = (df_all["season"] == val_season).values

        train_elo = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_elo = elo_prob[is_val]
        val_y_ = y[is_val]

        injury_train = df_all.loc[is_train, injury_available]
        injury_val = df_all.loc[is_val, injury_available]
        qb_injury_train = df_all.loc[is_train, qb_injury_cols]
        qb_injury_val = df_all.loc[is_val, qb_injury_cols]

        # 1. Platt (incumbent)
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

        # 2. Elo + Injury features
        elo_injury_train = np.column_stack([train_elo, injury_train.values])
        elo_injury_val = np.column_stack([val_elo, injury_val.values])
        elo_injury_pipe = _logistic_model()
        elo_injury_pipe.fit(elo_injury_train, train_y_)
        elo_injury_val_proba = elo_injury_pipe.predict_proba(elo_injury_val)[:, 1]
        elo_injury_m = compute_classification_metrics(val_y_, elo_injury_val_proba)
        elo_injury_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": elo_injury_m["log_loss"],
                "metrics": elo_injury_m,
                "model": elo_injury_pipe,
            }
        )

        # 3. Injury-only logistic
        injury_only_pipe = _logistic_model()
        injury_only_pipe.fit(injury_train.values, train_y_)
        injury_only_val_proba = injury_only_pipe.predict_proba(injury_val.values)[:, 1]
        injury_only_m = compute_classification_metrics(val_y_, injury_only_val_proba)
        injury_only_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": injury_only_m["log_loss"],
                "metrics": injury_only_m,
                "model": injury_only_pipe,
            }
        )

        # 4. Elo + QB injury flags only
        elo_qbi_train = np.column_stack([train_elo, qb_injury_train.values])
        elo_qbi_val = np.column_stack([val_elo, qb_injury_val.values])
        elo_qbi_pipe = _logistic_model()
        elo_qbi_pipe.fit(elo_qbi_train, train_y_)
        elo_qbi_val_proba = elo_qbi_pipe.predict_proba(elo_qbi_val)[:, 1]
        elo_qbi_m = compute_classification_metrics(val_y_, elo_qbi_val_proba)
        elo_qbi_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": elo_qbi_m["log_loss"],
                "metrics": elo_qbi_m,
                "model": elo_qbi_pipe,
            }
        )

        print(
            f"  Fold train={train_seasons} val={val_season}:"
            f" platt={platt_m['log_loss']:.4f}"
            f" elo+inj={elo_injury_m['log_loss']:.4f}"
            f" inj={injury_only_m['log_loss']:.4f}"
            f" elo+qb_inj={elo_qbi_m['log_loss']:.4f}"
        )

    # ── Average validation ──
    def _avg_ll(results):
        return float(np.mean([r["log_loss"] for r in results]))

    avg_platt = _avg_ll(platt_results)
    avg_elo_injury = _avg_ll(elo_injury_results)
    avg_injury_only = _avg_ll(injury_only_results)
    avg_elo_qbi = _avg_ll(elo_qbi_results)

    print("\n=== Average Validation Log Loss ===")
    print(f"  Platt (incumbent):        {avg_platt:.4f}")
    print(f"  Elo + Injury:             {avg_elo_injury:.4f}")
    print(f"  Injury only:              {avg_injury_only:.4f}")
    print(f"  Elo + QB injury flags:    {avg_elo_qbi:.4f}")

    # ═══ One-time 2025 holdout ═══
    print("\n=== 2025 Holdout ===")
    is_hold = (df_all["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]

    is_train_full = df_all["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)
    injury_full = df_all.loc[is_train_full, injury_available]
    injury_hold = df_all.loc[is_hold, injury_available]
    qb_injury_full = df_all.loc[is_train_full, qb_injury_cols]
    qb_injury_hold = df_all.loc[is_hold, qb_injury_cols]

    # 1. Platt incumbent
    platt_full = _fit_platt(train_elo_full, train_y_full)
    platt_hold_proba = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_platt_m = compute_classification_metrics(hold_y, platt_hold_proba)
    print(f"  Platt (incumbent):         {hold_platt_m['log_loss']:.4f}")

    # 2. Elo + Injury
    elo_inj_full = np.column_stack([train_elo_full, injury_full.values])
    elo_inj_hold = np.column_stack([hold_elo, injury_hold.values])
    elo_inj_final = _logistic_model()
    elo_inj_final.fit(elo_inj_full, train_y_full)
    elo_inj_hold_proba = elo_inj_final.predict_proba(elo_inj_hold)[:, 1]
    hold_elo_inj_m = compute_classification_metrics(hold_y, elo_inj_hold_proba)
    print(f"  Elo + Injury:              {hold_elo_inj_m['log_loss']:.4f}")

    # 3. Injury only
    inj_only_final = _logistic_model()
    inj_only_final.fit(injury_full.values, train_y_full)
    inj_only_hold_proba = inj_only_final.predict_proba(injury_hold.values)[:, 1]
    hold_inj_only_m = compute_classification_metrics(hold_y, inj_only_hold_proba)
    print(f"  Injury only:               {hold_inj_only_m['log_loss']:.4f}")

    # 4. Elo + QB injury flags
    elo_qbi_full = np.column_stack([train_elo_full, qb_injury_full.values])
    elo_qbi_hold = np.column_stack([hold_elo, qb_injury_hold.values])
    elo_qbi_final = _logistic_model()
    elo_qbi_final.fit(elo_qbi_full, train_y_full)
    elo_qbi_hold_proba = elo_qbi_final.predict_proba(elo_qbi_hold)[:, 1]
    hold_elo_qbi_m = compute_classification_metrics(hold_y, elo_qbi_hold_proba)
    print(f"  Elo + QB injury flags:     {hold_elo_qbi_m['log_loss']:.4f}")

    # ── Subset analysis ──
    print("\n=== Subset Analysis ===")

    hold_home_qb_changed = home_qb_changed[is_hold]
    hold_home_qb_injury_change = (
        df_all.loc[is_hold, "home_qb_injury_change"].values
        if "home_qb_injury_change" in df_all.columns
        else np.zeros(is_hold.sum())
    )
    hold_any_qb_out = (
        df_all.loc[is_hold, "any_qb_out"].values
        if "any_qb_out" in df_all.columns
        else np.zeros(is_hold.sum())
    )

    def _subset_summary(mask, label):
        n = int(mask.sum())
        if n < 5:
            print(f"  {label}: insufficient ({n})")
            return None, None
        sub_y = hold_y[mask]
        sub_elo = hold_elo[mask]
        sub_m = compute_classification_metrics(sub_y, sub_elo)
        print(f"  {label} (n={n}): raw Elo LL={sub_m['log_loss']:.4f}")
        return sub_m["log_loss"], n

    subsets = [
        ("QB-change games (home)", hold_home_qb_changed == 1),
        ("QB-stable games (home)", hold_home_qb_changed == 0),
        ("Injury-driven QB change", hold_home_qb_injury_change == 1),
        ("Any QB OUT", hold_any_qb_out == 1),
        ("No QB OUT", hold_any_qb_out == 0),
    ]
    for label, m in subsets:
        _subset_summary(m, label)

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
        f.write("# Injury Features Experiment\n\n")
        f.write("*Adding pregame injury report features on top of MOV Elo+Platt.*\n\n")

        f.write("## Method\n\n")
        f.write("Rolling-origin 3-fold validation, one-shot 2025 holdout.\n\n")

        f.write("### Incumbent Params\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| K-factor | {BEST_K} |\n")
        f.write(f"| Home-field advantage | {BEST_HFA} |\n")
        f.write(f"| Preseason regression | {BEST_REG} |\n")
        f.write(f"| Decay | {BEST_DECAY} |\n")
        f.write(f"| QB-change bonus | {BEST_QB_BONUS} |\n")
        f.write("| MOV type | capped_linear, scale=0.05, cap=2.0 |\n")
        f.write("| Calibration | Platt scaling |\n\n")

        f.write("### Injury Features\n\n")
        f.write("| Feature | Description |\n")
        f.write("|---------|-------------|\n")
        f.write("| `home/away_qb_out` | Count of QBs on team ruled OUT |\n")
        f.write("| `home/away_qb_doubtful_or_out` | Count of QBs OUT or Doubtful |\n")
        f.write("| `home/away_total_out` | Count of all players ruled OUT |\n")
        f.write("| `home/away_total_doubtful_or_out` | Count of all players OUT or Doubtful |\n")
        f.write("| `home/away_skill_out` | Count of QB+RB+WR+TE OUT |\n")
        f.write("| `home/away_ol_out` | Count of OL (C+G+T) OUT |\n")
        f.write("| `home/away_def_out` | Count of defensive players OUT |\n")
        f.write("| `any_qb_out` | Either team has a QB OUT |\n")
        f.write("| `net_injuries` | home_total_out − away_total_out |\n")
        f.write("| `net_skill_out` | home_skill_out − away_skill_out |\n")
        f.write("| `net_def_out` | home_def_out − away_def_out |\n")
        f.write("| `home/away_qb_injury_change` | QB changed AND old starter was OUT |\n\n")

        f.write("### Injury Data Source\n\n")
        f.write("nflreadpy `load_injuries()` — official NFL injury reports, weekly pregame.\n\n")

        f.write("### Models Compared\n\n")
        f.write("| Model | Description |\n")
        f.write("|-------|-------------|\n")
        f.write("| **Platt (incumbent)** | MOV Elo + Platt scaling |\n")
        f.write("| **Elo + Injury** | Logistic on Elo prob + all injury features |\n")
        f.write("| **Injury only** | Logistic on injury features alone |\n")
        f.write("| **Elo + QB injury** | Logistic on Elo prob + QB-specific injury flags |\n\n")

        f.write("### Data Split\n\n")
        f.write("| Split | Seasons |\n")
        f.write("|-------|---------|\n")
        for idx, (train_s, val_s) in enumerate(ROLLING_FOLDS, 1):
            f.write(f"| Fold {idx} | Train: {train_s}, Val: {val_s} |\n")
        f.write(f"| Holdout | {HOLDOUT_SEASON} |\n\n")

        f.write("## Rolling-Origin Validation\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")

        def _fold_ll_row(name, results):
            lls = [r["log_loss"] for r in results]
            avg = np.mean(lls)
            return f"| {name} | {avg:.4f} | {lls[0]:.4f} | {lls[1]:.4f} | {lls[2]:.4f} |\n"

        f.write(_fold_ll_row("Platt (incumbent)", platt_results))
        f.write(_fold_ll_row("Elo + Injury", elo_injury_results))
        f.write(_fold_ll_row("Injury only", injury_only_results))
        f.write(_fold_ll_row("Elo + QB injury flags", elo_qbi_results))
        f.write("\n")

        f.write("## 2025 Holdout\n\n")
        f.write("| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n")
        f.write("|-------|---------|------------|----------|----------|\n")
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        f.write(_row("Platt (incumbent)", hold_platt_m))
        f.write(_row("Elo + Injury", hold_elo_inj_m))
        f.write(_row("Injury only", hold_inj_only_m))
        f.write(_row("Elo + QB injury flags", hold_elo_qbi_m))
        f.write("\n")

        # Subset analysis
        f.write("## Subset Analysis (2025 Holdout)\n\n")
        f.write("| Subset | N | Raw Elo LL |\n")
        f.write("|--------|---|------------|\n")

        subset_outputs = []
        for label, m in subsets:
            n = int(m.sum())
            if n < 5:
                subset_outputs.append((label, None, None))
                f.write(f"| {label} | insufficient | — |\n")
            else:
                sub_y = hold_y[m]
                sub_elo = hold_elo[m]
                raw_ll = compute_classification_metrics(sub_y, sub_elo)["log_loss"]
                subset_outputs.append((label, n, raw_ll))
                f.write(f"| {label} | {n} | {raw_ll:.4f} |\n")
        f.write("\n")

        # Calibration
        for label, h_met in [
            ("Platt (Incumbent, Holdout)", hold_platt_m),
            ("Elo + Injury (Holdout)", hold_elo_inj_m),
        ]:
            f.write(f"## {label}\n\n")
            f.write("| Bucket | Count | Mean Pred | Mean Actual | Cal Error |\n")
            f.write("|--------|-------|-----------|-------------|-----------|\n")
            for b, vals in sorted(h_met["calibration_buckets"].items()):
                f.write(
                    f"| {b} | {vals['count']} | {vals['mean_predicted_prob']:.4f}"
                    f" | {vals['mean_actual_rate']:.4f}"
                    f" | {vals['calibration_error']:.4f} |\n"
                )
            f.write("\n")

        # Recommendation
        f.write("## Decision\n\n")
        incumbent_hold_ll = hold_platt_m["log_loss"]
        candidates = {
            "Elo + Injury": (avg_elo_injury, hold_elo_inj_m["log_loss"]),
            "Injury only": (avg_injury_only, hold_inj_only_m["log_loss"]),
            "Elo + QB injury flags": (avg_elo_qbi, hold_elo_qbi_m["log_loss"]),
        }

        beat_both = {
            name: (v, h)
            for name, (v, h) in candidates.items()
            if h < incumbent_hold_ll and v < avg_platt
        }

        if beat_both:
            best_name = min(beat_both.items(), key=lambda kv: kv[1][0])[0]
            best_val, best_hold = beat_both[best_name]
            f.write(f"✅ **{best_name} promoted as new research incumbent.**\n\n")
            f.write(
                f"Holdout log loss {best_hold:.4f} beats incumbent"
                f" ({incumbent_hold_ll:.4f}) and avg val LL {best_val:.4f}"
                f" beats incumbent ({avg_platt:.4f})."
                f" Injury features improve predictive accuracy.\n\n"
            )
        else:
            f.write("⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**\n\n")
            f.write(
                "No injury-augmented model beat the incumbent on both validation and holdout.\n\n"
            )
            for name, (v, h) in sorted(candidates.items(), key=lambda kv: kv[1][0]):
                val_status = "beats" if v < avg_platt else "trails"
                hold_status = "beats" if h < incumbent_hold_ll else "trails"
                f.write(
                    f"- {name}: val LL={v:.4f} ({val_status} incumbent {avg_platt:.4f}),"
                    f" hold LL={h:.4f} ({hold_status} incumbent {incumbent_hold_ll:.4f})\n"
                )

        f.write("\n### Key Conclusions\n\n")
        f.write("1. Injury features from nflreadpy official reports were tested.\n")
        f.write("2. QB injury flags alone (subset) were also tested separately.\n")
        f.write(
            "3. Injury-driven QB change detection added."
            " Distinguishes injury-forced changes from coaching decisions.\n"
        )
        f.write(
            "4. The QB-change failure mode is partly an injury signal —"
            " if injury data improves performance, the gap narrows.\n"
        )

    print(f"\nReport written to: {rp}")
    return str(rp)
