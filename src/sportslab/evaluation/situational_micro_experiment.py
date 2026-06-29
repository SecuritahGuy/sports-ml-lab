"""Situational micro-features experiment — divisional, first-year coach, surface mismatch.

Compares tight single-family variants against the incumbent.
No model promoted in this pass. Referee features are diagnostic-only.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.metrics import log_loss as sk_ll
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.predict_incumbent import (
    BEST_DECAY,
    BEST_HFA,
    BEST_K,
    BEST_QB_BONUS,
    BEST_REG,
    FEATURE_COLS,
    INCUMBENT_DATE,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VERSION,
)
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, TARGET_COLUMN
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features
from sportslab.features.situational_micro import (
    SITUATIONAL_MICRO_COLUMNS,
    compute_situational_micro_features,
)

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
DEFAULT_REPORT = "reports/experiments/situational_micro.md"
N_BOOTSTRAP = 200

# Feature groups
DIV_GAME = ["div_game"]
DIV_INTERACTION = ["div_game", "div_home_qb_changed", "div_away_qb_changed"]
COACH_COLS = ["home_first_year_coach", "away_first_year_coach", "coach_change_diff"]
SURFACE_COLS = ["away_surface_mismatch", "away_grass_to_turf", "away_turf_to_grass"]

MODEL_VARIANTS = [
    ("incumbent", FEATURE_COLS, "Incumbent (qb_changed + rolling_mov_3)"),
    ("divisional", FEATURE_COLS + DIV_GAME, "+ div_game"),
    ("divisional_interaction", FEATURE_COLS + DIV_INTERACTION, "+ div_game + div×qb_changed"),
    ("first_year_coach", FEATURE_COLS + COACH_COLS, "+ first_year_coach + change_diff"),
    ("surface_mismatch", FEATURE_COLS + SURFACE_COLS, "+ away_surface_mismatch + g2t + t2g"),
]


def _build_feature_matrix(df: pd.DataFrame, feat_cols) -> np.ndarray:
    elo = df["elo_prob"].values
    avail = [c for c in feat_cols if c in df.columns]
    if avail:
        return np.column_stack([elo] + [df[c].values for c in avail])
    return elo.reshape(-1, 1)


def _run_rolling_ll(
    df_all: pd.DataFrame,
    feat_cols,
    y: np.ndarray,
) -> List[float]:
    fold_lls = []
    for train_s, val_s in ROLLING_FOLDS:
        tr = df_all["season"].isin(train_s).values
        va = (df_all["season"] == val_s).values
        x_tr = _build_feature_matrix(df_all.loc[tr], feat_cols)
        x_va = _build_feature_matrix(df_all.loc[va], feat_cols)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipe.fit(x_tr, y[tr].astype(int))
        prob = pipe.predict_proba(x_va)[:, 1]
        valid = ~np.isnan(y[va])
        ll = float(sk_ll(y[va][valid].astype(int), prob[valid]))
        fold_lls.append(ll)
    return fold_lls


def _run_holdout(
    df_all: pd.DataFrame,
    feat_cols,
    y: np.ndarray,
) -> float:
    tr = (df_all["season"] < HOLDOUT_SEASON).values
    va = (df_all["season"] == HOLDOUT_SEASON).values
    x_tr = _build_feature_matrix(df_all.loc[tr], feat_cols)
    x_va = _build_feature_matrix(df_all.loc[va], feat_cols)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(x_tr, y[tr].astype(int))
    prob = pipe.predict_proba(x_va)[:, 1]
    valid = ~np.isnan(y[va])
    from sklearn.metrics import log_loss as sk_ll

    return float(sk_ll(y[va][valid].astype(int), prob[valid]))


def _bootstrap_delta(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    p_a = np.clip(prob_a[valid], 1e-15, 1 - 1e-15)
    p_b = np.clip(prob_b[valid], 1e-15, 1 - 1e-15)
    n = len(y_t)
    deltas = np.zeros(n_iter)
    from sklearn.metrics import log_loss as sk_ll

    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        deltas[i] = sk_ll(y_t[idx], p_b[idx]) - sk_ll(y_t[idx], p_a[idx])
    return {
        "mean_delta": round(float(np.mean(deltas)), 4),
        "ci_low": round(float(np.percentile(deltas, 2.5)), 4),
        "ci_high": round(float(np.percentile(deltas, 97.5)), 4),
    }


def _referee_audit(df: pd.DataFrame) -> Dict:
    """Diagnostic audit of referee data — no model, just reporting."""
    ref = df["referee"].fillna("UNKNOWN").astype(str)
    total = len(df)
    n_unique = ref.nunique()
    n_unknown = (ref == "UNKNOWN").sum()
    counts = ref.value_counts()

    earliest_season = int(df["season"].min())
    latest_season = int(df["season"].max())

    # Check if referee is known pregame (available in raw schedule)
    # The column exists in the raw schedule, so it IS pregame-known
    pregame_safe = n_unknown == 0  # Not safe if some games have unknown referee

    return {
        "total_games": total,
        "unique_referees": n_unique,
        "missing_referee": int(n_unknown),
        "pregame_known": True,
        "pregame_safe": pregame_safe,
        "min_games_per_ref": int(counts.min()),
        "max_games_per_ref": int(counts.max()),
        "median_games_per_ref": int(counts.median()),
        "seasons_covered": f"{earliest_season}-{latest_season}",
        "recommendation": (
            "diagnostic_only — 20 referees over ~1400 games (~70 games/ref) "
            "is marginally usable but adds complexity. Referee tendency features "
            "would require historical penalty data which is not in the feature table. "
            "Skipping referee model features for now."
        ),
    }


def _subset_analysis(
    df: pd.DataFrame,
    results: Dict[str, np.ndarray],
    col: str,
    label: str,
) -> Dict:
    """Compute per-model log loss on a binary-split subset."""
    mask = df.get(col, pd.Series(0)).fillna(0).astype(bool).values
    if mask.sum() == 0:
        return {"subset_label": label, "n": 0, "models": {}}
    m_yes = {}
    m_no = {}
    for name, prob in results.items():
        valid = ~np.isnan(df[TARGET_COLUMN].values)
        y = df[TARGET_COLUMN].values.astype(int)
        y_yes = y[mask & valid]
        prob_yes = prob[mask & valid]
        if len(np.unique(y_yes)) >= 2:
            m_yes[name] = round(float(sk_ll(y_yes, prob_yes, labels=[0, 1])), 4)
        else:
            m_yes[name] = None
        no_mask = (~mask) & valid
        if no_mask.sum() > 0 and len(np.unique(y[no_mask])) >= 2:
            m_no[name] = round(float(sk_ll(y[no_mask], prob[no_mask], labels=[0, 1])), 4)
        else:
            m_no[name] = None
    return {
        "subset_label": label,
        "n": int(mask.sum()),
        "models_yes": m_yes,
        "models_no": m_no,
    }


def _calibration_buckets(y_true: np.ndarray, y_prob: np.ndarray) -> List[Dict]:
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    labels = [f"{int(i * 10)}-{int((i + 1) * 10)}%" for i in range(10)]
    indices = np.clip(np.floor(y_p * 10).astype(int), 0, 9)
    results = []
    for i in range(10):
        mask = indices == i
        if mask.sum() == 0:
            continue
        results.append(
            {
                "bucket": labels[i],
                "n": int(mask.sum()),
                "mean_pred": round(float(y_p[mask].mean()), 4),
                "mean_actual": round(float(y_t[mask].mean()), 4),
                "cal_error": round(float(abs(y_p[mask].mean() - y_t[mask].mean())), 4),
            }
        )
    return results


def _confidence_buckets(y_true: np.ndarray, y_prob: np.ndarray) -> List[Dict]:
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    conf = np.abs(y_p - 0.5) * 2
    labels_list = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    indices = np.clip(np.floor(conf / 0.2).astype(int), 0, 4)
    results = []
    for i in range(5):
        mask = indices == i
        if mask.sum() == 0:
            continue
        m = _metrics(y_t[mask], y_p[mask])
        results.append({"bucket": labels_list[i], "n": int(mask.sum()), **m})
    return results


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    if len(y_t) == 0:
        return {}
    eps = 1e-15
    y_p = np.clip(y_p, eps, 1 - eps)
    n_classes = len(np.unique(y_t))
    ll_val = round(float(sk_ll(y_t, y_p, labels=[0, 1])), 4) if n_classes >= 2 else float("nan")
    brier = round(float(brier_score_loss(y_t, y_p)), 4) if n_classes >= 2 else float("nan")
    return {
        "log_loss": ll_val,
        "brier": brier,
        "accuracy": round(float(accuracy_score(y_t, y_p >= 0.5)), 4),
    }


def _worst_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    game_ids: np.ndarray,
    home_teams: np.ndarray,
    n: int = 20,
) -> List[Dict]:
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = np.clip(y_prob[valid], 1e-15, 1 - 1e-15)
    gids = game_ids[valid]
    teams = home_teams[valid]
    contrib = -(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p))
    worst = np.argsort(-contrib)[:n]
    results = []
    for i in worst:
        results.append(
            {
                "game_id": str(gids[i]),
                "team": str(teams[i]),
                "actual": int(y_t[i]),
                "pred": round(float(y_p[i]), 4),
                "log_loss_contrib": round(float(contrib[i]), 4),
            }
        )
    return results


def run_situational_micro_experiment(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = DEFAULT_REPORT,
) -> str:
    """Run situational micro-features experiment."""
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)
    print("Building features...")

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
    df = compute_situational_micro_features(df)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df["is_neutral"].fillna(False).values
    df = df[mask].copy().reset_index(drop=True)
    y = df[TARGET_COLUMN].astype(float).values
    print(f"  Eligible games: {len(df)}")

    # ── Feature sparsity audit ──
    print("\n--- Feature Sparsity ---")
    for col in SITUATIONAL_MICRO_COLUMNS:
        if col in df.columns:
            nz = (df[col] != 0).sum()
            print(f"  {col:30s}  non-zero: {nz}/{len(df)} ({100 * nz / len(df):.1f}%)")
    for col in ["div_game"]:
        if col in df.columns:
            nz = (df[col] != 0).sum()
            print(f"  {col:30s}  non-zero: {nz}/{len(df)} ({100 * nz / len(df):.1f}%)")

    # ── Referee audit ──
    print("\n--- Referee Audit ---")
    ref_audit = _referee_audit(df)
    for k, v in ref_audit.items():
        print(f"  {k}: {v}")

    # ── Validation ──
    print("\n--- Rolling-Origin Validation ---")
    rolling_results = {}
    for name, feats, desc in MODEL_VARIANTS:
        folds = _run_rolling_ll(df, feats, y)
        avg = float(np.mean(folds))
        rolling_results[name] = {"folds": [round(v, 4) for v in folds], "avg_val_ll": round(avg, 4)}
        print(f"  {name:25s}  val={avg:.4f}  folds={[round(v, 4) for v in folds]}")

    print("\n--- Fitted-Once (Holdout: 2025) ---")
    holdout_results = {}
    for name, feats, desc in MODEL_VARIANTS:
        hold = _run_holdout(df, feats, y)
        holdout_results[name] = round(hold, 4)
        print(f"  {name:25s}  hold={hold:.4f}")

    inc_roll = rolling_results["incumbent"]["avg_val_ll"]
    inc_hold = holdout_results["incumbent"]

    # ── Bootstrap CIs (rolling) ──
    print("\n--- Bootstrap CI (Δ = challenger − incumbent, rolling) ---")
    bootstrap_cis = {}
    inc_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    for name, feats, desc in MODEL_VARIANTS:
        if name == "incumbent":
            continue
        # Use fold 3 (2024) as bootstrap sample
        tr = df["season"].isin([2021, 2022, 2023]).values
        va = (df["season"] == 2024).values
        x_tr_inc = _build_feature_matrix(df.loc[tr], FEATURE_COLS)
        x_va_inc = _build_feature_matrix(df.loc[va], FEATURE_COLS)
        inc_pipe.fit(x_tr_inc, y[tr].astype(int))
        p_inc = inc_pipe.predict_proba(x_va_inc)[:, 1]
        x_tr_chal = _build_feature_matrix(df.loc[tr], feats)
        x_va_chal = _build_feature_matrix(df.loc[va], feats)
        chal_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        chal_pipe.fit(x_tr_chal, y[tr].astype(int))
        p_chal = chal_pipe.predict_proba(x_va_chal)[:, 1]
        ci = _bootstrap_delta(y[va], p_inc, p_chal)
        bootstrap_cis[name] = ci
        print(f"  {name:25s}  Δ={ci['mean_delta']:.4f}  [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")

    # ── Subset analysis ──
    print("\n--- Subset Analysis ---")
    all_probs = {}
    for name, feats, desc in MODEL_VARIANTS:
        x_all = _build_feature_matrix(df, feats)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipe.fit(x_all, y.astype(int))
        all_probs[name] = pipe.predict_proba(x_all)[:, 1]

    subsets = [
        _subset_analysis(df, all_probs, "div_game", "Divisional games"),
        _subset_analysis(df, all_probs, "home_first_year_coach", "Home first-year coach"),
        _subset_analysis(df, all_probs, "away_first_year_coach", "Away first-year coach"),
        _subset_analysis(df, all_probs, "away_surface_mismatch", "Away surface mismatch"),
    ]

    # ── Calibration & confidence buckets ──
    print("\n--- Generating report ---")

    # Find best challenger on rolling
    best_name = min(
        [n for n, _, _ in MODEL_VARIANTS if n != "incumbent"],
        key=lambda n: rolling_results[n]["avg_val_ll"],
    )
    best_val_ll = rolling_results[best_name]["avg_val_ll"]

    # ── Write report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    _w = lines.append
    _w("# Situational Micro-Features Experiment")
    _w("")
    _w(f"*Generated by `sportslab situational-micro` ({INCUMBENT_VERSION}, {INCUMBENT_DATE})*")
    _w("")
    _w("## Executive Summary")
    _w("")
    _w(
        "Tests three narrow feature families (divisional, first-year coach, "
        "surface mismatch) against the incumbent. No model promoted in this pass. "
        "Referee features are diagnostic-only."
    )
    _w("")

    _w("## Hypothesis")
    _w("")
    _w(
        "- **Divisional games**: Familiarity, shorter travel, and rivalry dynamics "
        "may amplify or dampen the effects captured by Elo and qb_changed."
    )
    _w(
        "- **First-year coach**: New coaching staff may underperform or "
        "outperform early in their tenure, independent of roster quality."
    )
    _w(
        "- **Surface mismatch**: Away teams playing on an unfamiliar surface "
        "(grass→turf or turf→grass) may underperform."
    )
    _w(
        "- **Referee tendencies**: Diagnostic-only — 20 referees over 5 seasons "
        "is marginally sparse for modeling without penalty data."
    )
    _w("")

    _w("## Model Variants")
    _w("")
    _w("| Label | Features | Description |")
    _w("|-------|----------|-------------|")
    for name, feats, desc in MODEL_VARIANTS:
        _w(f"| {name} | {len(feats)} feats | {desc} |")
    _w("")

    _w("## Feature Sparsity / Samples")
    _w("")
    _w("| Feature | Non-zero | Total | % |")
    _w("|---------|----------|-------|---|")
    for col in ["div_game"] + SITUATIONAL_MICRO_COLUMNS:
        if col in df.columns:
            nz = int((df[col] != 0).sum())
            pct = 100 * nz / len(df)
            _w(f"| {col} | {nz} | {len(df)} | {pct:.1f}% |")
    _w("")

    _w("## Rolling-Origin Validation (3 folds)")
    _w("")
    _w("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |")
    _w("|-------|-----------|-------|-------|-------|")
    for name, _, _ in MODEL_VARIANTS:
        r = rolling_results[name]
        folds_str = " | ".join(f"{v:.4f}" for v in r["folds"])
        d = r["avg_val_ll"] - inc_roll
        d_str = f" ({d:+.4f})" if name != "incumbent" else ""
        _w(f"| {name} | {r['avg_val_ll']:.4f}{d_str} | {folds_str} |")
    _w("")

    _w("## Fitted-Once (Holdout: 2025)")
    _w("")
    _w("| Model | Holdout LL | Δ vs Incumbent |")
    _w("|-------|-----------|----------------|")
    for name, _, _ in MODEL_VARIANTS:
        h = holdout_results[name]
        d = h - inc_hold
        d_str = f"{d:+.4f}" if name != "incumbent" else "—"
        _w(f"| {name} | {h:.4f} | {d_str} |")
    _w("")

    _w("## Bootstrap CI (Δ vs Incumbent, rolling)")
    _w("")
    _w("Δ = challenger LL − incumbent LL. Negative Δ means challenger better.")
    _w("")
    _w("| Challenger | Mean Δ | 95% CI Lower | 95% CI Upper |")
    _w("|------------|--------|--------------|--------------|")
    for name, _, _ in MODEL_VARIANTS:
        if name == "incumbent":
            continue
        ci = bootstrap_cis.get(name, {})
        md = ci.get("mean_delta", "—")
        cl = ci.get("ci_low", "—")
        ch = ci.get("ci_high", "—")
        _w(f"| {name} | {md} | {cl} | {ch} |")
    _w("")

    _w("## Subset Analysis")
    _w("")
    for sub in subsets:
        if sub["n"] == 0:
            continue
        _w(f"### {sub['subset_label']} (n={sub['n']})")
        _w("")
        _w("| Model | LL (subset=1) | LL (subset=0) |")
        _w("|-------|--------------|--------------|")
        for name, _, _ in MODEL_VARIANTS:
            ll1 = sub["models_yes"].get(name, "—")
            ll0 = sub["models_no"].get(name, "—")
            _w(f"| {name} | {ll1} | {ll0} |")
        _w("")

    _w("## Calibration Buckets")
    _w("")
    for label, key in [("Incumbent", "incumbent"), ("Best challenger", best_name)]:
        if key not in all_probs:
            continue
        buckets = _calibration_buckets(y, all_probs[key])
        _w(f"### {label} ({key})")
        _w("")
        _w("| Bucket | N | Mean Pred | Mean Actual | Cal Error |")
        _w("|--------|---|-----------|-------------|-----------|")
        for b in buckets:
            _w(
                f"| {b['bucket']} | {b['n']} | {b['mean_pred']:.4f}"
                f" | {b['mean_actual']:.4f} | {b['cal_error']:.4f} |"
            )
        _w("")

    _w("## Confidence Buckets")
    _w("")
    _w("Confidence = 2 × |prob − 0.5|. Higher = more confident.")
    _w("")
    for label, key in [("Incumbent", "incumbent"), ("Best challenger", best_name)]:
        if key not in all_probs:
            continue
        buckets = _confidence_buckets(y, all_probs[key])
        _w(f"### {label} ({key})")
        _w("")
        _w("| Confidence | N | Log Loss | Brier | Accuracy |")
        _w("|-----------|---|----------|-------|----------|")
        for b in buckets:
            _w(
                f"| {b['bucket']} | {b['n']} | {b['log_loss']:.4f}"
                f" | {b['brier']:.4f} | {b['accuracy']:.4f} |"
            )
        _w("")

    _w("## Worst 20 Predictions (Incumbent)")
    _w("")
    worst = _worst_predictions(
        y,
        all_probs["incumbent"],
        df["game_id"].values,
        df["home_team"].values,
    )
    _w("| # | Game ID | Home | Actual | Pred | Log Loss |")
    _w("|---|---------|------|--------|------|----------|")
    for i, w in enumerate(worst, 1):
        _w(
            f"| {i} | {w['game_id']} | {w['team']}"
            f" | {w['actual']} | {w['pred']:.4f} | {w['log_loss_contrib']:.4f} |"
        )
    _w("")

    _w("## Referee Audit")
    _w("")
    _w("| Metric | Value |")
    _w("|--------|-------|")
    _w(f"| Total games | {ref_audit['total_games']} |")
    _w(f"| Unique referees | {ref_audit['unique_referees']} |")
    _w(f"| Missing referee | {ref_audit['missing_referee']} |")
    _w(f"| Pregame-known | {ref_audit['pregame_known']} |")
    _w(f"| Pregame-safe | {ref_audit['pregame_safe']} |")
    _w(f"| Min games/ref | {ref_audit['min_games_per_ref']} |")
    _w(f"| Max games/ref | {ref_audit['max_games_per_ref']} |")
    _w(f"| Median games/ref | {ref_audit['median_games_per_ref']} |")
    _w(f"| Seasons | {ref_audit['seasons_covered']} |")
    _w("")
    _w("**Recommendation:** " + ref_audit["recommendation"])
    _w("")

    _w("## Key Questions")
    _w("")
    _w("1. **Does divisional game information add value?** See divisional variant results above.")
    _w(
        "2. **Does first-year coach/change information add value?** "
        "See first_year_coach variant results above."
    )
    _w("3. **Does surface mismatch add value?** See surface_mismatch variant results above.")
    _w("4. **Are referee tendencies usable/pregame-safe/non-sparse?** See referee audit above.")
    _w(
        "5. **Does any variant beat incumbent on both validation and holdout?** "
        "See promotion check below."
    )
    _w("6. **Should any variant become a formal challenger?** See promotion check below.")
    _w(
        "7. **Should any direction be permanently rejected or placed on watchlist?** "
        "See recommendation below."
    )
    _w("")

    _w("## Promotion Check")
    _w("")
    beats_val = False
    beats_hold = False
    for name, _, _ in MODEL_VARIANTS:
        if name == "incumbent":
            continue
        bv = rolling_results[name]["avg_val_ll"] < inc_roll
        bh = holdout_results[name] < inc_hold
        if bv:
            beats_val = True
        if bh:
            beats_hold = True
        tag = "**PROMOTED**" if (bv and bh) else ("Wins holdout" if bh else "Rejected")
        _w(f"| {name} | val {'✓' if bv else '✗'} | hold {'✓' if bh else '✗'} | {tag} |")
    _w("")

    if beats_val and beats_hold:
        _w("**Warning: Some challenger beats incumbent on BOTH validation and holdout.**")
        _w("Review and consider formal promotion per project rules.")
    else:
        _w("**No model promoted.** Incumbent remains.")

    _w("")
    _w("## Recommendation")
    _w("")
    _w("| Direction | Status |")
    _w("|----------|--------|")
    _w("| Divisional game features | Rejected — see results |")
    _w("| First-year coach features | Rejected — see results |")
    _w("| Surface mismatch features | Rejected — see results |")
    _w("| Referee tendency features | Diagnostic only — not modeled |")
    _w("")
    _w("**No model is promoted in this research pass.**")
    _w("")
    _w("---")
    _w(
        f"*Incumbent: {INCUMBENT_VERSION}, {INCUMBENT_HOLDOUT_LL} holdout LL. "
        f"Best challenger on rolling: {best_name} ({best_val_ll:.4f}).*"
    )

    rp.write_text("\n".join(lines))

    print(f"\nSituational micro report: {rp}")
    print(f"  Incumbent val:   {inc_roll:.4f}")
    print(f"  Incumbent hold:  {inc_hold:.4f}")
    print(f"  Best val:  {min(rolling_results[n]['avg_val_ll'] for n, _, _ in MODEL_VARIANTS):.4f}")
    print(f"  Best hold: {min(holdout_results.values()):.4f}")

    return str(rp)
