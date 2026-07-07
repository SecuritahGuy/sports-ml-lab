"""RALPH Loop 6: 5 focused challengers against v3.0.0 Frozen QB Overlay.

Challengers (all pregame-safe, no leakage):
1. prior_win_pct — prior-season win% as feature for early-season warmup
2. weather_missing — weather_missing_flag and is_dome/outdoor_game_flag
3. roof_enc — roof type encoding for retractable/open calibration
4. games_since_change — QB continuity feature (games_since_qb_change)
5. isotonic — Isotonic calibration replacing Platt
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.experiment_utils import compute_metrics
from sportslab.evaluation.fold_safe import (
    INCUMBENT_HOLDOUT_LL,
    check_promotion,
)
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features
from sportslab.features.weather import compute_weather_features

MIN_PROMOTION_DELTA = 0.001

BEST_K, BEST_HFA, BEST_REG, BEST_DECAY, BEST_QB_BONUS = 36, 40, 0.1, 32, 0.2

FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"

EXPERIMENT_REPORT = "reports/experiments/ralph6_challengers.md"

ELO_TO_LOGIT = np.log(10) / 400.0
OVERLAY_GAMMA = 1.0
OVERLAY_CAP = 40


def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def _build_gate_mask(df):
    h_changed = df.get("home_qb_changed", pd.Series(0)).values.astype(float)
    a_changed = df.get("away_qb_changed", pd.Series(0)).values.astype(float)
    h_starts = df.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    a_starts = df.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    return (h_changed == 1) | (a_changed == 1) | (h_starts < 17) | (a_starts < 17)


def _apply_frozen_overlay(prob, home_qb_adj, away_qb_adj, gate_mask):
    base_logit = _logit(prob)
    capped_h = np.clip(home_qb_adj, -OVERLAY_CAP, OVERLAY_CAP)
    capped_a = np.clip(away_qb_adj, -OVERLAY_CAP, OVERLAY_CAP)
    net_adj = capped_h - capped_a
    overlay = OVERLAY_GAMMA * net_adj * ELO_TO_LOGIT
    final_logit = base_logit + overlay * gate_mask.astype(float)
    return _sigmoid(final_logit)


def _build_base_df():
    df_raw = pd.read_parquet(FEATURE_TABLE_PATH)
    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_qb_adjustments(df)
    df = compute_situational_features(df)
    df = compute_weather_features(df)
    return df


def _filter_df(df):
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    return df[mask].copy().reset_index(drop=True)


def _compute_prior_season_win_pct(df):
    """Compute prior-season win% for each team for each row."""
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    team_season_wins: Dict[str, Dict[int, int]] = {}
    team_season_games: Dict[str, Dict[int, int]] = {}

    prior_home = []
    prior_away = []

    for _, row in out.iterrows():
        season = int(row["season"])
        home = row["home_team"]
        away = row["away_team"]

        def _get_prior_wp(team):
            prev = team_season_wins.get(team, {})
            prev_g = team_season_games.get(team, {})
            prior_season = season - 1
            w = prev.get(prior_season, 0)
            g = prev_g.get(prior_season, 0)
            return w / g if g > 0 else 0.5

        prior_home.append(_get_prior_wp(home))
        prior_away.append(_get_prior_wp(away))

        home_won = row.get("home_win")
        if pd.notna(home_won):
            for team, won in [(home, bool(home_won == 1)), (away, bool(home_won == 0))]:
                if team not in team_season_wins:
                    team_season_wins[team] = {}
                    team_season_games[team] = {}
                    prev_w = team_season_wins[team].get(season, 0)
                    team_season_wins[team][season] = prev_w + (1 if won else 0)
                team_season_games[team][season] = team_season_games[team].get(season, 0) + 1

    out["home_prior_win_pct"] = prior_home
    out["away_prior_win_pct"] = prior_away
    return out


# ── Challenger Model Builders ──

def _incumbent_model_fn(df, train_mask, val_mask):
    """Reproduce incumbent: Platt(qb_changed + rolling_mov_3) + overlay."""
    base_cols = [
        "home_qb_changed", "away_qb_changed",
        "home_rolling_mov_3", "away_rolling_mov_3",
    ]
    avail = [c for c in base_cols if c in df.columns]
    elo = df["elo_prob"].values.reshape(-1, 1)
    feat = df[avail].values if avail else None
    x_full = np.column_stack([elo, feat]) if feat is not None else elo
    y = df[TARGET_COLUMN].astype(int).values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(x_full[train_mask], y[train_mask])
    base_prob = pipe.predict_proba(x_full[val_mask])[:, 1]

    h_adj = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
    a_adj = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
    gate = _build_gate_mask(df)
    base_logit = _logit(base_prob)
    capped_h = np.clip(h_adj[val_mask], -OVERLAY_CAP, OVERLAY_CAP)
    capped_a = np.clip(a_adj[val_mask], -OVERLAY_CAP, OVERLAY_CAP)
    net_adj = capped_h - capped_a
    overlay = OVERLAY_GAMMA * net_adj * ELO_TO_LOGIT
    final_logit = base_logit + overlay * gate[val_mask].astype(float)
    return _sigmoid(final_logit)


def _make_challenger_fn(extra_cols: List[str], label: str) -> Callable:
    def _x_matrix(df, cols):
        elo = df["elo_prob"].values.reshape(-1, 1)
        f = df[cols].values if cols else None
        return np.column_stack([elo, f]) if f is not None else elo

    def model_fn(df, train_mask, val_mask):
        base_cols = [
            "home_qb_changed", "away_qb_changed",
            "home_rolling_mov_3", "away_rolling_mov_3",
        ]
        all_cols = base_cols + extra_cols
        avail = [c for c in all_cols if c in df.columns]
        missing = [c for c in extra_cols if c not in df.columns]
        if missing:
            print(f"  [{label}] Missing columns: {missing}")

        x_full = _x_matrix(df, avail)
        y = df[TARGET_COLUMN].astype(int).values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(x_full[train_mask], y[train_mask])
        base_prob = pipe.predict_proba(x_full[val_mask])[:, 1]

        h_adj = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
        a_adj = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
        gate = _build_gate_mask(df)
        base_logit = _logit(base_prob)
        capped_h = np.clip(h_adj[val_mask], -OVERLAY_CAP, OVERLAY_CAP)
        capped_a = np.clip(a_adj[val_mask], -OVERLAY_CAP, OVERLAY_CAP)
        net_adj = capped_h - capped_a
        overlay = OVERLAY_GAMMA * net_adj * ELO_TO_LOGIT
        final_logit = base_logit + overlay * gate[val_mask].astype(float)
        return _sigmoid(final_logit)
    return model_fn


def _make_isotonic_fn(extra_cols: List[str]) -> Callable:
    def _x_matrix(df, cols):
        elo = df["elo_prob"].values.reshape(-1, 1)
        f = df[cols].values if cols else None
        return np.column_stack([elo, f]) if f is not None else elo

    def model_fn(df, train_mask, val_mask):
        base_cols = [
            "home_qb_changed", "away_qb_changed",
            "home_rolling_mov_3", "away_rolling_mov_3",
        ]
        all_cols = base_cols + extra_cols
        avail = [c for c in all_cols if c in df.columns]

        x_full = _x_matrix(df, avail)
        y = df[TARGET_COLUMN].astype(int).values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(x_full[train_mask], y[train_mask])
        platt_prob = pipe.predict_proba(x_full[val_mask])[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pipe.predict_proba(x_full[train_mask])[:, 1], y[train_mask])
        base_prob = iso.transform(platt_prob)

        h_adj = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
        a_adj = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
        gate = _build_gate_mask(df)
        base_logit = _logit(base_prob)
        capped_h = np.clip(h_adj[val_mask], -OVERLAY_CAP, OVERLAY_CAP)
        capped_a = np.clip(a_adj[val_mask], -OVERLAY_CAP, OVERLAY_CAP)
        net_adj = capped_h - capped_a
        overlay = OVERLAY_GAMMA * net_adj * ELO_TO_LOGIT
        final_logit = base_logit + overlay * gate[val_mask].astype(float)
        return _sigmoid(final_logit)
    return model_fn


# ── Rolling-Origin Runner ──

def _run_fold_safe(df, model_fn) -> Tuple[List[Dict], float]:
    fold_metrics = []
    for train_seasons, val_season in ROLLING_FOLDS:
        me = df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
        tr = df["season"].isin(train_seasons).values & me
        va = (df["season"] == val_season).values & me
        preds = model_fn(df, tr, va)
        y_val = df.loc[va, TARGET_COLUMN].astype(float).values
        m = compute_metrics(y_val, preds)
        fold_metrics.append(m)
    avg_ll = float(np.mean([m["log_loss"] for m in fold_metrics if "log_loss" in m]))
    return fold_metrics, round(avg_ll, 4)


def _score_holdout(df, model_fn) -> Dict:
    train_mask = (df["season"].isin([2021, 2022, 2023, 2024]).values
                  & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values)
    ho = (df["season"] == HOLDOUT_SEASON).values & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    preds = model_fn(df, train_mask, ho)
    y_ho = df.loc[ho, TARGET_COLUMN].astype(float).values
    m = compute_metrics(y_ho, preds)
    from sklearn.metrics import roc_auc_score
    valid = ~np.isnan(y_ho)
    if valid.sum() > 1 and len(np.unique(y_ho[valid])) > 1:
        m["roc_auc"] = round(float(roc_auc_score(y_ho[valid], preds[valid])), 4)
    else:
        m["roc_auc"] = None
    return m


# ── Subgroup Analysis ──

def _subgroup_metrics(df, model_fn, condition, label) -> Dict:
    """Compute metrics on a subgroup defined by `condition` (boolean array).

    Trains on 2021-2024, evaluates on 2025 holdout ∩ condition.
    """
    if condition.sum() == 0:
        return {"label": label, "n": 0, "log_loss": None}
    train_mask = (df["season"].isin([2021, 2022, 2023, 2024]).values
                  & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values)
    ho = (df["season"] == HOLDOUT_SEASON).values & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    target_mask = ho & condition
    if target_mask.sum() == 0:
        return {"label": label, "n": 0, "log_loss": None}
    preds = model_fn(df, train_mask, target_mask)
    y_sub = df.loc[target_mask, TARGET_COLUMN].astype(float).values
    m = compute_metrics(y_sub, preds)
    return {"label": label, "n": int(target_mask.sum()), "log_loss": m.get("log_loss")}


# ── Subgroup definitions ──

_WEEKS_1_4 = None
_WEATHER_MISSING = None
_RETRACTABLE = None
_QB_CHANGED = None


def _subgroup_mask(df, key):
    if key == "early":
        return df["week"].values <= 4
    elif key == "weather_missing":
        return df.get("weather_missing_flag", pd.Series(0)).values == 1
    elif key == "retractable":
        roof_str = df["roof"].astype(str).str.lower()
        return roof_str.isin(["open", "retractable"]).values
    elif key == "qb_changed":
        hc = df.get("home_qb_changed", pd.Series(0)).values
        ac = df.get("away_qb_changed", pd.Series(0)).values
        return (hc == 1) | (ac == 1)
    return np.zeros(len(df), dtype=bool)


# ── Main ──

def run_ralph6_experiment(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = EXPERIMENT_REPORT,
) -> str:
    print("=== Building base feature spine ===")
    df = _build_base_df()
    df = _compute_prior_season_win_pct(df)
    df = _filter_df(df)
    print(f"  Eligible games: {len(df)}")

    # ── Define challengers ──
    challengers = {
        "incumbent": {
            "fn": _incumbent_model_fn,
            "cols": [],
            "desc": "Platt(qb_changed + rolling_mov_3) + frozen QB overlay",
        },
        "prior_win_pct": {
            "fn": _make_challenger_fn(
                ["home_prior_win_pct", "away_prior_win_pct"], "prior_win_pct"
            ),
            "cols": ["home_prior_win_pct", "away_prior_win_pct"],
            "desc": "Incumbent + prior-season win%",
        },
        "weather_missing": {
            "fn": _make_challenger_fn(
                ["weather_missing_flag", "is_dome", "outdoor_game_flag"],
                "weather_missing",
            ),
            "cols": ["weather_missing_flag", "is_dome", "outdoor_game_flag"],
            "desc": "Incumbent + weather_missing_flag + dome/outdoor",
        },
        "roof_enc": {
            "fn": _make_challenger_fn(["roof_enc"], "roof_enc"),
            "cols": ["roof_enc"],
            "desc": "Incumbent + roof_enc (label-encoded roof type)",
        },
        "games_since_change": {
            "fn": _make_challenger_fn(
                ["home_games_since_qb_change", "away_games_since_qb_change"],
                "games_since_change",
            ),
            "cols": ["home_games_since_qb_change", "away_games_since_qb_change"],
            "desc": "Incumbent + games_since_qb_change",
        },
        "isotonic": {
            "fn": _make_isotonic_fn([]),
            "cols": [],
            "desc": "Platt → Isotonic + incumbent qb_changed+mov3 + overlay",
        },
    }

    results = {}

    # ── Run each challenger ──
    for name, cfg in challengers.items():
        print(f"\n=== {name} ===")
        fn = cfg["fn"]
        fold_metrics, val_ll = _run_fold_safe(df, fn)
        hold_metrics = _score_holdout(df, fn)

        # Subgroup analysis
        subgroups = ["early", "weather_missing", "retractable", "qb_changed"]
        sub_results = {}
        for sk in subgroups:
            mask = _subgroup_mask(df, sk)
            sr = _subgroup_metrics(df, fn, mask, sk)
            sub_results[sk] = sr

        results[name] = {
            "val_ll": val_ll,
            "fold_metrics": fold_metrics,
            "holdout": hold_metrics,
            "subgroups": sub_results,
        }
        print(f"  Val LL: {val_ll:.4f}")
        print(f"  Holdout LL: {hold_metrics.get('log_loss', 'N/A')}")

    # ── Comparison vs incumbent ──
    inc_val = results["incumbent"]["val_ll"]
    inc_hold = results["incumbent"]["holdout"]["log_loss"]
    print("\n\n=== Incumbent Reference ===")
    print(f"  Val LL: {inc_val:.4f}")
    print(f"  Holdout LL: {inc_hold:.4f}")

    promoted = []
    rejected = []
    for name, r in results.items():
        if name == "incumbent":
            continue
        v = r["val_ll"]
        h = r["holdout"]["log_loss"]
        verdict = check_promotion(
            v, h, incumbent_val=inc_val, incumbent_holdout=inc_hold, delta=MIN_PROMOTION_DELTA,
        )
        if verdict["promoted"]:
            promoted.append(name)
            vd = verdict['val_delta']
            hd = verdict['holdout_delta']
            print(f"\n  ✅ PROMOTED: {name} (val Δ={vd:.4f}, hold Δ={hd:.4f})")
        else:
            rejected.append(name)
            vd = verdict['val_delta']
            hd = verdict['holdout_delta']
            print(f"\n  ❌ REJECTED: {name} (val Δ={vd:.4f}, hold Δ={hd:.4f})")

    # ── Write report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# RALPH Loop 6: Five Focused Challengers\n\n")
        f.write(f"*Incumbent: v3.0.0 Frozen QB Overlay (holdout LL {INCUMBENT_HOLDOUT_LL})*\n\n")
        f.write("## Challengers\n\n")
        f.write("| ID | Description | Hypothesis |\n")
        f.write("|----|-------------|-----------|\n")
        f.write("| prior_win_pct | +prior-season win% | Early weeks need better priors |\n")
        f.write("| weather_missing | +weather_missing+is_dome+outdoor"
                " | Missing weather flag adds signal |\n")
        f.write("| roof_enc | +roof_enc | Roof type corrects dome/retractable bias |\n")
        f.write("| games_since_change | +games_since_qb_change"
                " | QB continuity beyond binary changed flag |\n")
        f.write("| isotonic | Isotonic instead of Platt"
                " | Better calibration for all probabilities |\n\n")

        f.write("## Validation (Rolling-Origin)\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 | Δ vs Inc |\n")
        f.write("|-------|-----------|-------|-------|-------|----------|\n")
        for name, r in results.items():
            fm = r["fold_metrics"]
            delta = r["val_ll"] - inc_val
            f.write(
                f"| {name} | {r['val_ll']:.4f}"
                f" | {fm[0]['log_loss']:.4f}"
                f" | {fm[1]['log_loss']:.4f}"
                f" | {fm[2]['log_loss']:.4f}"
                f" | {delta:+.4f} |\n"
            )
        f.write("\n")

        f.write("## Holdout (2025)\n\n")
        f.write("| Model | Hold LL | Brier | Acc | AUC | Δ vs Inc |\n")
        f.write("|-------|---------|-------|-----|-----|----------|\n")
        for name, r in results.items():
            h = r["holdout"]
            delta = h["log_loss"] - inc_hold
            f.write(f"| {name} | {h.get('log_loss', 'N/A')}"
                    f" | {h.get('brier', 'N/A')} | {h.get('accuracy', 'N/A')}"
                    f" | {h.get('roc_auc', 'N/A')} | {delta:+.4f} |\n")
        f.write("\n")

        f.write("## Subgroup Impact (2025 Holdout)\n\n")
        subgroups_long = {
            "early": "Early season (Weeks 1-4)",
            "weather_missing": "Weather data missing",
            "retractable": "Retractable/open roof",
            "qb_changed": "QB changed",
        }
        f.write("| Subgroup | Model | N | Log Loss |\n")
        f.write("|----------|-------|---|----------|\n")
        for sk, slabel in subgroups_long.items():
            for name, r in results.items():
                sr = r["subgroups"].get(sk, {})
                n = sr.get("n", 0)
                ll = sr.get("log_loss")
                ll_str = f"{ll:.4f}" if ll is not None else "—"
                f.write(f"| {slabel} | {name} | {n} | {ll_str} |\n")
        f.write("\n")

        f.write("## Decisions\n\n")
        for name in promoted:
            r = results[name]
            v = r["val_ll"]
            h = r["holdout"]["log_loss"]
            f.write(f"### ✅ {name}\n\n")
            f.write(
                f"Val LL: {v:.4f} (Δ={v-inc_val:.4f}),"
                f" Holdout: {h:.4f} (Δ={h-inc_hold:.4f})\n\n"
            )
            f.write("Promoted — beats incumbent on both val and holdout with Δ≥0.001.\n\n")
        for name in rejected:
            r = results[name]
            v = r["val_ll"]
            h = r["holdout"]["log_loss"]
            if name == "incumbent":
                continue
            reason = "val "
            if v >= inc_val + MIN_PROMOTION_DELTA:
                reason += f"worse ({v-inc_val:+.4f}) "
            else:
                reason += f"better ({v-inc_val:+.4f}) "
            reason += "holdout "
            if h >= inc_hold + MIN_PROMOTION_DELTA:
                reason += f"worse ({h-inc_hold:+.4f})"
            else:
                reason += f"better ({h-inc_hold:+.4f})"
            f.write(f"### ❌ {name}\n\n")
            f.write(
                f"Val LL: {v:.4f} (Δ={v-inc_val:.4f}),"
                f" Holdout: {h:.4f} (Δ={h-inc_hold:.4f})\n\n"
            )
            f.write(f"Rejected — {reason}.\n\n")

        if not promoted:
            f.write("**No challenger beats the incumbent.**\n\n")
        else:
            f.write(f"**Promoted: {', '.join(promoted)}**\n\n")

        f.write("## Leakage Assessment\n\n")
        f.write("| Feature | Source | Leakage Risk | Live-Safe |\n")
        f.write("|---------|--------|--------------|-----------|\n")
        f.write("| prior_win_pct | Prior season results | None (available precseason) | Yes |\n")
        f.write("| weather_missing_flag | Feature table | None | Yes |\n")
        f.write("| is_dome | Stadium info | None | Yes |\n")
        f.write("| outdoor_game_flag | Stadium info | None | Yes |\n")
        f.write("| roof_enc | Stadium info | None | Yes |\n")
        f.write("| games_since_qb_change | Chronological QB tracker"
                " | None (pregame) | Yes (weekly tracker) |\n\n")

        f.write("## Next Steps\n\n")
        if promoted:
            f.write("1. Promote new incumbent with full registry update\n")
            f.write("2. Update model-trust report with new incumbent\n")
            f.write("3. Add regression tests for new features\n")
        else:
            f.write("1. All challengers rejected — incumbent unchanged\n")
            f.write("2. Preserve challenger code for future comparison\n")
            f.write("3. Consider testing with 2026 data when available\n\n")

    print(f"\nReport: {rp}")
    return str(rp)
