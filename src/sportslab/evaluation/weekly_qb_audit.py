"""Weekly QB source audit — compares QB sourcing strategies per game.

For a given season + week, compares three QB sources:
  1. Oracle: feature table's home_qb_id/away_qb_id (actual starters)
  2. Depth chart: preseason snapshot (auto_qb)
  3. Weekly tracker: prior-week actual starters (weekly_qb)

Shows per-game the QB identity, whether the overlay gate fires,
the overlay delta, and the resulting win probability under each source.

Usage:
    sportslab weekly-qb-audit --season 2026 --week 1
    sportslab weekly-qb-audit --season 2025 --week 5 --output audit.csv
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sportslab.evaluation.predict_incumbent import (
    BEST_DECAY,
    BEST_HFA,
    BEST_K,
    BEST_QB_BONUS,
    BEST_REG,
    ELO_TO_LOGIT,
    FEATURE_COLS,
    INCUMBENT_VERSION,
    OVERLAY_CAP,
    OVERLAY_GAMMA,
    _build_pipeline,
    _logit,
    _sigmoid,
)
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024]

QB_SOURCE_LABELS = {
    "oracle": "Feature table (actual)",
    "depth_chart": "Depth chart snapshot",
    "weekly_qb": "Weekly tracker",
}


def _load_ft() -> pd.DataFrame:
    fp = Path(FEATURE_TABLE_PATH)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURE_TABLE_PATH}")
    return pd.read_parquet(fp)


def _build_gate_mask(df: pd.DataFrame) -> np.ndarray:
    h_changed = df.get("home_qb_changed", pd.Series(0)).values.astype(float)
    a_changed = df.get("away_qb_changed", pd.Series(0)).values.astype(float)
    h_starts = (
        df.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    )
    a_starts = (
        df.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    )
    return (h_changed == 1) | (a_changed == 1) | (h_starts < 17) | (a_starts < 17)


def _compute_overlay_delta(
    home_qb_adj: np.ndarray, away_qb_adj: np.ndarray
) -> np.ndarray:
    capped_h = np.clip(home_qb_adj, -OVERLAY_CAP, OVERLAY_CAP)
    capped_a = np.clip(away_qb_adj, -OVERLAY_CAP, OVERLAY_CAP)
    net_adj = capped_h - capped_a
    return OVERLAY_GAMMA * net_adj * ELO_TO_LOGIT


def _apply_qb_source(
    df_combined: pd.DataFrame,
    qb_df: pd.DataFrame,
    pred_game_ids: set,
) -> pd.DataFrame:
    """Apply a QB source CSV to the combined dataframe and compute features."""
    out = df_combined.copy()

    # Override QB ids with the source
    qb_lookup = qb_df.set_index("game_id")[["home_qb_id", "away_qb_id"]]
    for gid in pred_game_ids:
        if gid in qb_lookup.index:
            out.loc[out["game_id"] == gid, "home_qb_id"] = qb_lookup.loc[gid, "home_qb_id"]
            out.loc[out["game_id"] == gid, "away_qb_id"] = qb_lookup.loc[gid, "away_qb_id"]

    # Compute features
    overrides = build_team_regression_overrides(
        out, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS,
    )
    out = compute_elo_features(
        out,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    out = compute_qb_features(out)
    out = compute_situational_features(out)
    out = compute_qb_adjustments(out)

    return out


def _source_to_qb_df(
    source: str, season: int, week: int
) -> Tuple[pd.DataFrame, str]:
    if source == "oracle":
        return None, "oracle"
    elif source == "depth_chart":
        from sportslab.features.qb_auto_source import build_auto_qb_csv
        return build_auto_qb_csv(season, week=week)
    elif source == "weekly_qb":
        from sportslab.features.qb_auto_source import build_weekly_qb_csv
        return build_weekly_qb_csv(season, week=week)
    else:
        raise ValueError(f"Unknown QB source: {source}")


def _extract_col(
    df: pd.DataFrame, col: str, pred_mask: np.ndarray, default=None
) -> np.ndarray:
    if col in df.columns:
        return df.loc[pred_mask, col].values
    return np.full(pred_mask.sum(), default) if default is not None else np.full(pred_mask.sum(), np.nan)


def run_weekly_qb_audit(
    season: int,
    week: int,
    output_path: Optional[str] = None,
) -> Dict[str, str]:
    """Compare QB sourcing strategies for a single week.

    Args:
        season: Season year.
        week: Week number.
        output_path: Optional CSV output path.

    Returns:
        Dict with {"report": path or message}.
    """
    print(f"=== Weekly QB Source Audit: {season} Week {week} ===\n")

    ft = _load_ft()

    # Historical training data (fixed)
    df_train = ft[ft["season"].isin(HISTORICAL_SEASONS) & ft[MODEL_ELIGIBLE_COLUMN]].copy()

    # Target week games
    target = ft[
        (ft["season"] == season)
        & (ft["week"] == week)
        & ft[MODEL_ELIGIBLE_COLUMN]
        & (~ft.get("is_neutral", pd.Series(False)).fillna(False))
        & ft["home_win"].notna()
    ].copy()

    if len(target) == 0:
        msg = f"No eligible graded games found for {season} week {week}"
        print(f"  {msg}")
        return {"report": msg}

    pred_game_ids = set(target["game_id"].values)
    n = len(target)
    print(f"  Target games: {n}")
    print(f"  QB sources to compare: oracle, depth_chart, weekly_qb\n")

    # Build combined dataframe once (same training for all sources)
    df_pred = target.copy()
    df_pred["home_win"] = pd.NA
    df_combined = pd.concat([df_train, df_pred], ignore_index=True)
    df_combined = df_combined.sort_values(
        ["season", "week", "gameday"]
    ).reset_index(drop=True)

    pred_mask = df_combined["game_id"].isin(pred_game_ids).values
    n_pred = pred_mask.sum()

    # Get the actual results for comparison
    actuals = df_combined.loc[pred_mask, "home_win"].values

    # Run each source
    sources = ["oracle", "depth_chart", "weekly_qb"]
    source_data = {}

    for src in sources:
        print(f"  Running source: {src}...", end=" ")
        qb_df, qb_label = _source_to_qb_df(src, season, week)

        df_src = df_combined.copy()
        if qb_df is not None:
            qb_lookup = qb_df.set_index("game_id")[["home_qb_id", "away_qb_id"]]
            for gid in pred_game_ids:
                if gid in qb_lookup.index:
                    row = qb_lookup.loc[gid]
                    df_src.loc[df_src["game_id"] == gid, "home_qb_id"] = row["home_qb_id"]
                    df_src.loc[df_src["game_id"] == gid, "away_qb_id"] = row["away_qb_id"]

        overrides = build_team_regression_overrides(
            df_src, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS,
        )
        df_src = compute_elo_features(
            df_src, k_factor=BEST_K, home_advantage=BEST_HFA,
            preseason_regression=BEST_REG, team_regression_overrides=overrides,
            decay_half_life=BEST_DECAY,
        )
        df_src = compute_qb_features(df_src)
        df_src = compute_situational_features(df_src)
        df_src = compute_qb_adjustments(df_src)

        # Elo probability from the incumbent
        elo_prob = df_src.loc[pred_mask, "elo_prob"].values
        feat_cols = [c for c in FEATURE_COLS if c in df_src.columns]
        has_feats = len(feat_cols) > 0

        # Fit Platt on training data
        train_mask = ~df_src["home_win"].isna().values & ~pred_mask
        train_elo = df_src.loc[train_mask, "elo_prob"].values
        train_feat = (
            df_src.loc[train_mask, feat_cols].values
            if has_feats
            else np.empty((train_mask.sum(), 0))
        )
        train_y = df_src.loc[train_mask, "home_win"].astype(int).values
        x_train = np.column_stack([train_elo, train_feat]) if train_feat.size else train_elo.reshape(-1, 1)

        pipe = _build_pipeline()
        pipe.fit(x_train, train_y)

        pred_feat = (
            df_src.loc[pred_mask, feat_cols].values
            if has_feats
            else np.empty((n_pred, 0))
        )
        x_pred = np.column_stack([elo_prob, pred_feat]) if pred_feat.size else elo_prob.reshape(-1, 1)
        platt_prob = pipe.predict_proba(x_pred)[:, 1]

        # QB overlay
        gate = _build_gate_mask(df_src)
        pred_gate = gate[pred_mask]
        h_adj = _extract_col(df_src, "home_qb_adj", pred_mask, 0.0).astype(float)
        a_adj = _extract_col(df_src, "away_qb_adj", pred_mask, 0.0).astype(float)
        overlay_delta = _compute_overlay_delta(h_adj, a_adj)
        final_prob = _apply_overlay(platt_prob, h_adj, a_adj, pred_gate)

        source_data[src] = {
            "home_qb_id": _extract_col(df_src, "home_qb_id", pred_mask, ""),
            "away_qb_id": _extract_col(df_src, "away_qb_id", pred_mask, ""),
            "home_qb_changed": _extract_col(df_src, "home_qb_changed", pred_mask, 0).astype(float),
            "away_qb_changed": _extract_col(df_src, "away_qb_changed", pred_mask, 0).astype(float),
            "home_qb_starts": _extract_col(df_src, "home_qb_team_starts_pre", pred_mask, 0).astype(float),
            "away_qb_starts": _extract_col(df_src, "away_qb_team_starts_pre", pred_mask, 0).astype(float),
            "gate_triggered": pred_gate.astype(int),
            "home_qb_adj": h_adj,
            "away_qb_adj": a_adj,
            "overlay_logit_delta": overlay_delta,
            "platt_prob": platt_prob,
            "final_prob": final_prob,
        }
        print(f"done ({source_data[src]['home_qb_id'].size} games)")

    # Build output DataFrame
    rows = []
    game_ids = df_combined.loc[pred_mask, "game_id"].values
    home_teams = df_combined.loc[pred_mask, "home_team"].values
    away_teams = df_combined.loc[pred_mask, "away_team"].values
    weeks = df_combined.loc[pred_mask, "week"].values
    gamedays = df_combined.loc[pred_mask, "gameday"].values

    for i in range(n_pred):
        row = {
            "game_id": game_ids[i],
            "season": season,
            "week": weeks[i],
            "gameday": gamedays[i],
            "home_team": home_teams[i],
            "away_team": away_teams[i],
            "actual_home_win": int(actuals[i]) if not pd.isna(actuals[i]) else pd.NA,
        }

        for src in sources:
            d = source_data[src]
            prefix = {"oracle": "oracle", "depth_chart": "dc", "weekly_qb": "wk"}[src]
            row[f"{prefix}_home_qb_id"] = d["home_qb_id"][i]
            row[f"{prefix}_away_qb_id"] = d["away_qb_id"][i]
            row[f"{prefix}_h_changed"] = int(d["home_qb_changed"][i])
            row[f"{prefix}_a_changed"] = int(d["away_qb_changed"][i])
            row[f"{prefix}_h_starts"] = int(d["home_qb_starts"][i]) if not np.isnan(d["home_qb_starts"][i]) else pd.NA
            row[f"{prefix}_a_starts"] = int(d["away_qb_starts"][i]) if not np.isnan(d["away_qb_starts"][i]) else pd.NA
            row[f"{prefix}_gate"] = int(d["gate_triggered"][i])
            row[f"{prefix}_h_adj"] = round(d["home_qb_adj"][i], 2)
            row[f"{prefix}_a_adj"] = round(d["away_qb_adj"][i], 2)
            row[f"{prefix}_overlay_logit"] = round(d["overlay_logit_delta"][i], 4)
            row[f"{prefix}_final_prob"] = round(d["final_prob"][i], 4)

        # Comparison columns
        def _str_cmp(a, b):
            if pd.notna(a) and pd.notna(b):
                return str(a) != str(b)
            return False
        def _str_eq(a, b):
            if pd.notna(a) and pd.notna(b):
                return str(a) == str(b)
            return False

        dc_id = row["dc_home_qb_id"]
        wk_id = row["wk_home_qb_id"]
        or_id = row["oracle_home_qb_id"]
        row["h_qb_wk_vs_dc"] = _str_cmp(wk_id, dc_id)
        row["a_qb_wk_vs_dc"] = _str_cmp(row["wk_away_qb_id"], row["dc_away_qb_id"])
        row["h_qb_wk_vs_oracle"] = _str_eq(wk_id, or_id)
        row["a_qb_wk_vs_oracle"] = _str_eq(row["wk_away_qb_id"], row["oracle_away_qb_id"])
        row["gate_change_wk_vs_dc"] = bool(row["wk_gate"] != row["dc_gate"])
        row["prob_diff_wk_vs_dc"] = round(row["wk_final_prob"] - row["dc_final_prob"], 4)
        row["prob_diff_wk_vs_oracle"] = round(row["wk_final_prob"] - row["oracle_final_prob"], 4)

        rows.append(row)

    out_df = pd.DataFrame(rows)
    print(f"\n  Audit generated: {len(out_df)} games")

    # Summary stats (handle NA in boolean comparisons)
    n_gate_change = out_df["gate_change_wk_vs_dc"].fillna(False).astype(bool).sum()
    n_qb_diff = (
        out_df["h_qb_wk_vs_dc"].fillna(False).astype(bool)
        | out_df["a_qb_wk_vs_dc"].fillna(False).astype(bool)
    ).sum()
    n_wk_correct_h = out_df["h_qb_wk_vs_oracle"].sum() if "h_qb_wk_vs_oracle" in out_df.columns else 0
    n_wk_correct_a = out_df["a_qb_wk_vs_oracle"].sum() if "a_qb_wk_vs_oracle" in out_df.columns else 0
    n_wk_known = len(out_df) if "h_qb_wk_vs_oracle" in out_df.columns else 0

    print(f"  Games where QB differs (weekly vs snapshot): {n_qb_diff}/{n}")
    print(f"  Games where gate changes (weekly vs snapshot): {n_gate_change}/{n}")
    print(f"  Weekly QB matches oracle: {n_wk_correct_h}/{n_wk_known} home, {n_wk_correct_a}/{n_wk_known} away")

    max_prob_diff = out_df["prob_diff_wk_vs_dc"].abs().max()
    mean_abs_prob_diff = out_df["prob_diff_wk_vs_dc"].abs().mean()
    print(f"  Max prob diff (weekly vs snapshot): {max_prob_diff:.4f}")
    print(f"  Mean abs prob diff (weekly vs snapshot): {mean_abs_prob_diff:.4f}")

    # Save
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")
        result = {"report": str(out_path)}
    else:
        print("\n  Columns:")
        for c in out_df.columns:
            print(f"    {c}")
        result = {"report": "console output"}

    return result


def _apply_overlay(
    platt_prob: np.ndarray,
    home_qb_adj: np.ndarray,
    away_qb_adj: np.ndarray,
    gate_mask: np.ndarray,
) -> np.ndarray:
    base_logit = _logit(platt_prob)
    capped_h = np.clip(home_qb_adj, -OVERLAY_CAP, OVERLAY_CAP)
    capped_a = np.clip(away_qb_adj, -OVERLAY_CAP, OVERLAY_CAP)
    net_adj = capped_h - capped_a
    overlay = OVERLAY_GAMMA * net_adj * ELO_TO_LOGIT
    final_logit = base_logit + overlay * gate_mask.astype(float)
    return _sigmoid(final_logit)
