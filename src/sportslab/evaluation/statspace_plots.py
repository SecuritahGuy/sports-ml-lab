"""StatSpace metric visualizations: pairwise scatterplots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sportslab.evaluation.team_profiles import build_team_profiles

_METRICS = [
    ("fraud_detector_rating", "FDR", "Overall Team Quality"),
    ("doba_score", "DOBA", "Offensive Efficiency"),
    ("chaos_rate", "Chaos Rate", "Defensive Disruption"),
    ("aggression_score", "Aggression Score", "4th-Down Aggressiveness"),
    ("qb_lift_index", "QB Lift", "QB Value Beyond Support"),
]

_SEASONS = [2021, 2022, 2023, 2024, 2025]
_SEASON_COLORS = {2021: "#1f77b4", 2022: "#ff7f0e", 2023: "#2ca02c",
                  2024: "#d62728", 2025: "#9467bd"}


def plot_all_pairs(profiles: pd.DataFrame, out_dir: str = "reports/figures"):
    """Generate pairwise scatterplots for all StatSpace metrics."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = len(_METRICS)
    metric_cols = [m[0] for m in _METRICS]
    metric_names = [m[1] for m in _METRICS]

    # Plot each pair
    for i in range(n):
        for j in range(i + 1, n):
            col_x, name_x, desc_x = _METRICS[j]
            col_y, name_y, desc_y = _METRICS[i]

            fig, ax = plt.subplots(figsize=(8, 6))

            for s in _SEASONS:
                sub = profiles[profiles["season"] == s].dropna(subset=[col_x, col_y])
                if sub.empty:
                    continue
                ax.scatter(sub[col_x], sub[col_y],
                          c=_SEASON_COLORS[s], label=str(s), alpha=0.6, s=30)

            # Correlation
            valid = profiles.dropna(subset=[col_x, col_y])
            if len(valid) > 2:
                corr = valid[col_x].corr(valid[col_y])
                ax.set_title(f"{name_x} vs {name_y}  (r={corr:.3f})", fontsize=12)
            else:
                ax.set_title(f"{name_x} vs {name_y}", fontsize=12)

            # Label extreme teams (top/bottom 3 in each metric)
            for metric_col, label_name in [(col_x, name_x), (col_y, name_y)]:
                extremes = valid.nlargest(3, metric_col)
                for _, r in extremes.iterrows():
                    ax.annotate(f"{r['team']}{r['season']}",
                                (r[col_x], r[col_y]), fontsize=6, alpha=0.7)
                extremes = valid.nsmallest(3, metric_col)
                for _, r in extremes.iterrows():
                    ax.annotate(f"{r['team']}{r['season']}",
                                (r[col_x], r[col_y]), fontsize=6, alpha=0.7)

            ax.set_xlabel(f"{name_x}\n{desc_x}", fontsize=9)
            ax.set_ylabel(f"{name_y}\n{desc_y}", fontsize=9)
            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
            ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
            ax.legend(title="Season", fontsize=7, title_fontsize=8)
            fig.tight_layout()

            x_slug = name_x.lower().replace(" ", "_")
            y_slug = name_y.lower().replace(" ", "_")
            fname = f"statspace_{x_slug}_vs_{y_slug}.png"
            fig.savefig(out / fname, dpi=150)
            plt.close(fig)
            print(f"  Saved {out / fname}")

    # Summary correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    valid = profiles.dropna(subset=metric_cols)
    corr_matrix = valid[metric_cols].corr()
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(metric_names, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(metric_names, fontsize=8)
    for i2 in range(n):
        for j2 in range(n):
            val = corr_matrix.values[i2, j2]
            ax.text(j2, i2, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if abs(val) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("StatSpace Metric Correlations", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "statspace_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out / 'statspace_correlation_heatmap.png'}")

    print(f"\nAll plots saved to {out}/")


def build_statspace_plots(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    output_dir: str = "reports/figures",
) -> str:
    profiles = build_team_profiles(ft_path=ft_path)
    plot_all_pairs(profiles, out_dir=output_dir)
    return output_dir


if __name__ == "__main__":
    build_statspace_plots()
