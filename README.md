# Sports ML Lab

An NFL-first sports machine learning research lab. This project builds reproducible, explainable NFL prediction models using the nflverse Python ecosystem (`nflreadpy`).

## Season Scope

**2021–current only.** This project intentionally limits data to the modern NFL era (2021 onward) for:
- Consistent data structure and availability
- Cleaner forward-looking research
- Avoidance of stale play-style and rule environments

No training, testing, backtesting, or tuning on seasons before 2021.

## Ingestion

NFL schedule/game-level data is pulled via [nflreadpy](https://github.com/nflverse/nflreadpy) and saved as parquet.

```bash
# Ingest default seasons (2021–2025)
make ingest-nfl

# Or specify custom seasons (must all be 2021 or later)
sportslab ingest-nfl --seasons 2021 2022 2023 2024 2025
```

Ingestion requires internet access to download from nflverse repositories.

## Structure

- `src/` — Source code
- `tests/` — Unit tests
- `data/raw/nfl/` — Raw ingested data
- `docs/` — Documentation
- `configs/` — Configuration files

## License

MIT License.

## Dashboard (GitHub Pages)

A static project dashboard is available at `docs/` for GitHub Pages:

```bash
# Build or update the dashboard
make build-dashboard

# Or
sportslab build-dashboard
```

### Pages

- `docs/index.md` — Homepage with incumbent summary and registry stats
- `docs/benchmarks.md` — Leaderboard by category, promotion rules
- `docs/predictions.md` — Prediction schema, confidence buckets, caution flags
- `docs/model-card.md` — Full model documentation
- `docs/experiments.md` — All experiments grouped by outcome

### Enable GitHub Pages

1. Go to repo Settings → Pages
2. Source: "Deploy from a branch"
3. Branch: `main`, folder: `/docs`
4. Site will build at `https://<username>.github.io/sports-ml-lab/`

No internet access is required to build the dashboard — it reads local benchmark
registry files and prediction artifacts.

## Research Incumbent

**Model:** Standard Elo + qb_changed + rolling_mov_3 + Platt
**Holdout log loss:** 0.6262
**Validation log loss:** 0.6334

See `reports/benchmarks/nfl_research_incumbent.md` for full details.
