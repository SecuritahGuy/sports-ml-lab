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
