# ingest-debug — Debug NFL ingestion

```bash
python -m pip show nflreadpy
python -c "import nflreadpy; print(nflreadpy); print(getattr(nflreadpy, '__version__', 'unknown'))"
sportslab --help || true
sportslab ingest-nfl --help || true
make test
```
