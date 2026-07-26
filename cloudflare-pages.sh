#!/usr/bin/env bash
# Cloudflare Pages build script for Sports ML Lab
# Cloudflare already runs `pip install .` (from pyproject.toml),
# so pandas/numpy/scikit-learn are available.
# We just need pyarrow for parquet I/O.
set -e
echo "=== Sports ML Lab — Cloudflare Pages Build ==="
pip install --quiet pyarrow 2>&1 | tail -1
python src/sportslab/evaluation/build_team_site.py
echo "=== Build complete ==="
