# Cloudflare Pages build script for Sports ML Lab
# Installs minimal dependencies and builds the static team site.
#
# Usage (Cloudflare Pages settings):
#   Build command: bash cloudflare-pages.sh
#   Publish directory: site

set -e
echo "=== Sports ML Lab — Cloudflare Pages Build ==="

# Install minimal Python deps (pandas+pyarrow for parquet; numpy for player value aggregation)
echo "Installing dependencies..."
pip install --quiet pandas numpy pyarrow 2>&1 | tail -1
echo "Dependencies installed."

# Build the team site
echo "Building team site..."
python src/sportslab/evaluation/build_team_site.py

echo "=== Build complete ==="
