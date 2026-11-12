#!/bin/sh
# First start: download the data (a few minutes; kept in ./data on the host, so only once).
set -e
if [ ! -f data/raw/hp.json ] || [ ! -f data/raw/phenotype.hpoa ]; then
    echo "First start: downloading the datasets (HPO, Orphanet, phenopacket-store)..."
    python -m raredx.data
fi
exec uvicorn raredx.api:app --host 0.0.0.0 --port 8000
