# raredx

Rare-disease differential diagnosis support: from a clinical description to a ranked, explained list of candidate rare diseases.

> **Not a diagnostic tool.** raredx is a research prototype to support clinicians. It does not diagnose and must not be used for clinical decisions.

Built for the Global AI Hackathon (Geek Room x HackCulture), November 2026.

## Status

Work in progress.

## How it works

_TBD_

## Results

_TBD: evaluation on phenopacket-store with leakage- and contamination-aware splits._

## Quickstart

_TBD_

## Repository layout

```
raredx/        all the code, one file per pipeline step (data.py, ranker.py, fusion.py, ...)
data/          downloaded and processed data (not in git; created by `make data`)
results/       evaluation results (versioned)
settings.yaml  paths, budget and model choices
```

## Use of AI tools

_TBD (required by the hackathon rules)._

## Data sources and attribution

The data is not stored in this repository: `make data` downloads it from the original sources.

- **Human Phenotype Ontology (HPO)** and its disease annotations. This product uses the Human Phenotype Ontology (version recorded in the downloaded files), https://hpo.jax.org. HPO files are used unmodified and are not redistributed.
- **phenopacket-store** (GA4GH / Monarch Initiative), BSD-3-Clause.
- **Orphanet / Orphadata**, CC BY 4.0.
- **PubMed** (NCBI E-utilities), for publication dates.

## License

MIT for the code in this repository. Each data source keeps its own license.
