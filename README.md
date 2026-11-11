# raredx

Rare-disease differential diagnosis support: from a clinical description to a ranked, explained list of candidate rare diseases.

> **Not a diagnostic tool.** raredx is a research prototype to support clinicians. It does not diagnose and must not be used for clinical decisions.

Built for the Global AI Hackathon (Geek Room x HackCulture), November 2026.

## Status

Work in progress. The data pipeline, the ontology ranker, the LLM ensemble and the learned fusion work end to end; free-text symptom extraction, explanations per candidate, live evidence with Tavily, the API and the web interface work. Final results on the held-out test sample are below.

## How it works

```
Clinical note (free text)
   v
Symptom extraction    (raredx/phenotype.py)  LLM lists symptoms -> matched to HPO terms
   v
Patient symptoms (HPO terms), reviewed by the doctor
   |-- Ontology ranker   (raredx/ranker.py)    ranks all 8,465 OMIM diseases by symptom similarity
   |-- LLM ensemble      (raredx/ensemble.py)  each LLM proposes its 10 most likely diagnoses
   v
Learned fusion (raredx/fusion.py)              LightGBM re-ranks the candidates of both sources
   v
Ranked differential diagnosis
```

- **Symptom extraction.** An LLM reads the note and lists each symptom as a standard clinical term, marking the ones the patient explicitly does not have ("hearing is normal"). Each term is matched to the Human Phenotype Ontology (HPO): by exact name or synonym first, otherwise by meaning with OpenAI embeddings over the ~41,000 HPO names and synonyms. The five closest terms are kept so the doctor can correct the match.
- **Ontology ranker.** Compares the patient's symptoms with each disease's symptoms in the Human Phenotype Ontology annotations. Rare symptoms count more than common ones (information content), and related symptoms count partially through the HPO hierarchy. No LLM involved.
- **LLM ensemble.** Each LLM receives the symptoms by readable name, plus age and sex, and returns its top 10 as JSON. Disease names are translated to OMIM codes using OMIM names, Orphanet synonyms and gene symbols (raredx/diseases.py); OMIM numbers proposed by the LLM are verified, not trusted.
- **Learned fusion.** For each patient, one row per candidate disease with what each source says about it (positions, scores, how many LLMs propose it, prevalence...). A LightGBM ranking model (lambdarank) trained on 1,000 published cases learns how much to trust each source.
- **Explanations.** For each of the top 5 candidates, from the ontology files only (instant, nothing invented): the patient's findings it explains, absent findings that argue against it, frequent findings of the disease not mentioned yet, and the genes to test. One LLM call then writes a sentence per candidate using only those facts.
- **Most useful next findings.** Among the findings of the top candidates, the ones that best separate them (expected information gain, from HPO annotation frequencies). The doctor can mark one as present or absent and the differential is recomputed.
- **Live evidence.** For the top 3 candidates, [Tavily](https://tavily.com) searches the web at the moment of the query: recent literature (PubMed, PMC, GeneReviews), clinical trials (ClinicalTrials.gov) and patient information and organisations (Orphanet, NORD, GARD). It is only shown to the doctor with its links; it never changes the ranking, so the evaluation is not affected. About 9 Tavily credits per patient, cached.
- **Gateway.** Every LLM call goes through [Bifrost](https://github.com/maximhq/bifrost) (raredx/llm.py), with an exact response cache, a hard spending limit and per-provider rate limiting for free tiers.
- **Models.** Ensemble: GPT-6 Luna (OpenAI, about 0.0005 $ per patient) and Gemini 3.5 Flash-Lite (Google AI Studio free tier). Contamination check: gpt-oss-120b (Groq free tier). Larger models were left out to keep the whole project within a small budget.
- **Privacy note.** Free tiers may use the data sent to them to improve their models. That is fine for this evaluation, which only uses already-published cases, but real patient data would require paid plans or self-hosted models.

## Evaluation

Cases come from [phenopacket-store](https://github.com/monarch-initiative/phenopacket-store) (10,374 published patients, 780 diseases, confirmed diagnoses). The evaluation is designed so the system cannot benefit from information it would not have in practice:

- **Split by paper.** Train / validation / test are split by source publication, never by patient (70 / 10 / 20 %). Samples of 1,000 / 200 / 500 cases, at most 5 per paper; the exact case lists are versioned in `results/samples/`.
- **Leakage guard.** When a case is evaluated, the ranker ignores every HPO annotation derived from that case's own paper. Without this guard the ranker's top-1 accuracy doubles (14 % -> 28 %), artificially.
- **Symptom extraction (round trip).** There are no public clinical notes with known HPO codes for these patients, so an LLM writes a short clinical note from each published case and the extractor turns it back into codes. We measure how many original symptoms come back and, above all, whether the ranker still finds the disease with the extracted codes (`results/extraction_val.json`). The same model family writes and reads the notes, so real notes may be harder.
- **Contamination.** LLMs may have memorised published cases. Commercial models were trained after the most recent case in the dataset, so contamination is checked with the open model gpt-oss-120b (cutoff July 2024). Test: 86 pairs of cases of the same disease (47 diseases), one published before the cutoff and one after. If the LLM is much better on the older case of each pair while the ontology ranker (which has no cutoff) is not, the LLM is likely remembering published cases. Result: no sign of it (see below).
- **Metrics.** Top-1 / 5 / 10 accuracy and mean reciprocal rank (MRR), with 95 % bootstrap confidence intervals. Ties are counted fairly.

### Final results on the held-out test sample (500 cases, `make test`)

These 500 patients, from papers not used anywhere else, were never used to build or tune anything. The fusion evaluated is the one the app uses (`models/fusion.pkl`). Full numbers with 95 % confidence intervals: `results/test.json`.

| Method | Top 1 | Top 5 | Top 10 | MRR |
|---|---|---|---|---|
| Ontology ranker alone | 11.8 % | 17.8 % | 21.2 % | 0.15 |
| Gemini 3.5 Flash-Lite alone | 14.4 % | 22.0 % | 24.0 % | 0.18 |
| GPT-6 Luna alone (best single source) | 15.6 % | 26.4 % | 31.8 % | 0.21 |
| Reciprocal rank fusion (no learning) | 10.0 % | 21.8 % | 28.2 % | 0.16 |
| Learned fusion, logistic regression | 16.6 % | 28.2 % | 34.4 % | 0.23 |
| **Learned fusion, LightGBM (the app)** | **26.0 %** | **36.8 %** | **41.0 %** | **0.31** |

- **The learned fusion puts the correct disease first in 26 % of patients, against 15.6 % for the best single source** (GPT-6 Luna): +10.4 points (95 % CI +7.4 to +13.6, paired bootstrap over patients, p < 0.001). In top 10: 41 % against 31.8 % (+9.2 points, CI +6.2 to +12.6).
- The gain comes from learning how to combine the sources: a fusion without learning (RRF) or a linear one (logistic regression) stays close to the best single source.
- **Ceiling:** the correct disease is among the candidates in 49.8 % of test patients; the fusion can only reorder those.
- **Confidence indicator** (`results/confidence.json`): each differential is labelled low, medium or high by how much its first candidate stands out among the ten shown. Thresholds were set on validation (three equal groups) and checked on test:

  | Level | First suggestion right (validation / test) | Correct disease in top 10 (validation / test) |
  |---|---|---|
  | Low | 4.5 % / 5.1 % | 17.9 % / 28.3 % |
  | Medium | 12.1 % / 20.8 % | 22.7 % / 34.9 % |
  | High | 52.2 % / 48.8 % | 64.2 % / 58.2 % |

  The interface shows the doctor these numbers for the patient's level: a high-confidence differential is right first about half the time, a low one almost never.
- **Reproduce everything** from the response caches with `make eval` (no new LLM calls once the caches are complete).

### Results on validation (200 cases; the test sample is kept for the final evaluation)

| Method | Top 1 | Top 5 | Top 10 | MRR |
|---|---|---|---|---|
| Ontology ranker alone | 14.0 % | 24.0 % | 24.0 % | 0.18 |
| GPT-6 Luna alone | 14.5 % | 24.5 % | 27.0 % | 0.19 |
| Gemini 3.5 Flash-Lite alone | 12.0 % | 23.0 % | 25.0 % | 0.17 |
| Reciprocal rank fusion (no learning) | 11.0 % | 24.0 % | 29.0 % | 0.17 |
| Learned fusion, logistic regression | 16.0 % | 27.0 % | 29.5 % | 0.21 |
| **Learned fusion, LightGBM** | **23.0 %** | **32.5 %** | **35.0 %** | **0.27** |
| *LightGBM with GPT-6 Luna as the only LLM* | *23.5 %* | *31.5 %* | *33.5 %* | *0.27* |

- **The learned fusion is the main gain:** top 1 goes from 14-14.5 % for the best single source to 23 %. A fusion without learning (RRF) does not get there.
- **The second LLM adds little:** with Gemini, top 10 goes from 33.5 % to 35 % and the correct disease is among the candidates in 43 % of cases instead of 41.5 %, but top 1 does not change. Gemini mostly proposes the same diseases as GPT-6 Luna (in validation it finds 7 that Luna misses, Luna 12 that Gemini misses).
- **Ablations** (`results/ablation_val.json`, top 1 with everything 23 %): without the ranker's columns 15.5 %, without the LLMs' columns 18 %. Prevalence, disease-level and patient-level columns change top 1 by at most 1 point.
- **Learning curve:** 250 / 500 / 1,000 training cases give 21 / 20 / 23 % top 1, nearly flat, so the training sample stays at 1,000 cases.
- **Ceiling:** the correct disease is among the candidates (ranker top 50 + LLM lists) in 43 % of validation cases; the fusion can only reorder those.
- **Cost and time per patient** (`results/cost_time.json`, 10 patients with real calls): about **10.5 s**, almost all of it waiting for the LLM (ranker 0.003 s, fusion 0.02 s), and **0.0005 $** with GPT-6 Luna. With several LLMs queried in parallel, time is that of the slowest one.

### Contamination check (gpt-oss-120b, 86 pairs of cases of the same disease)

`results/contamination.json`. Top 10 accuracy on cases published before and after the model's training cutoff (July 2024):

| | Before cutoff (could have been read) | After cutoff (cannot have been read) |
|---|---|---|
| gpt-oss-120b | 9.3 % | 10.5 % |
| Ontology ranker (no cutoff, difficulty control) | 29.1 % | 33.7 % |

No sign of memorisation: the LLM is not better on the cases it could have read. The test has limited power (gpt-oss-120b finds few diagnoses on these cases, and the 95 % interval of the difference is -8 to +15 points), and it cannot be run on the commercial models in the ensemble, which were trained after the most recent case in the dataset. Their results may include some memorisation; this is a limitation of every evaluation on published cases.

### Symptom extraction (round trip, 100 validation cases)

`results/extraction_val.json`, examples in `results/extraction_examples_val.json`.

| | Value |
|---|---|
| Original symptoms recovered exactly | 82.7 % (precision 82.4 %) |
| ... exactly or as a parent / child term | 92.2 % |
| Absent symptoms recovered | 87.7 % |
| Ranker top 10 with the original codes | 28 % |
| Ranker top 10 with the codes extracted from the note | 27 % |

Going through a free-text note costs almost nothing in diagnostic accuracy (top 1: 16 % -> 14 %, top 10: 28 % -> 27 %, within the confidence intervals). About 0.0002 $ per note with GPT-6 Luna.

## Quickstart

Requirements: Python 3.10+, Docker (for Bifrost), API keys for OpenAI, Google AI Studio (free), Groq (free) and Tavily (live evidence; optional).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env            # then add your API keys
make data                       # download datasets, build the case table, fetch publication dates
make split                      # train / validation / test samples
make gateway                    # start Bifrost on http://localhost:8080
                                # after editing bifrost/config.json: docker compose restart bifrost
                                # after adding keys to .env:      docker compose up -d --force-recreate bifrost
python3 -m raredx.llm ping      # check the gateway and the API key
```

Evaluation commands (results are written to `results/`):

```bash
python3 -m raredx.evaluate ranker --split val                          # ranker alone
python3 -m raredx.evaluate llm --model openai/gpt-6-luna --split val   # one LLM alone
python3 -m raredx.evaluate fusion                                      # train + evaluate the fusion
python3 -m raredx.evaluate ablation                                    # remove one group of columns at a time
python3 -m raredx.evaluate timing                                      # cost and time per patient
python3 -m raredx.evaluate contamination                               # older vs newer cases (gpt-oss-120b)
python3 -m raredx.evaluate extraction --limit 100                      # note -> symptoms round trip
```

Start the web interface (with Bifrost running), then open http://localhost:8000:

```bash
make api                        # API + interface; interactive API docs at http://localhost:8000/docs
```

The interface has three steps: write the clinical note, review the coded symptoms (each one linked to the words it came from, with alternatives and an HPO search to add findings), and the differential, with live progress and where each source ranked each disease.

Run the whole flow on a clinical note (LangGraph: extraction -> doctor review -> ranker and LLMs in parallel -> fusion):

```bash
make train                      # train the fusion on the training sample -> models/fusion.pkl
python3 -m raredx.pipeline "3-year-old girl who does not walk yet, with frequent seizures, a small head and low-set ears. Hearing and heart are normal." --sex female --age 3
```

Try the symptom extraction alone on your own text:

```bash
python3 -m raredx.phenotype "4-year-old boy with seizures since age 2, widely spaced eyes and very long fingers. Hearing is normal."
```

## Repository layout

```
raredx/
  phenotype.py  clinical note -> HPO symptoms (LLM + embeddings)
  data.py       download datasets, build the case table, fetch PubMed dates
  diseases.py   disease names <-> OMIM codes (OMIM, Orphanet, gene symbols)
  ranker.py     ontology ranker
  llm.py        Bifrost client: exact cache and spending limit
  ensemble.py   LLM prompts, answer parsing
  fusion.py     candidate tables and fusion models
  explain.py    explanations per candidate and most useful next findings
  evidence.py   live evidence from the web with Tavily
  pipeline.py   the full flow as a LangGraph graph, with a pause for the doctor's review
  api.py        FastAPI: cases, review, progress events, HPO search; serves the interface
frontend/       the web interface (plain HTML, CSS and JavaScript, no build step)
  evaluate.py   splits, metrics and every evaluation command
models/         trained fusion model (committed, so the app needs no training)
results/        versioned evaluation results and case samples
bifrost/        Bifrost gateway configuration
settings.yaml   data versions, budget, models, evaluation parameters
data/           downloaded and processed data (not in git; created by `make data`)
```

## Use of AI tools

_TBD (required by the hackathon rules)._

## Data sources and attribution

The data is not stored in this repository: `make data` downloads it from the original sources.

- **Human Phenotype Ontology (HPO)** and its disease annotations. This product uses the Human Phenotype Ontology (version recorded in the downloaded files), https://hpo.jax.org. HPO files are used unmodified and are not redistributed.
- **phenopacket-store** (GA4GH / Monarch Initiative), BSD-3-Clause.
- **Orphanet / Orphadata**, CC BY 4.0.
- **PubMed** (NCBI E-utilities), for publication dates.
- **Tavily** web search, at query time, for the live evidence shown in the interface (links to the original sources).

## License

MIT for the code in this repository. Each data source keeps its own license.
