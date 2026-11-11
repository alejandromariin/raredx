# raredx

**Differential diagnosis support for rare diseases.** A doctor writes a clinical note; raredx codes the findings, lets the doctor review them, and returns a ranked, explained list of rare diseases to consider, with the findings that would best tell them apart and live evidence from the literature.

> **Not a diagnostic tool.** raredx is a research prototype to support clinicians. It does not diagnose and must not be used for clinical decisions.

Built for the Global AI Hackathon (Geek Room x HackCulture), November 2026.

| On 500 held-out published patients | Correct disease ranked first | Correct disease in the top 10 |
|---|---|---|
| Best single source (GPT-6 Luna) | 15.6 % | 31.8 % |
| **raredx (learned fusion of ontology + 2 LLMs)** | **26.0 %** | **41.0 %** |

+10.4 points in top 1 (95 % CI +7.4 to +13.6, paired bootstrap, p < 0.001), with an evaluation built to avoid leakage from the published cases. About 12 s and 0.001 $ per patient.

## The problem

- **Rare diseases are rare one by one, not together.** More than 6,000 rare diseases affect an estimated 263 to 446 million people worldwide, 3.5 to 5.9 % of the population ([Nguengang Wakap et al., 2020](https://pubmed.ncbi.nlm.nih.gov/31527858/)).
- **Diagnosis takes years.** In the Rare Barometer survey of people living with a rare disease, the average time to diagnosis was 4.7 years ([Faye et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38755315/)).
- **No single clinician knows them all.** A pediatrician or general geneticist may see a given syndrome once in a career. What helps is a short list of plausible candidates, why each one fits, and what to check next.

**Who it is for:** clinical geneticists, pediatricians and neurologists preparing a genetic work-up or a referral. **How it is used:** the doctor pastes or writes the clinical note (any language), reviews the findings coded from it, and gets the differential in seconds, with the evidence behind each candidate and the next findings worth checking. The doctor stays in charge at every step.

## What it does

1. **Case.** A free-text note, in any language, plus age and sex.
2. **Review.** The findings read from the note, coded with the Human Phenotype Ontology (HPO). Each one is linked to the words it came from; the doctor can switch present / absent, pick another term or add one with an HPO search.
3. **Differential.** Live progress of each source (ontology ranker, each LLM, fusion, explanations, live evidence), then the ten most likely diseases with:
   - where each source ranked it, a link to OMIM and a **confidence level** measured on held-out cases;
   - for the top 5, **why**: patient findings it explains, absent findings that argue against it, frequent findings not mentioned yet, the genes to test, and a one-sentence summary;
   - **most useful next findings**: the findings that would best separate the top candidates; marking one present or absent recomputes the differential;
   - **live evidence** for the top 3 (recent literature, clinical trials, patient organisations) found at query time with Tavily.

Every piece of information says where it comes from (HPO files, an LLM, Tavily).

## What's new

Phenotype-driven tools rank diseases by comparing a patient's HPO terms with curated disease annotations; LLM-based systems, including recent agentic ones such as DeepRare ([Zhao et al., 2025](https://arxiv.org/abs/2506.20430)), propose diagnoses from the clinical description. raredx combines both and adds:

- **A learned fusion, measured against each source.** A ranking model (LightGBM, lambdarank) learns from 1,000 published cases when to trust the ontology and when to trust each LLM. It beats the best single source by 10 points in top 1 on held-out cases, while combining the same sources without learning (reciprocal rank fusion) or linearly (logistic regression) does not.
- **An evaluation designed against leakage.** Train / validation / test are split by publication, and when a case is evaluated the ontology ranker ignores every annotation curated from that case's own paper. Without this guard the ranker's top-1 accuracy doubles (14 % -> 28 % on validation): many disease annotations come from the very papers the test cases are taken from. LLM memorisation is checked with an open model whose training cutoff falls inside the dataset.
- **The next most useful finding, as an interactive loop.** Expected information gain over the top candidates, from HPO annotation frequencies, turns the list into a guided work-up: check this finding, mark it, see the differential update.
- **A calibrated confidence level.** Low / medium / high, with what that level achieved on held-out patients (first suggestion right about half the time when high, almost never when low).
- **Fully reproducible and cheap.** Open source, every result regenerated with one command from cached responses, about 0.001 $ per patient with a small paid model and a free one.

## How it works

```
Clinical note (free text)
   v
Symptom extraction   phenotype.py   LLM lists findings -> matched to HPO terms (exact name, then embeddings)
   v
Doctor review        pipeline.py    the LangGraph flow pauses until the doctor confirms the findings
   v
   |-- Ontology ranker   ranker.py     ranks all 8,465 OMIM diseases by HPO similarity (no LLM)
   |-- LLM ensemble      ensemble.py   GPT-6 Luna and Gemini 3.5 Flash-Lite, 10 diagnoses each, in parallel
   v
Learned fusion       fusion.py      LightGBM re-ranks the candidates of all sources + confidence level
   v
Explanations         explain.py     fits / against / typical / genes from HPO files; next most useful findings
   v
   |-- Summary           explain.py    one LLM call, a sentence per candidate, only from the facts above
   |-- Live evidence     evidence.py   Tavily: literature, trials, patient organisations (top 3)
   v
Ranked, explained differential (api.py + frontend/)
```

- **Symptom extraction.** An LLM reads the note and lists each finding as a standard clinical term, marking the ones the patient explicitly does not have ("hearing is normal"). Each term is matched to HPO by exact name or synonym first, otherwise by meaning with OpenAI embeddings over the ~41,000 HPO names and synonyms; the five closest terms are kept so the doctor can correct the match.
- **Ontology ranker.** Compares the patient's findings with each disease's HPO annotations. Rare findings count more than common ones (information content), and related findings count partially through the HPO hierarchy, minus a penalty for findings the patient does not have.
- **LLM ensemble.** Each LLM receives the findings by readable name, plus age and sex, and returns its top 10 as JSON. Disease names are translated to OMIM codes using OMIM names, Orphanet synonyms and gene symbols (`diseases.py`); OMIM numbers proposed by an LLM are verified, not trusted.
- **Learned fusion.** For each patient, one row per candidate disease with what each source says about it (positions, scores, how many LLMs propose it, prevalence, ...). LightGBM learns from 1,000 training cases how much to trust each source.
- **Explanations and next findings.** From the HPO files only, so they are instant and cannot invent anything. The summary LLM sees only those facts.
- **Live evidence.** Shown to the doctor with its links; it never changes the ranking, so it does not affect the evaluation.
- **Gateway.** Every LLM call goes through [Bifrost](https://github.com/maximhq/bifrost) (`llm.py`), one OpenAI-compatible endpoint for OpenAI, Google AI Studio and Groq, with an exact response cache, a hard spending limit, per-provider rate limits for free tiers, and fast failure in the app: if one LLM does not answer, the differential is computed without it and the doctor is told.

## Evaluation

Cases come from [phenopacket-store](https://github.com/monarch-initiative/phenopacket-store) (10,374 published patients, 780 diseases, confirmed diagnoses, GA4GH phenopackets). The evaluation is designed so the system cannot benefit from information it would not have in practice:

- **Split by paper.** Train / validation / test are split by source publication, never by patient (70 / 10 / 20 %). Samples of 1,000 / 200 / 500 cases, at most 5 per paper; the exact case lists are versioned in `results/samples/`.
- **Leakage guard.** When a case is evaluated, the ranker ignores every HPO annotation derived from that case's own paper. Without this guard the ranker's top-1 accuracy doubles (14 % -> 28 %), artificially.
- **Contamination.** LLMs may have memorised published cases; checked with the open model gpt-oss-120b, whose training cutoff (July 2024) falls inside the dataset (see below).
- **Symptom extraction.** A round trip: an LLM writes a clinical note from each published case and the extractor turns it back into codes (see below).
- **Metrics.** Top-1 / 5 / 10 accuracy and mean reciprocal rank (MRR), with 95 % bootstrap confidence intervals; ties are counted fairly; methods are compared patient by patient (paired bootstrap).
- **One command.** `make eval` regenerates every result in `results/` from the cached responses; `make test` runs only the final test.

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

## Cost and time

Measured on the three illustrative cases of the interface with all caches off, so every call is real (`python -m raredx.evaluate app-timing`, `results/cost_time_app.json`):

| Step | Time per patient |
|---|---|
| Note -> findings ready for review (LLM extraction + HPO matching) | 5.6 s |
| "Compute differential" -> full result (ranker, two LLMs in parallel, fusion, explanations, summary and Tavily in parallel) | 12.0 s |
| of which: ranker / fusion / explanations | < 0.2 s together |

- **Cost: about 0.0009 $ per patient** (GPT-6 Luna for extraction, diagnosis and summary, plus OpenAI embeddings). Gemini 3.5 Flash-Lite runs on the Google AI Studio free tier. Tavily: 9 searches (credits) per patient.
- **The whole project** (all evaluations, about 4,000 LLM calls) cost under 1 $ of OpenAI usage.

## Limitations

- **Published cases are not real notes.** Patients come from papers, and the notes used to test extraction were written by an LLM. Real clinical notes are messier.
- **The ceiling is the candidate list.** The correct disease is among the candidates in about half of test patients; the fusion can only reorder them.
- **Memorisation cannot be ruled out** for the commercial LLMs, trained after the most recent case in the dataset; the check on an older open model found no sign of it, with limited statistical power.
- **Phenotype only.** No genomic data; in practice a differential like this guides which genetic test to order, it does not replace it.
- **Privacy.** The free tiers used here may use submitted data to improve their models: fine for published cases, not for real patients. The app warns not to enter identifying data.
- **Not validated clinically, not a medical device.**

## Roadmap

1. **Private deployment:** self-hosted open models behind the same Bifrost gateway (no code change, only `settings.yaml`), so patient notes never leave the hospital.
2. **Genomics:** combine the differential with exome or genome variants (e.g. Exomiser) to rank candidate genes, the step that confirms most diagnoses.
3. **Larger candidate pool:** more diverse LLMs and retrieval over GeneReviews to raise the ceiling.
4. **Standards:** read and write cases as GA4GH phenopackets to plug into existing clinical pipelines.
5. **Clinical evaluation:** retrospective study on de-identified notes with a genetics department, then a prospective one.

## Quickstart

Requirements: Python 3.10+, Docker (for Bifrost), API keys for OpenAI, Google AI Studio (free), Groq (free; contamination check only) and Tavily (live evidence; optional).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env            # then add your API keys
make data                       # download datasets, build the case table, fetch publication dates
make split                      # train / validation / test samples
make gateway                    # start Bifrost on http://localhost:8080
python3 -m raredx.llm ping      # check the gateway and the API key
make api                        # API + interface on http://localhost:8000 (API docs at /docs)
```

The trained fusion model and its confidence levels are in `models/`, so the app works without training. To retrain and reproduce the evaluation:

```bash
make train                      # fusion on the training sample -> models/fusion.pkl
make test                       # final evaluation on the 500 test cases -> results/test.json
make eval                       # every evaluation (needs the LLM answers: cached, or about 4,000 calls)
```

Other commands:

```bash
python3 -m raredx.pipeline "3-year-old girl who does not walk yet, with frequent seizures, a small head and low-set ears." --sex female --age 3
python3 -m raredx.phenotype "4-year-old boy with seizures since age 2, widely spaced eyes and very long fingers."
python3 -m raredx.evidence "Rett syndrome"
python3 -m raredx.evaluate llm --model openai/gpt-6-luna --split val    # one LLM alone
```

Bifrost: after editing `bifrost/config.json`, `docker compose restart bifrost`; after adding keys to `.env`, `docker compose up -d --force-recreate bifrost`.

## Repository layout

```
raredx/
  data.py       download datasets, build the case table, fetch PubMed dates
  phenotype.py  clinical note -> HPO findings (LLM + embeddings)
  ranker.py     ontology ranker
  diseases.py   disease names <-> OMIM codes (OMIM, Orphanet, gene symbols)
  llm.py        Bifrost client: exact cache, spending limit, retries
  ensemble.py   LLM prompts, answer parsing
  fusion.py     candidate tables and fusion models
  explain.py    explanations per candidate and most useful next findings
  evidence.py   live evidence from the web with Tavily
  pipeline.py   the full flow as a LangGraph graph, with a pause for the doctor's review
  api.py        FastAPI: cases, review, progress events, HPO search; serves the interface
  evaluate.py   splits, metrics and every evaluation command
frontend/       the web interface (plain HTML, CSS and JavaScript, no build step)
models/         trained fusion model and confidence levels (committed)
results/        versioned evaluation results and case samples
bifrost/        Bifrost gateway configuration
settings.yaml   data versions, budget, models, evaluation parameters
data/           downloaded and processed data (not in git; created by `make data`)
```

## Use of AI tools

- **Inside the product:** GPT-6 Luna (OpenAI) extracts findings, proposes diagnoses and writes the summaries; Gemini 3.5 Flash-Lite (Google) proposes diagnoses; OpenAI embeddings match findings to HPO terms; gpt-oss-120b (Groq) is used only for the contamination check; LightGBM is trained on published cases; Tavily provides the live evidence. All LLM calls go through Bifrost.
- **To build it:** the project was developed with Claude (Anthropic) as an AI pair programmer, used for design discussions, writing and reviewing code and documentation, and running checks. The author chose the problem and the approach, reviewed the code, ran every experiment on his machine, and is responsible for the result.

## Data sources and attribution

The data is not stored in this repository: `make data` downloads it from the original sources.

- **Human Phenotype Ontology (HPO)** and its disease annotations. This product uses the Human Phenotype Ontology (version recorded in the downloaded files), https://hpo.jax.org. HPO files are used unmodified and are not redistributed.
- **phenopacket-store** (GA4GH / Monarch Initiative), BSD-3-Clause.
- **Orphanet / Orphadata**, CC BY 4.0.
- **PubMed** (NCBI E-utilities), for publication dates.
- **Tavily** web search, at query time, for the live evidence shown in the interface (links to the original sources).

## License

MIT for the code in this repository. Each data source keeps its own license.
