"""Data step: download the datasets and build the tables the rest of the project uses.

Run it with `make data` (or `python -m hoofbeat.data`). It does three things, in order:
  1. Download the raw files into data/raw/.
  2. Build data/processed/cases.jsonl: one line per patient (symptoms, diagnosis, source paper).
  3. Ask PubMed when each source paper was published -> data/processed/pmid_dates.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


# =============================================================================
# 1. DOWNLOAD
# =============================================================================

# Fixed versions so everyone (and every rerun) gets exactly the same data.
# Orphadata has no versioned download links: it publishes a new release every July
# and December, so its files may change between runs.
PHENOPACKET_STORE_VERSION = "0.1.27"
HPO_VERSION = "v2026-09-01"

SOURCES: dict[str, str] = {
    # 1. REAL PATIENTS (phenopacket-store)
    # About 10,000 patients taken from medical papers, one per file.
    # Example: patient 4 from paper PMID:23086397. Has global developmental delay
    #   (HP:0012736) and does NOT have delayed myelination (HP:0002188).
    #   Diagnosis: epileptic encephalopathy 14 (OMIM:614959).
    # What it's for:
    #   - Training the fusion model (LightGBM).
    #   - Evaluating the system: measuring how often it gets the diagnosis right.
    #   - Providing the example cases for the demo.
    "all_phenopackets.zip": f"https://github.com/monarch-initiative/phenopacket-store/releases/download/{PHENOPACKET_STORE_VERSION}/all_phenopackets.zip",

    # 2. SYMPTOM DICTIONARY (HPO ontology)
    # About 20,000 symptoms, each with a code, name, synonyms and "parents".
    # Example: HP:0001166 = Arachnodactyly. Synonyms: "spider fingers",
    #   "long slender fingers". Parent: "long fingers".
    # What it's for:
    #   - Translating what the doctor writes into HPO codes
    #     ("spider fingers" -> HP:0001166).
    #   - Letting the ranker know which symptoms are related and which are rare
    #     (rare ones count more).
    "hp.json": f"https://github.com/obophenotype/human-phenotype-ontology/releases/download/{HPO_VERSION}/hp.json",

    # 3. SYMPTOMS OF EACH DISEASE (HPO annotations)
    # A "disease -> symptom" table, with how often it appears and the paper that backs it.
    # Example: Marfan (OMIM:154700) -> HP:0000486, seen in 110 of 573 patients,
    #   according to PMID:8172269.
    # What it's for:
    #   - The core of the ranker: it compares the patient's symptoms with each disease's.
    #   - Explanations: which symptoms match and which are missing.
    #   - Picking the most useful next test.
    #   - Anti-cheating guard: dropping anything that comes from the same paper as the
    #     case being evaluated.
    "phenotype.hpoa": f"https://github.com/obophenotype/human-phenotype-ontology/releases/download/{HPO_VERSION}/phenotype.hpoa",

    # 3b. WHICH DISEASES EACH GENE CAUSES (HPO)
    # Example: gene MED13L -> OMIM:616789.
    # What it's for:
    #   - Translating disease names that LLMs give by gene ("MED13L syndrome",
    #     "KAT6B-related disorder") into OMIM codes.
    "genes_to_disease.txt": f"https://github.com/obophenotype/human-phenotype-ontology/releases/download/{HPO_VERSION}/genes_to_disease.txt",

    # 4. DISEASE CODE TRANSLATOR (Orphanet)
    # Every rare disease in Orphanet with its names and its OMIM equivalent.
    # Example: ORPHA:558 = Marfan = OMIM:154700.
    # What it's for:
    #   - Turning the disease names the LLMs return into codes, so their answers can be
    #     compared with the ranker's.
    #   - Linking each OMIM code to its Orphanet code so we can use prevalence (file 5).
    #   - Showing readable names to the doctor.
    "orphanet_product1.xml": "https://www.orphadata.com/data/xml/en_product1.xml",

    # 5. HOW COMMON EACH DISEASE IS (prevalence, Orphanet)
    # Example: Marfan affects between 1 and 5 people in every 10,000.
    # What it's for:
    #   - One more signal for the fusion model: between two candidates that match
    #     equally well, the more common one is usually more likely.
    "orphanet_product9_prev.xml": "https://www.orphadata.com/data/xml/en_product9_prev.xml",
}


def _fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hoofbeat/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)


def download_all(raw_dir: Path, force: bool = False) -> list[str]:
    """Download every source; returns the names that failed."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, url in SOURCES.items():
        dest = raw_dir / name
        if dest.exists() and not force:
            print(f"  = {name} (already present)")
            continue
        try:
            print(f"  ↓ {name}")
            _fetch(url, dest)
        except Exception as exc:  # network errors should not hide the other downloads
            print(f"  ✗ {name}: {exc}")
            failed.append(name)

    zip_path = raw_dir / "all_phenopackets.zip"
    out_dir = raw_dir / "phenopackets"
    if zip_path.exists() and (force or not out_dir.exists()):
        print("  ⇢ unzipping phenopackets")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    return failed


# =============================================================================
# 2. CASES TABLE
# Turns each patient JSON into one line of data/processed/cases.jsonl, e.g.:
#   {"case_id": "PMID_15266616_100", "pmid": "PMID:15266616",
#    "disease_id": "OMIM:147791", "disease_label": "Jacobsen syndrome",
#    "hpo_present": ["HP:0000078", ...], "hpo_excluded": ["HP:0000175", ...],
#    "sex": "FEMALE", "age": "P5Y"}
# =============================================================================

def parse_phenopacket(path: Path) -> dict | None:
    pp = json.loads(path.read_text(encoding="utf-8"))

    refs = [r["id"] for r in pp.get("metaData", {}).get("externalReferences", []) if r.get("id", "").startswith("PMID:")]
    interps = pp.get("interpretations", [])
    if len(refs) != 1 or len(interps) != 1:
        return None  # we need exactly one source publication and one diagnosis

    disease = interps[0]["diagnosis"]["disease"]
    present, excluded = [], []
    for feat in pp.get("phenotypicFeatures", []):
        (excluded if feat.get("excluded") else present).append(feat["type"]["id"])

    subject = pp.get("subject", {})
    age = subject.get("timeAtLastEncounter", {}).get("age", {}).get("iso8601duration")

    return {
        "case_id": pp["id"],
        "pmid": refs[0],
        "disease_id": disease["id"],
        "disease_label": disease.get("label", ""),
        "hpo_present": sorted(set(present)),
        "hpo_excluded": sorted(set(excluded)),
        "sex": subject.get("sex"),
        "age": age,
    }


def build_cases(raw_dir: Path, out_path: Path) -> dict:
    files = sorted((raw_dir / "phenopackets").rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No phenopackets under {raw_dir / 'phenopackets'}; run the download first.")

    rows, skipped_shape, skipped_empty = [], 0, 0
    for f in files:
        row = parse_phenopacket(f)
        if row is None:
            skipped_shape += 1
        elif not row["hpo_present"]:
            skipped_empty += 1  # nothing to rank from
        else:
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    return {
        "files": len(files),
        "cases": len(rows),
        "skipped_shape": skipped_shape,
        "skipped_no_present_hpo": skipped_empty,
        "diseases": len({r["disease_id"] for r in rows}),
        "publications": len({r["pmid"] for r in rows}),
    }


# =============================================================================
# 3. PUBLICATION DATES (PubMed)
# The dates are not in any downloaded file: we ask PubMed for each paper, e.g.
#   PMID:15266616 -> 2004-08-01
# Used to separate the cases published after each LLM was trained (contamination guard).
# Without an NCBI API key PubMed allows 3 requests/s; with NCBI_API_KEY in .env, 10/s.
# =============================================================================

ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
BATCH = 200
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _normalize(raw: str) -> str | None:
    """'2012 Nov 5' / '2012 Nov' / '2012' / '2012 Winter' -> '2012-11-05' / '2012-11-01' / '2012-01-01'."""
    m = re.match(r"(\d{4})(?:\s+([A-Za-z]{3}))?(?:\s+(\d{1,2}))?", raw or "")
    if not m:
        return None
    year, mon, day = m.group(1), MONTHS.get(m.group(2) or "", 1), int(m.group(3) or 1)
    return f"{year}-{mon:02d}-{day:02d}"


def fetch_dates(pmids: list[str], out_path: Path) -> dict:
    done: dict[str, str] = {}
    if out_path.exists():
        with open(out_path, newline="") as fh:
            done = {r["pmid"]: r["pubdate"] for r in csv.DictReader(fh)}
    todo = sorted({p for p in pmids if p not in done})

    api_key, email = os.environ.get("NCBI_API_KEY"), os.environ.get("NCBI_EMAIL")
    pause = 0.11 if api_key else 0.35

    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        params = {"db": "pubmed", "retmode": "json", "id": ",".join(p.removeprefix("PMID:") for p in chunk), "tool": "hoofbeat"}
        if api_key:
            params["api_key"] = api_key
        if email:
            params["email"] = email
        url = f"{ESUMMARY}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            result = json.load(resp)["result"]
        for uid in result.get("uids", []):
            item = result[uid]
            date = _normalize(item.get("pubdate", "")) or _normalize(item.get("epubdate", ""))
            done[f"PMID:{uid}"] = date or ""
        print(f"  pubmed {min(i + BATCH, len(todo))}/{len(todo)}")
        time.sleep(pause)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pmid", "pubdate"])
        for pmid in sorted(done):
            writer.writerow([pmid, done[pmid]])

    missing = sum(1 for p in pmids if not done.get(p))
    return {"pmids": len(set(pmids)), "missing_date": missing}


# =============================================================================
# COMMAND LINE
# =============================================================================

def _load_env(path: Path) -> None:
    """Minimal .env loader so this step runs before any dependency is installed."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--skip-pubmed", action="store_true", help="do not fetch publication dates")
    args = parser.parse_args()

    _load_env(ROOT / ".env")
    raw, processed = RAW, PROCESSED

    print("1/3 Downloading")
    failed = download_all(raw, force=args.force)

    print("2/3 Building cases table")
    stats = build_cases(raw, processed / "cases.jsonl")
    print("  " + json.dumps(stats))

    if not args.skip_pubmed:
        print("3/3 Fetching publication dates from PubMed")
        pmids = [json.loads(l)["pmid"] for l in open(processed / "cases.jsonl", encoding="utf-8")]
        print("  " + json.dumps(fetch_dates(pmids, processed / "pmid_dates.csv")))

    if failed:
        print(f"\nWARNING: failed downloads: {', '.join(failed)}")


if __name__ == "__main__":
    main()
