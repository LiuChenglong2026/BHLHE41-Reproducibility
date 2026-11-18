#!/usr/bin/env python3
"""Download missing public inputs or verify the archived source snapshots.

The archived snapshots are the reference inputs for the reported results. Fresh
upstream downloads must match their recorded SHA-256 checksums; this prevents a
silent change in a remote annotation or API response from being mistaken for a
reproduction of the manuscript.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import requests
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "source_manifest.json"
ACCESS_DATE = "2026-11-17"
NCBI_PLATFORM = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl}&targ=self&form=text&view=full"

STATIC_SOURCES = [
    {"path":"data/raw/GSE65127_series_matrix.txt.gz", "kind":"series_matrix",
     "url":"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65127/matrix/GSE65127_series_matrix.txt.gz",
     "role":"GSE65127 development cohort series matrix"},
    {"path":"data/raw/GSE75819_series_matrix.txt.gz", "kind":"series_matrix",
     "url":"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75819/matrix/GSE75819_series_matrix.txt.gz",
     "role":"GSE75819 development cohort series matrix"},
    {"path":"data/raw/GSE53146_series_matrix.txt.gz", "kind":"series_matrix",
     "url":"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE53nnn/GSE53146/matrix/GSE53146_series_matrix.txt.gz",
     "role":"GSE53146 external-validation series matrix"},
    {"path":"data/annotations/GPL570.annot.gz", "kind":"gzip_annotation",
     "url":"https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz",
     "role":"GPL570 probe annotation"},
    {"path":"data/annotations/GPL6884.txt", "kind":"platform_text",
     "url":NCBI_PLATFORM.format(gpl="GPL6884"), "role":"GPL6884 full platform table"},
    {"path":"data/annotations/GPL14951.txt", "kind":"platform_text",
     "url":NCBI_PLATFORM.format(gpl="GPL14951"), "role":"GPL14951 complete platform table"},
    {"path":"data/source/Bian_2019_PMC6601371_fullText.xml", "kind":"xml",
     "url":"https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6601371/fullTextXML",
     "role":"Bian et al. full-text XML for Tables III–IV"},
]

DERIVED_SOURCES = [
    {"path":"data/source/Bian_2019_Tables_III_IV_40_unique_genes.csv", "kind":"file",
     "url":"Derived by scripts/05_graying_concordance.py from PMC6601371 Tables III–IV",
     "role":"Forty unique published gray-hair genes"},
    {"path":"data/source/TCGA_SKCM_cBioPortal_expression_RSEM.csv", "kind":"file",
     "url":"https://www.cbioportal.org/api; study=skcm_tcga; profile=skcm_tcga_rna_seq_v2_mrna",
     "role":"TCGA-SKCM expression snapshot"},
    {"path":"data/source/TCGA_SKCM_cBioPortal_patient_clinical.csv", "kind":"file",
     "url":"https://www.cbioportal.org/api/studies/skcm_tcga/clinical-data?clinicalDataType=PATIENT",
     "role":"TCGA-SKCM patient clinical snapshot"},
    {"path":"data/source/CCLE_2019_cutaneous_melanoma_expression_RPKM.csv", "kind":"file",
     "url":"https://www.cbioportal.org/api; study=ccle_broad_2019; profile=ccle_broad_2019_rna_seq_mrna",
     "role":"CCLE cutaneous-melanoma expression snapshot"},
    {"path":"data/source/CCLE_2019_cutaneous_melanoma_sample_annotations.csv", "kind":"file",
     "url":"https://www.cbioportal.org/api/studies/ccle_broad_2019/clinical-data?clinicalDataType=SAMPLE",
     "role":"CCLE sample-selection annotations"},
    {"path":"data/source/cBioPortal_source_metadata.json", "kind":"file",
     "url":"Recorded from the cBioPortal requests in scripts/06_cbioportal_melanoma.py",
     "role":"cBioPortal study, profile, selection and access-date metadata"},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_structure(path: Path, kind: str) -> None:
    """Reject truncated or corrupt downloads before parsing."""
    if path.stat().st_size == 0:
        raise ValueError(f"Empty source file: {path}")
    if kind == "series_matrix":
        with gzip.open(path, "rt") as handle:
            text = handle.read()
        if "!series_matrix_table_begin" not in text or "!series_matrix_table_end" not in text:
            raise ValueError(f"Incomplete GEO series matrix: {path}")
    elif kind == "gzip_annotation":
        with gzip.open(path, "rt", errors="strict") as handle:
            lines = handle.read().splitlines()
        if len(lines) < 29 or "ID" not in lines[27] or "Gene symbol" not in lines[27]:
            raise ValueError(f"Unexpected or truncated GPL570 annotation: {path}")
    elif kind == "platform_text":
        text = path.read_text(errors="strict")
        if "\x00" in text:
            raise ValueError(f"NUL bytes indicate a corrupt platform table: {path}")
        lines = text.splitlines()
        if "!platform_table_begin" not in lines or "!platform_table_end" not in lines:
            raise ValueError(f"Incomplete platform table: {path}")
    elif kind == "xml":
        root = etree.parse(str(path))
        if not root.xpath(".//table-wrap"):
            raise ValueError(f"No tables found in source XML: {path}")


def record_for(definition: dict) -> dict:
    path = ROOT / definition["path"]
    validate_structure(path, definition["kind"])
    return {**definition, "access_date":ACCESS_DATE,
            "bytes":path.stat().st_size, "sha256":sha256(path)}


def write_manifest() -> None:
    records = [record_for(item) for item in STATIC_SOURCES + DERIVED_SOURCES]
    payload = {"manifest_version":1,
               "description":"Reference input snapshots for the reported analysis; SHA-256 verification is mandatory.",
               "sources":records}
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {MANIFEST} with {len(records)} source records")


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing {MANIFEST}; use the supplied reference archive")
    return json.loads(MANIFEST.read_text())


def verify_records(records: list[dict], require_all: bool=True) -> list[str]:
    messages = []
    failures = []
    for item in records:
        path = ROOT / item["path"]
        if not path.exists():
            if require_all:
                failures.append(f"MISSING {item['path']}")
            continue
        validate_structure(path, item["kind"])
        actual_bytes = path.stat().st_size
        actual_hash = sha256(path)
        if actual_bytes != item["bytes"] or actual_hash != item["sha256"]:
            failures.append(f"CHECKSUM {item['path']} expected {item['sha256']} but found {actual_hash}")
        else:
            messages.append(f"OK {item['path']} {actual_hash}")
    if failures:
        raise RuntimeError("Source verification failed:\n" + "\n".join(failures) +
                           "\nUse the archived snapshots for exact reproduction; do not silently substitute changed upstream files.")
    return messages


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent":"BHLHE41-public-transcriptome-reproduction/1.0"}
    for attempt in range(3):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30,300)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
            partial.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true", help="Download only missing static GEO/Europe PMC inputs, then verify them")
    group.add_argument("--verify-only", action="store_true", help="Verify every archived input, including cBioPortal snapshots (default)")
    group.add_argument("--write-manifest", action="store_true", help="Maintainer action: record checksums of the current reference inputs")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    manifest = load_manifest()
    if args.download:
        by_path = {item["path"]:item for item in manifest["sources"]}
        for definition in STATIC_SOURCES:
            path = ROOT / definition["path"]
            if not path.exists():
                print(f"Downloading {definition['url']}")
                download(definition["url"], path)
            expected = by_path.get(definition["path"])
            if expected is None:
                raise RuntimeError(f"Manifest lacks {definition['path']}")
            verify_records([expected])
            print(f"Verified {definition['path']}")
        print("Static public inputs are present and match the reference manifest. cBioPortal snapshots are checked by the full audit.")
    else:
        messages = verify_records(manifest["sources"], require_all=True)
        print("\n".join(messages))
        print(f"Verified {len(messages)} source snapshots against {MANIFEST}")


if __name__ == "__main__":
    main()
