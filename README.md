# BHLHE41 lightweight code-and-results supplement

This supplement accompanies **“BHLHE41 and Melanocyte-Lineage Programs Across Vitiligo, Hair Graying and Melanoma: A Public-Transcriptome Study.”** It is designed to remain below the journal's 4 MB supplementary-ZIP limit without changing the analysis code, locked model or reported numerical results.

The study is a secondary analysis of public transcriptomes and published aggregate tables. It does not establish melanocyte abundance, causal regulation, calibrated diagnosis, prognosis or clinical utility.

## Included

- All analysis and figure-generation scripts and the one-command driver.
- Exact direct and transitive software-version pins.
- Public-source URLs, access dates, byte counts and SHA-256 checksums.
- Compact reference cBioPortal expression/clinical snapshots and the Bian full-text XML, included to prevent changing upstream responses from altering an exact reproduction.
- Development sample metadata and final study-specific scaler parameters.
- Twenty-five-fold assignments, LASSO tuning, feature ranks, Borda scores, fold coefficients, training-fold top-500 statistics and strict out-of-fold predictions.
- Final locked six-gene model, selection stability and full-data consensus ranks.
- External probe mapping, sample-level scores, AUC, effect sizes and bootstrap intervals.
- Cross-cohort paired differential-expression results, replicated-gene tables and BHLHE41 analyses.
- Gray-hair comparisons, CCLE/TCGA result tables and the independent 121-check audit report.
- Provenance hashes linking this supplement to the verified full analysis archive.

## Deliberately excluded to meet the size limit

Large GEO series-matrix and platform snapshots, large paired-delta matrices, full per-cohort DE tables, the 10 MB training-fold scaler matrix and figures are not duplicated here. The workflow retrieves or regenerates them. Compact cBioPortal and Bian source snapshots are included. Figures are submitted separately with the manuscript.

`FULL_ARCHIVE_PROVENANCE.json` records the full-archive and key-model hashes. The included audit report documents the completed reference run; it is not a claim that this lightweight ZIP contains the large raw inputs.

## Reproduce the complete analysis

The first run requires internet access to retrieve the public inputs:

```bash
python3.12 -m pip install -r requirements-lock.txt
python3 run_analysis.py --download
```

The downloader retrieves missing large GEO series matrices and platform annotations. Compact cBioPortal expression/clinical snapshots and the Bian full-text XML are included, so those analyses use the reference inputs without depending on a changed API response. The gray-hair script reconstructs the published table extract. Every downloaded or derived source must match the archived SHA-256 manifest. If an upstream resource has changed, the workflow stops rather than silently calling the output an exact reproduction.

After the public inputs have been obtained, subsequent runs are offline:

```bash
python3 run_analysis.py
python3 scripts/09_checksums.py --verify
```

The complete workflow ends with the independent audit and a newly generated `results/core_output_checksums.sha256`. The original large training-fold scaler matrix is regenerated during this process.

## Key files

| File | Contents |
|---|---|
| `scripts/03_machine_learning.py` | Fold-isolated scaling, screening, selection and coefficient fitting |
| `results/machine_learning/leave_one_pair_out_predictions.csv` | Fifty strict out-of-fold scores |
| `results/machine_learning/leave_one_pair_out_training_top500_statistics.csv.gz` | Training-only top-500 statistics for every fold |
| `results/machine_learning/final_signature_model.json` | Locked genes, coefficients, intercept and processing rule |
| `results/external_validation/GSE53146_validation_summary.json` | External AUC, effect sizes, exact test and intervals |
| `results/audit/reproducibility_audit.json` | Reference independent audit: 121 checks passed |

## License and disclosure

Original analysis code is MIT-licensed; third-party data remain under their source terms. See `LICENSE` and `DATA_NOTICES.md`. During manuscript revision, Kimi (Moonshot AI) assisted with code refactoring, reproducibility checks, numerical cross-checks and language editing. The authors remain responsible for verification and final content.
