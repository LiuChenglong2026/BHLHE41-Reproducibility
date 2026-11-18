# Data provenance and permitted use

The original analysis code is distributed under the MIT License. That license does not relicense third-party data.

## GEO

GSE65127, GSE75819 and GSE53146 series matrices and the GPL570, GPL6884 and GPL14951 annotations were obtained from the NCBI Gene Expression Omnibus (GEO). The exact URLs, access date, byte counts and SHA-256 checksums are recorded in `data/source_manifest.json`. Users must follow NCBI/GEO terms and cite the original studies.

## Bian et al. (2019)

`Bian_2019_PMC6601371_fullText.xml` was obtained from Europe PMC (PMCID: PMC6601371). The derived CSV contains group-level values transcribed programmatically from published Tables III–IV. No donor-level gray-hair data were analyzed. Cite the source publication when using these values.

## cBioPortal: TCGA-SKCM and CCLE 2019

TCGA-SKCM (`skcm_tcga`) and CCLE 2019 (`ccle_broad_2019`) snapshots were accessed through the public cBioPortal API on 2026-11-17. Study identifiers, molecular-profile identifiers and selection rules are recorded in `data/source/cBioPortal_source_metadata.json`. These data remain subject to their original source terms, including any requirements associated with TCGA and the Cancer Cell Line Encyclopedia. The archive contains only de-identified public expression and clinical fields used in the analysis.

## Snapshot policy

The supplied snapshots are the reference inputs for the reported numerical results. A fresh upstream download is not assumed to be identical. The download and audit scripts reject missing, truncated, NUL-corrupted or checksum-mismatched files instead of silently substituting changed upstream content.
