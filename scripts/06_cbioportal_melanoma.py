#!/usr/bin/env python3
"""Fetch (if needed) and analyze cBioPortal TCGA-SKCM and CCLE 2019 data."""
import json, time
from datetime import date
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from geo_io import ROOT

SOURCE = ROOT / "data" / "source"
OUT = ROOT / "results" / "melanoma"
SOURCE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://www.cbioportal.org/api"
GENES = {"BHLHE41":79365,"MITF":4286,"KIT":3815,"AURKA":6790,"TYR":7299,
         "TYRP1":7306,"DCT":1638,"MLANA":2315,"PMEL":6490,"CA14":23632,"TUBB2B":347733}


def request(method, endpoint, **kwargs):
    import requests
    for attempt in range(5):
        response = requests.request(method, BASE + endpoint, timeout=120, **kwargs)
        if response.status_code == 429:
            time.sleep(20*(attempt+1)); continue
        response.raise_for_status(); return response.json()
    raise RuntimeError("cBioPortal rate limit persisted")


def fetch_expression(study, profile, sample_ids=None):
    sample_list = request("GET", "/sample-lists/" + profile)
    ids = sample_list["sampleIds"] if sample_ids is None else sample_ids
    rev = {v:k for k,v in GENES.items()}
    payload = {"entrezGeneIds": list(GENES.values())}
    payload["sampleIds" if sample_ids is not None else "sampleListId"] = ids if sample_ids is not None else profile
    data = request("POST", "/molecular-profiles/" + profile + "/molecular-data/fetch", json=payload)
    rows = [{"sample":x["sampleId"],"patient":x["patientId"],"gene":rev[x["entrezGeneId"]],"value":x["value"]}
            for x in data if x.get("value") is not None]
    return pd.DataFrame(rows).pivot(index="sample", columns="gene", values="value")


def ensure_sources():
    tcga_path = SOURCE / "TCGA_SKCM_cBioPortal_expression_RSEM.csv"
    ccle_path = SOURCE / "CCLE_2019_cutaneous_melanoma_expression_RPKM.csv"
    if not tcga_path.exists():
        fetch_expression("skcm_tcga", "skcm_tcga_rna_seq_v2_mrna").to_csv(tcga_path, index_label="sample")
    patient_path = SOURCE / "TCGA_SKCM_cBioPortal_patient_clinical.csv"
    if not patient_path.exists():
        pd.DataFrame(request("GET", "/studies/skcm_tcga/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY&pageSize=100000")).to_csv(patient_path, index=False)
    annotations_path = SOURCE / "CCLE_2019_cutaneous_melanoma_sample_annotations.csv"
    if not ccle_path.exists() or not annotations_path.exists():
        clinical = pd.DataFrame(request("GET", "/studies/ccle_broad_2019/clinical-data?clinicalDataType=SAMPLE&projection=SUMMARY&pageSize=100000"))
        key = clinical[clinical.clinicalAttributeId == "CANCER_TYPE_DETAILED"].set_index("sampleId").value
        selected = key[key == "Cutaneous Melanoma"].index.tolist()
        if not ccle_path.exists():
            fetch_expression("ccle_broad_2019", "ccle_broad_2019_rna_seq_mrna", selected).to_csv(ccle_path, index_label="sample")
        if not annotations_path.exists():
            clinical[clinical.sampleId.isin(selected)].to_csv(annotations_path, index=False)


def main():
    ensure_sources()
    # CCLE 2019 RNA-seq, all 58 cutaneous melanoma samples with expression.
    ccle = pd.read_csv(SOURCE / "CCLE_2019_cutaneous_melanoma_expression_RPKM.csv", index_col=0)
    ccle_log = np.log2(np.maximum(ccle, 0) + 1)
    rows = []
    for gene in ["MITF","DCT","MLANA","TYRP1","PMEL","TYR","KIT","AURKA"]:
        pair = ccle_log[["BHLHE41", gene]].dropna()
        result = stats.spearmanr(pair.BHLHE41, pair[gene])
        rows.append({"gene": gene, "n": len(pair), "rho": result.statistic, "p": result.pvalue})
    corr = pd.DataFrame(rows); corr["FDR_BH"] = multipletests(corr.p, method="fdr_bh")[1]
    corr.to_csv(OUT / "CCLE_2019_58_melanoma_BHLHE41_correlations.csv", index=False)
    high = ccle.MITF > ccle.MITF.median()
    hi, lo = ccle.loc[high,"BHLHE41"], ccle.loc[~high,"BHLHE41"]
    mw = stats.mannwhitneyu(hi, lo, alternative="two-sided", method="asymptotic")
    ccle_summary = {"study": "ccle_broad_2019", "profile": "ccle_broad_2019_rna_seq_mrna",
                    "n_cutaneous_melanoma_RNAseq": len(ccle), "MITF_median": float(ccle.MITF.median()),
                    "BHLHE41_mean_MITF_high_RPKM": float(hi.mean()), "BHLHE41_mean_MITF_low_RPKM": float(lo.mean()),
                    "mean_fold_ratio": float(hi.mean()/lo.mean()), "mann_whitney_two_sided_asymptotic_p": float(mw.pvalue)}
    (OUT / "CCLE_2019_summary.json").write_text(json.dumps(ccle_summary, indent=2))
    ccle_log.to_csv(OUT / "CCLE_2019_58_melanoma_log2_expression.csv", index_label="sample")

    # TCGA-SKCM legacy study: 472 expression samples; one sorted sample per patient for survival.
    tcga = pd.read_csv(SOURCE / "TCGA_SKCM_cBioPortal_expression_RSEM.csv", index_col=0)
    tcga.index = tcga.index.astype(str)
    log = np.log2(np.maximum(tcga, 0) + 1)
    program_genes = ["MITF","DCT","MLANA","TYRP1","PMEL","TYR"]
    log["melanocytic_program"] = log[program_genes].apply(lambda c:(c-c.mean())/c.std(ddof=1)).mean(axis=1)
    rows = []
    for predictor in ["BHLHE41", "AURKA"]:
        for target in program_genes + ["melanocytic_program"]:
            pair = log[[predictor, target]].dropna(); result = stats.spearmanr(pair[predictor], pair[target])
            rows.append({"predictor": predictor, "target": target, "n": len(pair),
                         "rho": result.statistic, "p": result.pvalue})
    result = stats.spearmanr(log.BHLHE41, log.AURKA)
    rows.append({"predictor":"BHLHE41","target":"AURKA","n":len(log),"rho":result.statistic,"p":result.pvalue})
    pd.DataFrame(rows).to_csv(OUT / "TCGA_SKCM_472_sample_correlations.csv", index=False)

    patient = pd.read_csv(SOURCE / "TCGA_SKCM_cBioPortal_patient_clinical.csv")
    os = patient.pivot(index="patientId", columns="clinicalAttributeId", values="value")[["OS_STATUS","OS_MONTHS"]]
    os["time_months"] = pd.to_numeric(os.OS_MONTHS, errors="coerce")
    os["event"] = os.OS_STATUS.map({"1:DECEASED":1,"0:LIVING":0})
    surv = log.copy(); surv["patient"] = surv.index.str[:12]
    surv = surv.sort_index().drop_duplicates("patient", keep="first").join(os[["time_months","event"]], on="patient")
    surv = surv.dropna(subset=["time_months","event"]); surv = surv[surv.time_months >= 0].copy()
    surv["MITF_group"] = np.where(surv.MITF > surv.MITF.median(), "High", "Low")
    results, km_rows = [], []
    for gene in ["BHLHE41", "AURKA", "melanocytic_program"]:
        surv["z"] = (surv[gene]-surv[gene].mean())/surv[gene].std(ddof=1)
        fit = CoxPHFitter().fit(surv[["time_months","event","z"]], "time_months", "event")
        s = fit.summary.loc["z"]
        median = surv[gene].median(); high = surv[gene] > median
        lr = logrank_test(surv.time_months[high], surv.time_months[~high],
                          surv.event[high], surv.event[~high])
        results.append({"gene":gene,"n":len(surv),"events":int(surv.event.sum()),
                        "HR_per_SD":s["exp(coef)"],"HR95_low":s["exp(coef) lower 95%"],
                        "HR95_high":s["exp(coef) upper 95%"],"cox_p":s["p"],
                        "median_log2_expression_or_score":median,"n_high":int(high.sum()),
                        "median_split_logrank_p":float(lr.p_value)})
        surv[gene+"_median_group"] = np.where(high, "High", "Low")
    summary = pd.DataFrame(results)
    summary.to_csv(OUT / "TCGA_SKCM_survival_summary.csv", index=False)
    surv.to_csv(OUT / "TCGA_SKCM_459_patient_survival_analysis.csv", index_label="sample")
    meta = {"study":"skcm_tcga","profile":"skcm_tcga_rna_seq_v2_mrna","expression_samples":len(tcga),
            "unique_patients":int(tcga.index.str[:12].nunique()),"survival_patients":len(surv),
            "events":int(surv.event.sum()),"expression_transform":"log2(max(RSEM,0)+1)",
            "survival_sample_rule":"one sample per patient, first after sorting sample ID; OS_MONTHS >= 0 and known OS_STATUS",
            "analysis":"univariable Cox per SD and secondary median-split log-rank; exploratory and unadjusted"}
    (OUT / "TCGA_SKCM_analysis_metadata.json").write_text(json.dumps(meta, indent=2))
    source_meta_path = SOURCE / "cBioPortal_source_metadata.json"
    if not source_meta_path.exists():
        source_meta = {"access_date":date.today().isoformat(), "api":BASE,
                       "TCGA":{"study":"skcm_tcga", "molecular_profile":"skcm_tcga_rna_seq_v2_mrna",
                               "sample_list":"skcm_tcga_rna_seq_v2_mrna", "expression_samples":len(tcga)},
                       "CCLE":{"study":"ccle_broad_2019", "molecular_profile":"ccle_broad_2019_rna_seq_mrna",
                               "sample_list":"ccle_broad_2019_rna_seq_mrna",
                               "selection":"CANCER_TYPE_DETAILED == Cutaneous Melanoma and available RNA-seq expression",
                               "expression_samples":len(ccle)}}
        source_meta_path.write_text(json.dumps(source_meta, indent=2))
    print(json.dumps({"CCLE":ccle_summary,"TCGA":meta,"survival":results}, indent=2, default=float))


if __name__ == "__main__": main()
