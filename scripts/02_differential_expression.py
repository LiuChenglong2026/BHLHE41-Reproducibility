#!/usr/bin/env python3
"""Study-stratified paired differential expression and BHLHE41 analyses."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from geo_io import ROOT, load_development, paired_deltas, classifier_data

OUT = ROOT / "results" / "differential_expression"
INTER = ROOT / "results" / "intermediate"
for p in [OUT, INTER]: p.mkdir(parents=True, exist_ok=True)


def paired_test(delta):
    n = delta.shape[1]
    mean = delta.mean(axis=1)
    sd = delta.std(axis=1, ddof=1)
    se = sd / np.sqrt(n)
    test = stats.ttest_1samp(delta.to_numpy(), 0, axis=1, nan_policy="omit")
    q = multipletests(test.pvalue, method="fdr_bh")[1]
    return pd.DataFrame({"log2FC": mean, "sd_paired_delta": sd, "se": se,
                         "t": test.statistic, "p": test.pvalue, "FDR_BH": q}, index=delta.index)


def main():
    a, b, m1, m2 = load_development()
    pd.concat([m1, m2]).to_csv(INTER / "development_sample_metadata.csv", index_label="sample")
    d1, pairs1 = paired_deltas(a, m1[m1.status.isin(["Lesional", "NonLesional"])])
    d2, pairs2 = paired_deltas(b, m2)
    r1, r2 = paired_test(d1), paired_test(d2)
    r1.to_csv(OUT / "GSE65127_paired_DE_all_genes.csv", index_label="gene")
    r2.to_csv(OUT / "GSE75819_paired_DE_all_genes.csv", index_label="gene")
    d1.to_csv(INTER / "GSE65127_paired_deltas.csv.gz", index_label="gene", compression="gzip")
    d2.to_csv(INTER / "GSE75819_paired_deltas.csv.gz", index_label="gene", compression="gzip")
    common = r1.index.intersection(r2.index)
    joined = r1.loc[common].add_prefix("GSE65127_").join(r2.loc[common].add_prefix("GSE75819_"))
    joined["sample_size_weighted_log2FC"] = (10*joined.GSE65127_log2FC + 15*joined.GSE75819_log2FC)/25
    joined["same_direction"] = np.sign(joined.GSE65127_log2FC) == np.sign(joined.GSE75819_log2FC)
    joined["replicated_FDR05_same_direction"] = ((joined.GSE65127_FDR_BH < .05) &
                                                   (joined.GSE75819_FDR_BH < .05) & joined.same_direction)
    joined["replicated_large_effect"] = (joined.replicated_FDR05_same_direction &
                                          (joined.GSE65127_log2FC.abs() > 1) & (joined.GSE75819_log2FC.abs() > 1))
    joined.sort_values(["replicated_FDR05_same_direction", "GSE65127_FDR_BH", "GSE75819_FDR_BH"],
                       ascending=[False, True, True]).to_csv(OUT / "cross_cohort_paired_DE.csv", index_label="gene")
    joined[joined.replicated_FDR05_same_direction].to_csv(OUT / "replicated_14_genes.csv", index_label="gene")
    joined[joined.replicated_large_effect].to_csv(OUT / "replicated_large_effect_6_genes.csv", index_label="gene")
    expr, cm = classifier_data()
    expr.to_csv(INTER / "classifier_unscaled_13511_genes_50_samples.csv.gz", index_label="gene", compression="gzip")

    # BHLHE41 is only measurable in GSE65127.
    bmeta = m1.copy()
    bmeta["expression"] = a.loc["BHLHE41", bmeta.index].to_numpy(float)
    bmeta["status_order"] = bmeta.status.map({"Healthy": 0, "NonLesional": 1, "PeriLesional": 2, "Lesional": 3})
    lmm = smf.mixedlm("expression ~ status_order", bmeta, groups=bmeta.donor).fit(reml=True)
    patient = bmeta[bmeta.status != "Healthy"]
    patient_lmm = smf.mixedlm("expression ~ status_order", patient, groups=patient.donor).fit(reml=True)
    les = bmeta[bmeta.status == "Lesional"]
    non = bmeta[bmeta.status == "NonLesional"].set_index("donor").loc[les.donor]
    bdiff = les.expression.to_numpy() - non.expression.to_numpy()
    bpaired = stats.ttest_1samp(bdiff, 0)
    spear = stats.spearmanr(bmeta.expression, bmeta.status_order)
    group_means = bmeta.groupby("status").expression.mean()
    correlations = []
    genes = ["MITF", "KIT", "AURKA"]
    program_genes = ["MITF", "DCT", "MLANA", "TYRP1", "PMEL", "TYR"]
    z = a.loc[program_genes].sub(a.loc[program_genes].mean(axis=1), axis=0).div(a.loc[program_genes].std(axis=1, ddof=1), axis=0)
    program = z.mean(axis=0)
    for gene in genes:
        rho = stats.spearmanr(a.loc["BHLHE41"], a.loc[gene])
        correlations.append({"comparator": gene, "rho": rho.statistic, "p": rho.pvalue, "n": a.shape[1]})
    rho = stats.spearmanr(a.loc["BHLHE41"], program)
    correlations.append({"comparator": "mature_melanocyte_program", "rho": rho.statistic, "p": rho.pvalue, "n": a.shape[1]})
    corr = pd.DataFrame(correlations)
    corr.columns = [c.strip() for c in corr.columns]
    corr.to_csv(OUT / "GSE65127_BHLHE41_correlations.csv", index=False)
    bmeta.to_csv(OUT / "GSE65127_BHLHE41_sample_values.csv", index_label="sample")
    summary = {
        "annotation": "GPL570 Gene symbol and GPL6884 Symbol; ambiguous multi-symbol probes excluded; multiple probes collapsed by median",
        "genes_GSE65127": len(a), "genes_GSE75819": len(b), "common_genes": len(common),
        "GSE65127_significant_absFC1_FDR05": int(((r1.log2FC.abs()>1)&(r1.FDR_BH<.05)).sum()),
        "GSE75819_significant_absFC1_FDR05": int(((r2.log2FC.abs()>1)&(r2.FDR_BH<.05)).sum()),
        "replicated_FDR05_same_direction": int(joined.replicated_FDR05_same_direction.sum()),
        "replicated_large_effect": int(joined.replicated_large_effect.sum()),
        "BHLHE41_group_means": group_means.to_dict(),
        "BHLHE41_paired_log2FC": float(np.mean(bdiff)), "BHLHE41_paired_p": float(bpaired.pvalue),
        "BHLHE41_descriptive_spearman_rho": float(spear.statistic), "BHLHE41_descriptive_spearman_p": float(spear.pvalue),
        "BHLHE41_mixed_model_beta_per_status": float(lmm.params.status_order),
        "BHLHE41_mixed_model_p": float(lmm.pvalues.status_order),
        "BHLHE41_patient_only_mixed_model_beta": float(patient_lmm.params.status_order),
        "BHLHE41_patient_only_mixed_model_p": float(patient_lmm.pvalues.status_order),
        "note": "Spearman trend is descriptive; mixed model accounts for repeated measures within donors."
    }
    (OUT / "differential_and_BHLHE41_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
