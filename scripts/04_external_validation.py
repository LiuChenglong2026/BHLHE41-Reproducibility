#!/usr/bin/env python3
"""External validation of the locked six-gene linear score in GSE53146."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from geo_io import ROOT, read_series_matrix, read_gpl14951, map_probes

SEED = 20261117
N_BOOT = 2000
OUT = ROOT / "results" / "external_validation"
OUT.mkdir(parents=True, exist_ok=True)


def stratified_bootstrap(y, score, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    auc, delta, diff = [], [], []
    for _ in range(n_boot):
        i0 = rng.choice(idx0, len(idx0), replace=True)
        i1 = rng.choice(idx1, len(idx1), replace=True)
        s0, s1 = score[i0], score[i1]
        a = roc_auc_score(np.r_[np.zeros(len(i0)), np.ones(len(i1))], np.r_[s0, s1])
        auc.append(a); delta.append(2*a-1); diff.append(s1.mean()-s0.mean())
    return {name: np.percentile(vals, [2.5, 97.5]).tolist()
            for name, vals in {"auc": auc, "cliffs_delta": delta, "mean_difference": diff}.items()}


def main():
    model = json.loads((ROOT / "results" / "machine_learning" / "final_signature_model.json").read_text())
    genes = model["genes"]
    raw, meta = read_series_matrix("GSE53146")
    annotation = read_gpl14951()
    probe_map = annotation[annotation.Symbol.isin(genes) & annotation.ID.isin(raw.index)][["ID", "Symbol"]]
    probe_map.sort_values(["Symbol", "ID"]).to_csv(OUT / "GSE53146_GPL14951_six_gene_probe_map.csv", index=False)
    mapped = map_probes(raw, annotation, "ID", "Symbol")
    missing = set(genes) - set(mapped.index)
    if missing:
        raise ValueError(f"Genes absent from probe-level cross-platform map: {missing}")
    expr = np.log2(np.maximum(mapped.loc[genes], 0) + 1)
    z = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1, ddof=1), axis=0)
    coef = pd.Series(model["coefficients"])
    score = z.mul(coef, axis=0).sum(axis=0) + model["intercept"]
    y = pd.Series([0]*5 + [1]*5, index=score.index, name="vitiligo")
    if list(score.index) != list(raw.columns):
        raise ValueError("External sample order mismatch")
    result = pd.DataFrame({"sample": score.index, "title": meta["title"].values,
                           "group": np.where(y == 1, "Vitiligo", "Control"),
                           "y": y.values, "vitiligo_discrimination_score": score.values})
    for gene in genes:
        result[gene + "_z"] = z.loc[gene].values
    result.to_csv(OUT / "GSE53146_sample_scores.csv", index=False)
    u = stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided", method="exact")
    auc = roc_auc_score(y, score)
    cliffs = 2*auc - 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    s1, s0 = score[y == 1], score[y == 0]
    pooled_sd = np.sqrt(((n1-1)*s1.var(ddof=1)+(n0-1)*s0.var(ddof=1))/(n1+n0-2))
    d = (s1.mean()-s0.mean())/pooled_sd
    hedges_g = d * (1 - 3/(4*(n1+n0)-9))
    ci = stratified_bootstrap(y.to_numpy(), score.to_numpy())
    fpr, tpr, thresholds = roc_curve(y, score)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(OUT / "GSE53146_roc_coordinates.csv", index=False)
    summary = {"dataset": "GSE53146", "n_control": n0, "n_vitiligo": n1,
               "genes": genes, "score_direction": "higher favors vitiligo",
               "control_mean_score": float(s0.mean()), "vitiligo_mean_score": float(s1.mean()),
               "mean_difference": float(s1.mean()-s0.mean()),
               "mean_difference_95ci_stratified_bootstrap": ci["mean_difference"],
               "mann_whitney_U": float(u.statistic), "two_sided_exact_p": float(u.pvalue),
               "auc": float(auc), "auc_95ci_stratified_bootstrap": ci["auc"],
               "cliffs_delta": float(cliffs), "cliffs_delta_95ci_stratified_bootstrap": ci["cliffs_delta"],
               "hedges_g": float(hedges_g), "n_bootstrap": N_BOOT, "seed": SEED,
               "normalization": "log2(max(intensity,0)+1), then label-free gene-wise z scaling within GSE53146",
               "annotation": "Official GEO GPL14951 full platform table (ID/Symbol); ambiguous multi-symbol probes excluded; multiple probes collapsed by median",
               "interpretation": "Independent biological cohort validation of score ranking; not validation of a fixed clinical threshold or calibrated probability."}
    (OUT / "GSE53146_validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
