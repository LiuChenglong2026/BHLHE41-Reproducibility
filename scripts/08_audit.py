#!/usr/bin/env python3
"""Independently audit source integrity, fold isolation and key reported results.

The machine-learning audit does not merely inspect file names. For every outer
fold it rebuilds study-specific scalers from training samples, recomputes paired
screening statistics, refits the final logistic coefficients, recalculates the
held-out scores and checks Borda aggregation against the saved artifacts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from geo_io import ROOT, classifier_data

SEED = 20261117
N_BOOT = 2000
RES = ROOT / "results"
ML = RES / "machine_learning"
EXT = RES / "external_validation"
DE = RES / "differential_expression"
REPORT = RES / "audit" / "reproducibility_audit.json"
C_GRID = np.array([0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0])
GENES6 = ["DCT","TYR","MC1R","FAM24B","WFS1","RPL31"]
EXPECTED_PROBES = {"DCT":"ILMN_1701783", "TYR":"ILMN_1788774", "MC1R":"ILMN_1653319",
                   "FAM24B":"ILMN_2140799", "WFS1":"ILMN_1759023", "RPL31":"ILMN_1754195"}
CHECKS = []


def require(condition, name, details=None):
    if not condition:
        raise AssertionError(f"{name}: {details or 'condition failed'}")
    CHECKS.append({"name":name, "status":"passed", "details":details})


def close(actual, expected, name, rtol=1e-9, atol=1e-9):
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    error = float(np.max(np.abs(actual_array-expected_array))) if actual_array.size else 0.0
    require(np.allclose(actual_array, expected_array, rtol=rtol, atol=atol), name,
            {"max_absolute_error":error, "rtol":rtol, "atol":atol})
    return error


def load_json(path):
    return json.loads(Path(path).read_text())


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def own_scalers(expr, meta, train_samples):
    """Independent implementation of training-only, study-specific z scaling."""
    z = pd.DataFrame(index=expr.index, columns=expr.columns, dtype=float)
    params = {}
    for study in sorted(meta.study.unique()):
        tr = meta.index[(meta.study == study) & meta.index.isin(train_samples)]
        all_study = meta.index[meta.study == study]
        mean = expr.loc[:,tr].mean(axis=1)
        sd = expr.loc[:,tr].std(axis=1, ddof=1)
        z.loc[:,all_study] = expr.loc[:,all_study].sub(mean,axis=0).div(sd,axis=0)
        params[study] = (mean,sd)
    return z, params


def own_paired_t(z, meta, train_samples):
    deltas = []
    for pair in sorted(meta.loc[train_samples,"pair_id"].unique()):
        group = meta[(meta.pair_id == pair) & meta.index.isin(train_samples)]
        les = group.index[group.label == 1].tolist()
        non = group.index[group.label == 0].tolist()
        assert len(les) == len(non) == 1
        deltas.append(z[les[0]].to_numpy()-z[non[0]].to_numpy())
    d = np.column_stack(deltas)
    mean = d.mean(axis=1)
    sd = d.std(axis=1,ddof=1)
    t = np.divide(mean,sd/np.sqrt(d.shape[1]),out=np.zeros_like(mean),where=sd>0)
    return pd.DataFrame({"gene":z.index,"paired_delta":mean,"paired_t":t,"abs_t":np.abs(t)}).set_index("gene")


def check_sources():
    downloader = load_script("download_data",ROOT/"scripts"/"00_download_data.py")
    manifest = downloader.load_manifest()
    messages = downloader.verify_records(manifest["sources"],require_all=True)
    require(len(messages) == 13,"source_snapshots_sha256",{"verified":len(messages)})
    gpl = ROOT/"data"/"annotations"/"GPL14951.txt"
    raw = gpl.read_bytes()
    require(b"\x00" not in raw and b"!platform_table_end" in raw,
            "gpl14951_complete_and_uncorrupted",{"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})


def check_fold_isolation():
    expr,meta = classifier_data()
    pred = pd.read_csv(ML/"leave_one_pair_out_predictions.csv")
    assignments = pd.read_csv(ML/"leave_one_pair_out_fold_assignments.csv")
    fold_features = pd.read_csv(ML/"leave_one_pair_out_fold_features.csv")
    fold_coef = pd.read_csv(ML/"leave_one_pair_out_fold_model_coefficients.csv")
    scalers = pd.read_csv(ML/"leave_one_pair_out_training_scaler_parameters.csv.gz")
    top500 = pd.read_csv(ML/"leave_one_pair_out_training_top500_statistics.csv.gz")
    ranks = pd.read_csv(ML/"leave_one_pair_out_all_feature_ranks.csv")
    pair_ids = sorted(meta.pair_id.unique())
    require(expr.shape == (13511,50),"development_matrix_dimensions",{"shape":list(expr.shape)})
    require(len(pair_ids) == 25 and len(pred) == 50 and pred["sample"].nunique() == 50,
            "strict_oof_prediction_completeness",{"pairs":len(pair_ids),"predictions":len(pred)})
    require(len(assignments) == 1250 and len(scalers) == 25*2*13511 and len(top500) == 25*500,
            "fold_artifact_dimensions",{"assignments":len(assignments),"scaler_rows":len(scalers),"top500_rows":len(top500)})
    max_scaler_error = max_t_error = max_coef_error = max_score_error = 0.0
    for fold,pair in enumerate(pair_ids,start=1):
        test_samples = meta.index[meta.pair_id == pair].tolist()
        train_samples = meta.index[meta.pair_id != pair].tolist()
        assert len(test_samples) == 2 and len(train_samples) == 48
        assert set(meta.loc[test_samples,"label"]) == {0,1}
        af = assignments[assignments.fold == fold]
        assert len(af) == 50
        assert set(af.loc[af.role == "test","sample"]) == set(test_samples)
        assert set(af.loc[af.role == "train","sample"]) == set(train_samples)
        assert (af.loc[af.role == "test","pair_id"] == pair).all()
        assert (af.loc[af.role == "train","pair_id"] != pair).all()
        pf = pred[pred.fold == fold]
        assert len(pf) == 2 and set(pf["sample"]) == set(test_samples) and (pf.held_out_pair == pair).all()
        ff = fold_features[fold_features.fold == fold]
        assert len(ff) == 1 and ff.iloc[0].held_out_pair == pair
        assert int(ff.iloc[0].n_train_samples) == 48 and int(ff.iloc[0].n_test_samples) == 2
        z,params = own_scalers(expr,meta,train_samples)
        for study,(mean,sd) in params.items():
            saved = scalers[(scalers.fold == fold) & (scalers.study == study)].set_index("gene").loc[expr.index]
            assert len(saved) == 13511
            max_scaler_error = max(max_scaler_error,
                float(np.max(np.abs(saved["mean"].to_numpy()-mean.to_numpy()))),
                float(np.max(np.abs(saved["sd"].to_numpy()-sd.to_numpy()))))
            assert np.allclose(saved["mean"],mean,rtol=2e-8,atol=2e-7)
            assert np.allclose(saved["sd"],sd,rtol=2e-8,atol=2e-7)
        paired = own_paired_t(z,meta,train_samples)
        order = np.lexsort((paired.index.to_numpy(),-paired.abs_t.to_numpy()))
        expected_top = paired.index[order[:500]].tolist()
        saved_top = top500[top500.fold == fold]
        assert len(saved_top) == 500 and saved_top.gene.nunique() == 500
        assert set(saved_top.gene) == set(expected_top)
        recomputed = paired.loc[saved_top.gene]
        max_t_error = max(max_t_error,float(np.max(np.abs(recomputed.paired_t.to_numpy()-saved_top.paired_t.to_numpy()))))
        assert np.allclose(recomputed.paired_t,saved_top.paired_t,rtol=2e-8,atol=2e-7)
        assert np.allclose(recomputed.paired_delta,saved_top.paired_delta,rtol=2e-8,atol=2e-7)
        genes = ff.iloc[0].final_six_genes.split(";")
        assert len(genes) == 6 and len(set(genes)) == 6
        model = LogisticRegression(penalty="l2",solver="liblinear",C=1.0,max_iter=5000,random_state=SEED)
        model.fit(z.loc[genes,train_samples].T.to_numpy(float),meta.loc[train_samples,"label"].to_numpy(int))
        saved_coef = fold_coef[fold_coef.fold == fold].set_index("gene").loc[genes]
        max_coef_error = max(max_coef_error,float(np.max(np.abs(saved_coef.coefficient.to_numpy()-model.coef_[0]))),
                             float(np.max(np.abs(saved_coef.intercept.to_numpy()-model.intercept_[0]))))
        assert np.allclose(saved_coef.coefficient,model.coef_[0],rtol=1e-8,atol=1e-8)
        assert np.allclose(saved_coef.intercept,model.intercept_[0],rtol=1e-8,atol=1e-8)
        score = model.decision_function(z.loc[genes,test_samples].T.to_numpy(float))
        observed = pf.set_index("sample").loc[test_samples,"decision_score"].to_numpy()
        max_score_error = max(max_score_error,float(np.max(np.abs(observed-score))))
        assert np.allclose(observed,score,rtol=1e-8,atol=1e-8)
        rr = ranks[ranks.fold == fold].copy().set_index("gene",drop=False)
        rr.index.name = None  # avoid index/column ambiguity during the gene tie-break sort
        computed_borda = pd.Series(0.0,index=rr.index)
        computed_count = pd.Series(0,index=rr.index,dtype=int)
        for column,key in [("LASSO","lasso_genes"),("SVM_RFE","svm_rfe_genes"),("Random_forest","random_forest_genes")]:
            returned = ff.iloc[0][key].split(";") if isinstance(ff.iloc[0][key],str) and ff.iloc[0][key] else []
            k = len(returned)
            valid = rr[column].notna()
            computed_borda.loc[valid] += (k-rr.loc[valid,column]+1)/k
            computed_count.loc[valid] += 1
        assert np.allclose(computed_borda,rr.borda_score,rtol=1e-10,atol=1e-10)
        assert np.array_equal(computed_count.to_numpy(),rr.selection_count.to_numpy())
        assert np.allclose(rr.abs_paired_t,paired.loc[rr.index,"abs_t"],rtol=2e-8,atol=2e-7)
        ordered = rr.assign(recomputed_borda=computed_borda,recomputed_count=computed_count).sort_values(
            ["recomputed_borda","recomputed_count","abs_paired_t","gene"],ascending=[False,False,False,True])
        assert ordered.gene.iloc[:6].tolist() == genes
        assert set(rr.index[rr.in_final_six.astype(bool)]) == set(genes)
        tuning = pd.read_csv(ML/f"fold_{fold:02d}_lasso_tuning.csv")
        assert np.allclose(tuning.C,C_GRID)
        scores = tuning.inner_grouped_oof_auc.to_numpy()
        best = C_GRID[int(np.flatnonzero(scores >= scores.max()-1e-12)[0])]
        assert np.isclose(best,ff.iloc[0].best_lasso_C)
        CHECKS.append({"name":f"fold_{fold:02d}_independent_refit","status":"passed",
                       "details":{"held_out_pair":pair,"train_samples":48,"test_samples":2,"top500":500,"six_genes":genes}})
    require(True,"all_25_folds_recomputed_without_test_samples",{
        "max_scaler_absolute_error":max_scaler_error,"max_paired_t_absolute_error":max_t_error,
        "max_coefficient_absolute_error":max_coef_error,"max_held_out_score_absolute_error":max_score_error})
    return expr,meta,pred


def check_final_model_and_performance(expr,meta,pred):
    final = load_json(ML/"final_signature_model.json")
    require(final["genes"] == GENES6,"locked_model_gene_order",{"genes":final["genes"]})
    require(final["three_way_intersection"] == ["DCT"],"full_data_three_method_intersection",final["three_way_intersection"])
    z,params = own_scalers(expr,meta,meta.index.tolist())
    for study,(mean,sd) in params.items():
        saved = pd.read_csv(ML/f"final_scaler_{study}.csv",index_col=0).loc[expr.index]
        close(saved["mean"],mean,f"final_{study}_scaler_mean",rtol=2e-8,atol=2e-7)
        close(saved["sd"],sd,f"final_{study}_scaler_sd",rtol=2e-8,atol=2e-7)
    model = LogisticRegression(penalty="l2",solver="liblinear",C=1.0,max_iter=5000,random_state=SEED)
    model.fit(z.loc[GENES6,meta.index].T.to_numpy(float),meta.label.to_numpy(int))
    observed = np.array([final["coefficients"][g] for g in GENES6])
    close(observed,model.coef_[0],"final_locked_coefficients",rtol=1e-9,atol=1e-9)
    close(final["intercept"],model.intercept_[0],"final_locked_intercept",rtol=1e-9,atol=1e-9)
    stability = pd.read_csv(ML/"six_gene_selection_stability.csv").set_index("gene")
    require(int(stability.loc["DCT","folds_selected"]) == 25 and int(stability.loc["MC1R","folds_selected"]) == 25,
            "dct_mc1r_selected_in_every_fold")
    require(int(stability.loc["RPL31","folds_selected"]) == 2,"rpl31_instability_disclosed",{"folds":2,"frequency":0.08})
    cv = load_json(ML/"lopocv_summary.json")
    y = pred.y.to_numpy(); score = pred.decision_score.to_numpy(); pairs = pred.held_out_pair.to_numpy()
    auc = roc_auc_score(y,score)
    close(auc,0.9872,"development_auc_regression_anchor")
    close(auc,cv["auc"],"development_auc_matches_summary")
    rng = np.random.default_rng(SEED)
    unique_pairs = np.unique(pairs)
    by_pair = {p:np.flatnonzero(pairs == p) for p in unique_pairs}
    boot = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique_pairs,size=len(unique_pairs),replace=True)
        idx = np.concatenate([by_pair[p] for p in sampled])
        if len(np.unique(y[idx])) == 2:
            boot.append(roc_auc_score(y[idx],score[idx]))
    ci = np.percentile(boot,[2.5,97.5])
    close(ci,cv["cluster_bootstrap_95ci"],"development_donor_pair_bootstrap_ci")
    saved_boot = np.load(ML/"lopocv_cluster_bootstrap_auc.npy")
    close(boot,saved_boot,"development_bootstrap_replicates")
    external = pd.read_csv(EXT/"GSE53146_sample_scores.csv")
    es = load_json(EXT/"GSE53146_validation_summary.json")
    require(len(external) == 10 and external.y.sum() == 5,"external_sample_completeness",{"n":len(external),"cases":int(external.y.sum())})
    zcols = [g+"_z" for g in GENES6]
    coef = np.array([final["coefficients"][g] for g in GENES6])
    recalculated = external[zcols].to_numpy().dot(coef)+final["intercept"]
    close(recalculated,external.vitiligo_discrimination_score,"external_scores_from_locked_coefficients")
    probes = pd.read_csv(EXT/"GSE53146_GPL14951_six_gene_probe_map.csv")
    require(dict(zip(probes.Symbol,probes.ID)) == EXPECTED_PROBES,"official_gpl14951_six_gene_probe_map",EXPECTED_PROBES)
    ey = external.y.to_numpy(); escore = external.vitiligo_discrimination_score.to_numpy()
    s0,s1 = escore[ey == 0],escore[ey == 1]
    eauc = roc_auc_score(ey,escore)
    close(eauc,0.96,"external_auc_regression_anchor")
    u = stats.mannwhitneyu(s1,s0,alternative="two-sided",method="exact")
    close(u.statistic,24.0,"external_mann_whitney_u")
    close(u.pvalue,0.015873015873015872,"external_exact_two_sided_p")
    close(s1.mean()-s0.mean(),es["mean_difference"],"external_mean_score_difference")
    n1,n0 = len(s1),len(s0)
    pooled = np.sqrt(((n1-1)*s1.var(ddof=1)+(n0-1)*s0.var(ddof=1))/(n1+n0-2))
    hedges = ((s1.mean()-s0.mean())/pooled)*(1-3/(4*(n1+n0)-9))
    close(hedges,es["hedges_g"],"external_hedges_g")
    rng = np.random.default_rng(SEED)
    idx0,idx1 = np.flatnonzero(ey == 0),np.flatnonzero(ey == 1)
    aucs,deltas,diffs = [],[],[]
    for _ in range(N_BOOT):
        i0 = rng.choice(idx0,len(idx0),replace=True); i1 = rng.choice(idx1,len(idx1),replace=True)
        b0,b1 = escore[i0],escore[i1]
        a = roc_auc_score(np.r_[np.zeros(len(i0)),np.ones(len(i1))],np.r_[b0,b1])
        aucs.append(a); deltas.append(2*a-1); diffs.append(b1.mean()-b0.mean())
    for values,key in [(aucs,"auc_95ci_stratified_bootstrap"),(deltas,"cliffs_delta_95ci_stratified_bootstrap"),
                       (diffs,"mean_difference_95ci_stratified_bootstrap")]:
        close(np.percentile(values,[2.5,97.5]),es[key],f"external_{key}")
    close(2*eauc-1,es["cliffs_delta"],"external_cliffs_delta")


def check_differential_and_graying():
    summary = load_json(DE/"differential_and_BHLHE41_summary.json")
    require(summary["common_genes"] == 13511 and summary["replicated_FDR05_same_direction"] == 14 and
            summary["replicated_large_effect"] == 6,"differential_result_dimensions",
            {k:summary[k] for k in ["common_genes","replicated_FDR05_same_direction","replicated_large_effect"]})
    require(summary["genes_GSE65127"] == 20848 and summary["genes_GSE75819"] == 25440,
            "mapped_gene_counts")
    require(summary["GSE65127_significant_absFC1_FDR05"] == 24 and
            summary["GSE75819_significant_absFC1_FDR05"] == 1290,"large_effect_significant_counts")
    rep = pd.read_csv(DE/"replicated_14_genes.csv")
    expected14 = {"OCA2","TYR","BCAN","CA14","MLANA","DCT","SETD4","SLC6A17","KIF1A","LGI3","FABP5","SCARB1","TYRP1","BACE2"}
    require(set(rep.gene) == expected14,"fourteen_replicated_gene_identities")
    require((rep.GSE65127_log2FC < 0).sum() == 12 and (rep.GSE65127_log2FC > 0).sum() == 2,
            "twelve_down_two_up_replicated_genes")
    large = pd.read_csv(DE/"replicated_large_effect_6_genes.csv")
    require(set(large.gene) == {"TYR","CA14","MLANA","DCT","LGI3","TYRP1"},"six_large_effect_gene_identities")
    close(summary["BHLHE41_paired_log2FC"],-0.10026500000000035,"bhlhe41_paired_log2fc")
    close(summary["BHLHE41_paired_p"],0.23379112716165346,"bhlhe41_paired_p")
    close(summary["BHLHE41_mixed_model_beta_per_status"],-0.06546760510913888,"bhlhe41_donor_aware_beta")
    close(summary["BHLHE41_mixed_model_p"],0.017651467699681535,"bhlhe41_donor_aware_p")
    gray = load_json(RES/"graying"/"graying_concordance_summary.json")
    require(gray["unique_published_genes"] == 40 and gray["measurable_in_both_vitiligo_studies"] == 31 and
            gray["concordant"] == 20,"graying_primary_counts")
    test = stats.binomtest(20,31,.5,alternative="two-sided")
    close(test.pvalue,gray["two_sided_exact_binomial_p"],"graying_exact_binomial_p")
    ci = test.proportion_ci(confidence_level=.95)
    close([ci.low,ci.high],gray["proportion_95ci_exact"],"graying_exact_proportion_ci")
    require(gray["sensitivity_both_vitiligo_studies_same_direction_n"] == 14 and
            gray["sensitivity_concordant"] == 12,"graying_selected_sensitivity_counts")
    gray31 = pd.read_csv(RES/"graying"/"Bian_vitiligo_31_gene_direction_comparison.csv")
    require("BHLHE41" not in set(gray31.gene),"bhlhe41_not_forced_into_unmeasurable_cross_cohort_comparison")


def check_melanoma():
    ccle = pd.read_csv(ROOT/"data"/"source"/"CCLE_2019_cutaneous_melanoma_expression_RPKM.csv",index_col=0)
    ccle_log = np.log2(np.maximum(ccle,0)+1)
    ccorr = pd.read_csv(RES/"melanoma"/"CCLE_2019_58_melanoma_BHLHE41_correlations.csv").set_index("gene")
    require(len(ccle) == 58,"ccle_exact_cell_line_count",{"n":len(ccle)})
    mitf = stats.spearmanr(ccle_log.BHLHE41,ccle_log.MITF)
    close(mitf.statistic,ccorr.loc["MITF","rho"],"ccle_bhlhe41_mitf_rho")
    close(mitf.pvalue,ccorr.loc["MITF","p"],"ccle_bhlhe41_mitf_p")
    cs = load_json(RES/"melanoma"/"CCLE_2019_summary.json")
    high = ccle.MITF > ccle.MITF.median(); hi,lo = ccle.loc[high,"BHLHE41"],ccle.loc[~high,"BHLHE41"]
    close(hi.mean()/lo.mean(),cs["mean_fold_ratio"],"ccle_mitf_group_fold_ratio")
    mw = stats.mannwhitneyu(hi,lo,alternative="two-sided",method="asymptotic")
    close(mw.pvalue,cs["mann_whitney_two_sided_asymptotic_p"],"ccle_tie_corrected_asymptotic_p")
    tcga = pd.read_csv(ROOT/"data"/"source"/"TCGA_SKCM_cBioPortal_expression_RSEM.csv",index_col=0)
    meta = load_json(RES/"melanoma"/"TCGA_SKCM_analysis_metadata.json")
    survival = pd.read_csv(RES/"melanoma"/"TCGA_SKCM_459_patient_survival_analysis.csv",index_col=0)
    require(len(tcga) == 472 and tcga.index.str[:12].nunique() == 469 and len(survival) == 459 and
            int(survival.event.sum()) == 220,"tcga_expression_survival_dimensions",
            {"expression":len(tcga),"patients":tcga.index.str[:12].nunique(),"survival":len(survival),"events":int(survival.event.sum())})
    log = np.log2(np.maximum(tcga,0)+1)
    program_genes = ["MITF","DCT","MLANA","TYRP1","PMEL","TYR"]
    program = log[program_genes].apply(lambda c:(c-c.mean())/c.std(ddof=1)).mean(axis=1)
    rho = stats.spearmanr(log.BHLHE41,program)
    tcorr = pd.read_csv(RES/"melanoma"/"TCGA_SKCM_472_sample_correlations.csv")
    row = tcorr[(tcorr.predictor == "BHLHE41") & (tcorr.target == "melanocytic_program")].iloc[0]
    close(rho.statistic,row.rho,"tcga_melanocytic_program_rho")
    close(rho.pvalue,row.p,"tcga_melanocytic_program_p")
    close(row.rho,0.333748, "tcga_current_rho_regression_anchor",rtol=1e-5,atol=1e-6)
    surv = pd.read_csv(RES/"melanoma"/"TCGA_SKCM_survival_summary.csv").set_index("gene")
    expected = {"BHLHE41":(1.064632645,0.361004014),"AURKA":(1.199506722,0.011020366),
                "melanocytic_program":(1.214668180,0.007552588)}
    for gene,(hr,p) in expected.items():
        close(surv.loc[gene,"HR_per_SD"],hr,f"tcga_{gene}_hr_per_sd",rtol=1e-8,atol=1e-9)
        close(surv.loc[gene,"cox_p"],p,f"tcga_{gene}_cox_p",rtol=1e-8,atol=1e-9)


def check_figures():
    manifest = load_json(ROOT/"figures"/"figure_manifest.json")
    expected = {"Figure_1","Figure_2","Figure_3","Figure_4","Figure_S1","Figure_S2","Graphical_Abstract"}
    require(set(manifest) == expected,"seven_figure_manifest_entries")
    for name in sorted(expected):
        for ext in ["png","tiff","pdf"]:
            path = ROOT/"figures"/f"{name}.{ext}"
            require(path.exists() and path.stat().st_size > 1000,f"figure_file_{name}_{ext}",{"bytes":path.stat().st_size if path.exists() else 0})
        with Image.open(ROOT/"figures"/f"{name}.png") as image:
            image.verify()
        with Image.open(ROOT/"figures"/f"{name}.tiff") as image:
            require(image.tag_v2.get(259) == 5,f"figure_{name}_tiff_lzw",{"compression_tag":image.tag_v2.get(259),"size":list(image.size)})
    with Image.open(ROOT/"figures"/"Graphical_Abstract.png") as image:
        dpi = image.info.get("dpi",(300,300))
        cm = [image.width/dpi[0]*2.54,image.height/dpi[1]*2.54]
        require(abs(cm[0]-17) < .02 and abs(cm[1]-5) < .02,"graphical abstract_png_exact_17cm_x_5cm",
                {"cm":cm,"pixels":list(image.size),"dpi":list(dpi)})
    pdf = (ROOT/"figures"/"Graphical_Abstract.pdf").read_bytes()
    match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",pdf)
    require(match is not None,"graphical abstract_pdf_mediabox_found")
    width,height = map(float,match.groups())
    cm = [width*2.54/72,height*2.54/72]
    require(abs(cm[0]-17) < .02 and abs(cm[1]-5) < .02,"graphical abstract_pdf_exact_17cm_x_5cm",{"cm":cm,"points":[width,height]})
    caption = (ROOT/"figures"/"Graphical_Abstract_caption.txt").read_text().strip()
    require(0 < len(caption.split()) <= 50,"graphical abstract_caption_word_limit",{"words":len(caption.split())})


def main():
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    started = datetime.now(timezone.utc)
    status = "passed"
    error = None
    try:
        check_sources()
        expr,meta,pred = check_fold_isolation()
        check_final_model_and_performance(expr,meta,pred)
        check_differential_and_graying()
        check_melanoma()
        check_figures()
        require((ROOT/"README.md").exists() and (ROOT/"requirements.txt").exists() and
                (ROOT/"run_analysis.py").exists(),"one_command_reproduction_files_present")
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        CHECKS.append({"name":"audit_execution","status":"failed","details":error})
    finally:
        report = {"status":status,"started_utc":started.isoformat(),
                  "finished_utc":datetime.now(timezone.utc).isoformat(),
                  "python":platform.python_version(),"seed":SEED,"bootstrap_replicates":N_BOOT,
                  "checks_passed":sum(x["status"] == "passed" for x in CHECKS),
                  "checks":CHECKS,"error":error,
                  "scope":"Source SHA-256; independent refit of every outer fold; final and external scores; bootstrap intervals; DE, graying, melanoma and figure integrity."}
        REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps({k:report[k] for k in ["status","checks_passed","error"]},indent=2))
        print(f"Audit report: {REPORT}")
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
