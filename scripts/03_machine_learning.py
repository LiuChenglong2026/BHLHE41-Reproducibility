#!/usr/bin/env python3
"""Leakage-controlled leave-one-donor-pair-out signature development.

Every outer training fold repeats cohort-wise scaling, paired top-500 screening,
LASSO, SVM-RFE, random-forest selection, consensus feature ranking and final
coefficient fitting. The two samples from the held-out donor are never used in
any fitting operation.
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
from geo_io import ROOT, classifier_data, fit_cohort_scalers

SEED = 20261117
N_BOOT = 2000
RESULTS = ROOT / "results" / "machine_learning"
RESULTS.mkdir(parents=True, exist_ok=True)
C_GRID = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])


def top_paired_features(z, meta, train_samples, n=500):
    train_meta = meta.loc[train_samples]
    deltas = []
    for _, group in train_meta.groupby("pair_id"):
        les = group.index[group.label == 1][0]
        non = group.index[group.label == 0][0]
        deltas.append(z[les] - z[non])
    d = np.column_stack(deltas)
    mean = d.mean(axis=1)
    sd = d.std(axis=1, ddof=1)
    t = np.divide(mean, sd / np.sqrt(d.shape[1]), out=np.zeros_like(mean), where=sd > 0)
    order = np.lexsort((z.index.to_numpy(), -np.abs(t)))
    selected = z.index[order[:n]]
    table = pd.DataFrame({"gene": z.index, "paired_delta": mean, "paired_t": t,
                          "abs_t": np.abs(t)}).sort_values(["abs_t", "gene"], ascending=[False, True])
    return list(selected), table


def tune_lasso(X, y, groups, seed):
    splitter = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y, groups))
    scores = []
    for c in C_GRID:
        oof = np.full(len(y), np.nan)
        for tr, va in splits:
            model = LogisticRegression(penalty="l1", solver="liblinear", C=float(c),
                                       max_iter=5000, random_state=seed)
            model.fit(X[tr], y[tr])
            oof[va] = model.decision_function(X[va])
        scores.append(roc_auc_score(y, oof))
    scores = np.asarray(scores)
    best = int(np.flatnonzero(scores >= scores.max() - 1e-12)[0])  # smallest C on ties
    return float(C_GRID[best]), pd.DataFrame({"C": C_GRID, "inner_grouped_oof_auc": scores})


def select_features(z, meta, train_samples, seed):
    top500, paired_table = top_paired_features(z, meta, train_samples)
    X = z.loc[top500, train_samples].T.to_numpy(float)
    y = meta.loc[train_samples, "label"].to_numpy(int)
    groups = meta.loc[train_samples, "pair_id"].to_numpy()
    best_c, tuning = tune_lasso(X, y, groups, seed)
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=best_c,
                               max_iter=5000, random_state=seed)
    lasso.fit(X, y)
    lasso_nonzero = np.flatnonzero(np.abs(lasso.coef_[0]) > 1e-10)
    lasso_order = lasso_nonzero[np.argsort(-np.abs(lasso.coef_[0, lasso_nonzero]), kind="stable")]
    lasso_genes = [top500[i] for i in lasso_order]

    svm = SVC(kernel="linear", C=1.0)
    rfe = RFE(estimator=svm, n_features_to_select=10, step=0.1)
    rfe.fit(X, y)
    selected_idx = np.flatnonzero(rfe.support_)
    # The refitted linear SVM coefficients rank the ten retained features.
    svm_order = selected_idx[np.argsort(-np.abs(rfe.estimator_.coef_[0]), kind="stable")]
    svm_genes = [top500[i] for i in svm_order]

    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                class_weight="balanced", random_state=seed, n_jobs=1)
    rf.fit(X, y)
    rf_order = np.argsort(-rf.feature_importances_, kind="stable")[:15]
    rf_genes = [top500[i] for i in rf_order]

    methods = {"LASSO": lasso_genes, "SVM_RFE": svm_genes, "Random_forest": rf_genes}
    records = []
    abst = paired_table.set_index("gene").abs_t
    for method, genes in methods.items():
        for rank, gene in enumerate(genes, start=1):
            k = len(genes)
            records.append({"gene": gene, "method": method, "rank": rank,
                            "borda_component": (k - rank + 1) / k})
    long = pd.DataFrame(records)
    if long.empty:
        raise RuntimeError("All feature selectors returned empty lists")
    consensus = long.pivot_table(index="gene", columns="method", values="rank", aggfunc="first")
    borda = long.groupby("gene").borda_component.sum()
    count = long.groupby("gene").method.nunique()
    consensus["borda_score"] = borda
    consensus["selection_count"] = count
    consensus["abs_paired_t"] = abst.reindex(consensus.index)
    consensus = consensus.reset_index().sort_values(
        ["borda_score", "selection_count", "abs_paired_t", "gene"],
        ascending=[False, False, False, True])
    six = consensus.gene.iloc[:6].tolist()
    details = {"best_lasso_C": best_c, "top500": top500, "paired_table": paired_table,
               "methods": methods, "consensus": consensus, "six_genes": six, "lasso_tuning": tuning}
    return six, details


def fit_final_model(z, meta, train_samples, genes):
    X = z.loc[genes, train_samples].T.to_numpy(float)
    y = meta.loc[train_samples, "label"].to_numpy(int)
    model = LogisticRegression(penalty="l2", solver="liblinear", C=1.0,
                               max_iter=5000, random_state=SEED)
    model.fit(X, y)
    return model


def cluster_bootstrap_auc(y, score, pairs, seed=SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    unique_pairs = np.unique(pairs)
    by_pair = {p: np.flatnonzero(pairs == p) for p in unique_pairs}
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_pairs, size=len(unique_pairs), replace=True)
        idx = np.concatenate([by_pair[p] for p in sampled])
        if len(np.unique(y[idx])) == 2:
            values.append(roc_auc_score(y[idx], score[idx]))
    return np.percentile(values, [2.5, 97.5]), np.asarray(values)


def main():
    start = time.time()
    expr, meta = classifier_data()
    pair_ids = sorted(meta.pair_id.unique())
    all_predictions, fold_rows, all_consensus = [], [], []
    fold_assignments, fold_scalers, fold_top500, fold_coefficients = [], [], [], []
    for fold, pair in enumerate(pair_ids, start=1):
        test_samples = meta.index[meta.pair_id == pair].tolist()
        train_samples = meta.index[meta.pair_id != pair].tolist()
        assert not set(test_samples).intersection(train_samples)
        assert set(meta.loc[test_samples, "label"]) == {0, 1}
        z, scaler_params = fit_cohort_scalers(expr, meta, train_samples)
        genes, details = select_features(z, meta, train_samples, SEED + fold)
        model = fit_final_model(z, meta, train_samples, genes)
        for sample in meta.index:
            fold_assignments.append({"fold": fold, "held_out_pair": pair, "sample": sample,
                                     "pair_id": meta.loc[sample, "pair_id"],
                                     "role": "test" if sample in test_samples else "train"})
        for study, params in scaler_params.items():
            fold_scalers.append(params.assign(fold=fold, held_out_pair=pair, study=study,
                                              gene=params.index))
        selected_stats = details["paired_table"].set_index("gene").loc[details["top500"]].copy()
        selected_stats.insert(0, "gene", selected_stats.index)
        selected_stats.insert(0, "held_out_pair", pair)
        selected_stats.insert(0, "fold", fold)
        fold_top500.append(selected_stats.reset_index(drop=True))
        for gene, coefficient in zip(genes, model.coef_[0]):
            fold_coefficients.append({"fold": fold, "held_out_pair": pair, "gene": gene,
                                      "coefficient": float(coefficient),
                                      "intercept": float(model.intercept_[0])})
        score = model.decision_function(z.loc[genes, test_samples].T.to_numpy(float))
        for sample, value in zip(test_samples, score):
            all_predictions.append({"fold": fold, "held_out_pair": pair, "sample": sample,
                                    "study": meta.loc[sample, "study"], "donor": meta.loc[sample, "donor"],
                                    "status": meta.loc[sample, "status"], "y": int(meta.loc[sample, "label"]),
                                    "decision_score": float(value),
                                    "probability_not_clinically_calibrated": float(1/(1+np.exp(-value)))})
        consensus = details["consensus"].copy()
        consensus.insert(0, "fold", fold)
        consensus.insert(1, "held_out_pair", pair)
        consensus["in_final_six"] = consensus.gene.isin(genes)
        all_consensus.append(consensus)
        row = {"fold": fold, "held_out_pair": pair, "n_train_samples": len(train_samples),
               "n_test_samples": len(test_samples), "best_lasso_C": details["best_lasso_C"],
               "lasso_genes": ";".join(details["methods"]["LASSO"]),
               "svm_rfe_genes": ";".join(details["methods"]["SVM_RFE"]),
               "random_forest_genes": ";".join(details["methods"]["Random_forest"]),
               "final_six_genes": ";".join(genes),
               "three_way_intersection": ";".join(sorted(set(details["methods"]["LASSO"]) &
                                                          set(details["methods"]["SVM_RFE"]) &
                                                          set(details["methods"]["Random_forest"])))}
        fold_rows.append(row)
        details["lasso_tuning"].assign(fold=fold, held_out_pair=pair).to_csv(
            RESULTS / f"fold_{fold:02d}_lasso_tuning.csv", index=False)
        print(f"fold {fold:02d}/25 {pair}: C={details['best_lasso_C']}; six={','.join(genes)}", flush=True)

    pred = pd.DataFrame(all_predictions).sort_values(["fold", "sample"])
    pred.to_csv(RESULTS / "leave_one_pair_out_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "leave_one_pair_out_fold_features.csv", index=False)
    pd.DataFrame(fold_assignments).to_csv(RESULTS / "leave_one_pair_out_fold_assignments.csv", index=False)
    pd.DataFrame(fold_coefficients).to_csv(RESULTS / "leave_one_pair_out_fold_model_coefficients.csv", index=False)
    pd.concat(fold_top500, ignore_index=True).to_csv(
        RESULTS / "leave_one_pair_out_training_top500_statistics.csv.gz", index=False, float_format="%.10g")
    pd.concat(fold_scalers, ignore_index=True).to_csv(
        RESULTS / "leave_one_pair_out_training_scaler_parameters.csv.gz", index=False, float_format="%.10g")
    consensus_all = pd.concat(all_consensus, ignore_index=True)
    consensus_all.to_csv(RESULTS / "leave_one_pair_out_all_feature_ranks.csv", index=False)
    y = pred.y.to_numpy(); score = pred.decision_score.to_numpy(); pairs = pred.held_out_pair.to_numpy()
    auc = roc_auc_score(y, score)
    (lo, hi), boot = cluster_bootstrap_auc(y, score, pairs)
    fpr, tpr, thresholds = roc_curve(y, score)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(RESULTS / "lopocv_roc_coordinates.csv", index=False)
    np.save(RESULTS / "lopocv_cluster_bootstrap_auc.npy", boot)
    summary = {"design": "leave-one-donor-pair-out; all fitting confined to 24 training pairs per fold",
               "n_folds": len(pair_ids), "n_donor_pairs": len(pair_ids), "n_samples": len(pred),
               "auc": float(auc), "cluster_bootstrap_95ci": [float(lo), float(hi)],
               "bootstrap_unit": "donor pair", "n_bootstrap": N_BOOT, "seed": SEED,
               "runtime_seconds": time.time()-start}
    (RESULTS / "lopocv_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    # Fit the deployable final model with exactly the same fixed selection rule.
    z, scaler_params = fit_cohort_scalers(expr, meta, meta.index.tolist())
    genes, details = select_features(z, meta, meta.index.tolist(), SEED)
    model = fit_final_model(z, meta, meta.index.tolist(), genes)
    details["consensus"].to_csv(RESULTS / "final_full_data_consensus_ranks.csv", index=False)
    details["lasso_tuning"].to_csv(RESULTS / "final_full_data_lasso_tuning.csv", index=False)
    final = {"genes": genes, "intercept": float(model.intercept_[0]),
             "coefficients": dict(zip(genes, model.coef_[0].astype(float))),
             "best_lasso_C": details["best_lasso_C"],
             "lasso_genes": details["methods"]["LASSO"],
             "svm_rfe_genes": details["methods"]["SVM_RFE"],
             "random_forest_genes": details["methods"]["Random_forest"],
             "three_way_intersection": sorted(set(details["methods"]["LASSO"]) &
                                               set(details["methods"]["SVM_RFE"]) &
                                               set(details["methods"]["Random_forest"])),
             "classifier": "L2 logistic regression, liblinear, C=1; score is not a calibrated clinical probability",
             "scaling": "gene-wise mean and sample SD fitted separately within each development study; external cohort uses its own label-free gene-wise z scaling",
             "seed": SEED}
    (RESULTS / "final_signature_model.json").write_text(json.dumps(final, indent=2))
    for study, params in scaler_params.items():
        params.to_csv(RESULTS / f"final_scaler_{study}.csv", index_label="gene")
    freq = consensus_all[consensus_all.in_final_six].groupby("gene").agg(
        folds_selected=("fold", "nunique"), mean_borda=("borda_score", "mean"),
        mean_abs_paired_t=("abs_paired_t", "mean")).sort_values(
        ["folds_selected", "mean_borda"], ascending=False)
    freq["selection_frequency"] = freq.folds_selected / len(pair_ids)
    freq.to_csv(RESULTS / "six_gene_selection_stability.csv", index_label="gene")
    print("FINAL MODEL", json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
