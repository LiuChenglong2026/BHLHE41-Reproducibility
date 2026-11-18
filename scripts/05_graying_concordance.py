#!/usr/bin/env python3
"""Reproducible extraction of Bian et al. Tables III–IV and direction analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from lxml import etree
from geo_io import ROOT

SOURCE = ROOT / "data" / "source"
OUT = ROOT / "results" / "graying"
SOURCE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6601371/fullTextXML"


def extract():
    path = SOURCE / "Bian_2019_PMC6601371_fullText.xml"
    if not path.exists():
        import requests
        response = requests.get(URL, timeout=90); response.raise_for_status()
        path.write_bytes(response.content)
    root = etree.parse(str(path))
    rows = []
    for tw in root.xpath(".//table-wrap"):
        label = "".join(tw.xpath("./label//text()")).strip()
        if label not in ["Table III.", "Table IV."]: continue
        for tr in tw.xpath(".//tbody/tr"):
            cells = [" ".join(c.itertext()).strip() for c in tr.xpath("./td")]
            if len(cells) >= 6:
                rows.append({"source_table": label.rstrip("."), "ensembl": cells[0], "gene": cells[1],
                             "black_fpkm": float(cells[2]), "white_fpkm": float(cells[3]),
                             "log2_white_vs_black": float(cells[4].replace("−", "-")),
                             "published_p": float(cells[5])})
    genes = pd.DataFrame(rows).drop_duplicates("gene")
    genes.to_csv(SOURCE / "Bian_2019_Tables_III_IV_40_unique_genes.csv", index=False)
    return genes


def main():
    genes = extract()
    de = pd.read_csv(ROOT / "results" / "differential_expression" / "cross_cohort_paired_DE.csv", index_col=0)
    x = genes[genes.gene.isin(de.index)].copy()
    x["GSE65127_log2FC"] = x.gene.map(de.GSE65127_log2FC)
    x["GSE75819_log2FC"] = x.gene.map(de.GSE75819_log2FC)
    x["vitiligo_sample_size_weighted_log2FC"] = x.gene.map(de.sample_size_weighted_log2FC)
    x["direction_concordant"] = np.sign(x.log2_white_vs_black) == np.sign(x.vitiligo_sample_size_weighted_log2FC)
    x["both_vitiligo_studies_same_direction"] = np.sign(x.GSE65127_log2FC) == np.sign(x.GSE75819_log2FC)
    x = x.sort_values(["direction_concordant", "gene"], ascending=[False, True])
    x.to_csv(OUT / "Bian_vitiligo_31_gene_direction_comparison.csv", index=False)
    k, n = int(x.direction_concordant.sum()), len(x)
    test = stats.binomtest(k, n, .5, alternative="two-sided")
    ci = test.proportion_ci(confidence_level=.95)
    stable = x[x.both_vitiligo_studies_same_direction]
    ks, ns = int(stable.direction_concordant.sum()), len(stable)
    stable_test = stats.binomtest(ks, ns, .5, alternative="two-sided")
    summary = {"source": "Bian et al., Exp Ther Med 2019;18:1155-1163; PMID 31316609; PMCID PMC6601371; Tables III-IV",
               "unique_published_genes": len(genes), "measurable_in_both_vitiligo_studies": n,
               "concordant": k, "proportion": k/n, "two_sided_exact_binomial_p": float(test.pvalue),
               "proportion_95ci_exact": [float(ci.low), float(ci.high)],
               "primary_effect": "sample-size-weighted mean of study-specific paired log2 fold changes (10 and 15 pairs)",
               "sensitivity_both_vitiligo_studies_same_direction_n": ns,
               "sensitivity_concordant": ks,
               "sensitivity_two_sided_exact_binomial_p": float(stable_test.pvalue),
               "interpretation": "Primary cross-disease direction concordance is exploratory and not statistically significant; the same-direction subset is descriptive and subject to selection."}
    (OUT / "graying_concordance_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
