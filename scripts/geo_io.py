#!/usr/bin/env python3
"""Shared, deterministic GEO parsing and probe-to-gene preprocessing."""
from pathlib import Path
import csv, gzip, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ANN = ROOT / "data" / "annotations"
OUT = ROOT / "results" / "intermediate"
OUT.mkdir(parents=True, exist_ok=True)


def read_series_matrix(gse):
    path = RAW / f"{gse}_series_matrix.txt.gz"
    with gzip.open(path, "rt") as handle:
        lines = handle.readlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("!series_matrix_table_begin")) + 1
    end = next(i for i, line in enumerate(lines) if line.startswith("!series_matrix_table_end"))
    from io import StringIO
    expr = pd.read_csv(StringIO("".join(lines[start:end])), sep="\t", index_col=0)
    expr.index = expr.index.astype(str)
    metadata = {}
    for line in lines[:start]:
        if line.startswith("!Sample_"):
            fields = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            metadata[fields[0][8:]] = [x.strip('"') for x in fields[1:]]
    meta = pd.DataFrame(metadata, index=expr.columns)
    return expr, meta


def read_gpl570():
    return pd.read_csv(ANN / "GPL570.annot.gz", sep="\t", skiprows=27, dtype=str, low_memory=False)


def read_gpl6884():
    from io import StringIO
    lines = (ANN / "GPL6884.txt").read_text(errors="replace").splitlines()
    start = lines.index("!platform_table_begin") + 1
    end = lines.index("!platform_table_end")
    table = pd.read_csv(StringIO("\n".join(lines[start:end])), sep="\t", dtype=str, low_memory=False)
    return table[table["ID"].astype(str).str.startswith("ILMN_")].copy()


def read_gpl14951():
    """Read the official GPL14951 ID/Symbol columns with strict file checks."""
    import sys
    csv.field_size_limit(sys.maxsize)
    path = ANN / "GPL14951.txt"
    text = path.read_text(errors="strict")
    if "\x00" in text:
        raise ValueError("NUL bytes indicate a corrupt GPL14951 download")
    lines = text.splitlines()
    if "!platform_table_end" not in lines:
        raise ValueError("Incomplete GPL14951 platform table")
    start = lines.index("!platform_table_begin") + 1
    end = lines.index("!platform_table_end")
    reader = csv.reader(lines[start:end], delimiter="\t")
    header = next(reader)
    symbol_index = header.index("Symbol")
    rows = [(row[0], row[symbol_index]) for row in reader
            if len(row) > symbol_index and row[0].startswith("ILMN_")]
    return pd.DataFrame(rows, columns=["ID", "Symbol"])


def map_probes(expr, annotation, id_col, symbol_col):
    mapping = annotation[[id_col, symbol_col]].dropna().copy()
    mapping[id_col] = mapping[id_col].astype(str)
    mapping[symbol_col] = mapping[symbol_col].astype(str).str.strip()
    mapping = mapping[(mapping[symbol_col] != "") & ~mapping[symbol_col].str.contains("///", regex=False)]
    mapping = mapping.drop_duplicates(id_col).set_index(id_col)[symbol_col]
    selected = expr.loc[expr.index.intersection(mapping.index)].copy()
    selected["gene"] = mapping.loc[selected.index].values
    return selected.groupby("gene", sort=True).median()


def load_development():
    x1, m1 = read_series_matrix("GSE65127")
    x2, m2 = read_series_matrix("GSE75819")
    a = map_probes(x1, read_gpl570(), "ID", "Gene symbol")
    b = map_probes(x2, read_gpl6884(), "ID", "Symbol")
    titles = m1["title"]
    status, donor = [], []
    for title in titles:
        if title.startswith("Healthy"):
            status.append("Healthy")
        elif title.startswith("Non Lesional"):
            status.append("NonLesional")
        elif title.startswith("Peri-Lesional"):
            status.append("PeriLesional")
        elif title.startswith("Lesional"):
            status.append("Lesional")
        else:
            raise ValueError(f"Unrecognized GSE65127 title: {title}")
        donor.append(re.search(r"\[(\d+)_", title).group(1))
    meta1 = pd.DataFrame({"sample": a.columns, "study": "GSE65127", "title": titles.values,
                          "status": status, "donor": donor}).set_index("sample")
    meta1["pair_id"] = "GSE65127_" + meta1["donor"]
    meta2 = pd.DataFrame({"sample": b.columns, "study": "GSE75819",
                          "title": m2["title"].values,
                          "status": ["NonLesional" if i % 2 == 0 else "Lesional" for i in range(b.shape[1])],
                          "donor": [f"D{i//2+1:02d}" for i in range(b.shape[1])]}).set_index("sample")
    meta2["pair_id"] = "GSE75819_" + meta2["donor"]
    # GSE65127 is already log2-scale. GSE75819 contains non-positive intensities;
    # floor at zero before the pre-specified log2(x+1) transformation.
    b = np.log2(np.maximum(b, 0) + 1)
    return a, b, meta1, meta2


def paired_deltas(expr, meta):
    pairs = []
    for pair_id, group in meta.groupby("pair_id", sort=True):
        lesion = group.index[group.status == "Lesional"].tolist()
        normal = group.index[group.status == "NonLesional"].tolist()
        if len(lesion) != 1 or len(normal) != 1:
            raise ValueError(f"Invalid pair: {pair_id}: {len(lesion)}, {len(normal)}")
        pairs.append((pair_id, lesion[0], normal[0]))
    delta = pd.DataFrame({pair: expr[lesion] - expr[normal] for pair, lesion, normal in pairs},
                         index=expr.index)
    return delta, pairs


def classifier_data():
    a, b, m1, m2 = load_development()
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]
    keep1 = m1.status.isin(["Lesional", "NonLesional"])
    m1 = m1[keep1]
    expr = pd.concat([a[m1.index], b[m2.index]], axis=1)
    meta = pd.concat([m1, m2], axis=0)
    meta["label"] = (meta.status == "Lesional").astype(int)
    return expr, meta


def fit_cohort_scalers(expr, meta, train_samples):
    params = {}
    transformed = pd.DataFrame(index=expr.index, columns=expr.columns, dtype=float)
    for study in meta.study.unique():
        tr = meta.index[(meta.study == study) & meta.index.isin(train_samples)]
        all_study = meta.index[meta.study == study]
        mean = expr.loc[:, tr].mean(axis=1)
        sd = expr.loc[:, tr].std(axis=1, ddof=1).replace(0, np.nan)
        if sd.isna().any():
            raise ValueError(f"Zero-variance genes in {study}: {sd.isna().sum()}")
        transformed.loc[:, all_study] = expr.loc[:, all_study].sub(mean, axis=0).div(sd, axis=0)
        params[study] = pd.DataFrame({"mean": mean, "sd": sd})
    if transformed.isna().any().any():
        raise ValueError("Non-finite standardized expression")
    return transformed, params
