#!/usr/bin/env python3
"""Generate every manuscript figure directly from saved analysis outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # deterministic headless rendering; no desktop authorization needed
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Bbox
from sklearn.metrics import roc_curve, auc
from lifelines import KaplanMeierFitter
from geo_io import ROOT

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
RES = ROOT / "results"
SOURCE = ROOT / "data" / "source"
COL = {"blue":"#2878B5","red":"#C82423","orange":"#F28E2B","green":"#2A9D8F",
       "gray":"#7A7A7A","light":"#DCE6F1","purple":"#7A5195","dark":"#25324B"}
plt.rcParams.update({"font.size":8,"axes.titlesize":9,"axes.labelsize":8,
                     "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":120,"savefig.bbox":"tight"})


def save(fig, name, manifest, panels, bbox_inches="tight"):
    # bbox_inches=None would otherwise fall back to rcParams['savefig.bbox']='tight'.
    # An explicit figure-sized Bbox plus zero padding preserves the exact physical canvas.
    exact_canvas = bbox_inches is None
    bbox = Bbox.from_bounds(0, 0, fig.get_figwidth(), fig.get_figheight()) if exact_canvas else bbox_inches
    for ext, dpi in [("png",300),("pdf",300),("tiff",600)]:
        kwargs = {"pil_kwargs":{"compression":"tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(OUT / f"{name}.{ext}", dpi=dpi, facecolor="white", bbox_inches=bbox,
                    pad_inches=0 if exact_canvas else 0.1, **kwargs)
    plt.close(fig)
    manifest[name] = {"panels":panels,"files":[f"{name}.{x}" for x in ["png","tiff","pdf"]]}
    print(f"Saved {name}", flush=True)


def panel(ax, letter, title):
    ax.text(-0.16,1.06,letter,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top")
    ax.set_title(title,loc="left",fontweight="bold",pad=8)


def volcano(ax, data, title, labels, highlight, offsets=None):
    offsets = offsets or {}
    x, q = data.log2FC, data.FDR_BH.clip(lower=1e-300)
    sig = data.index.isin(highlight)
    ax.scatter(x[~sig],-np.log10(q[~sig]),s=5,c="#B9C2CC",alpha=.45,rasterized=True)
    ax.scatter(x[sig],-np.log10(q[sig]),s=16,c=COL["red"],alpha=.85,edgecolor="white",linewidth=.25)
    ax.axvline(-1,color=COL["gray"],ls="--",lw=.6);ax.axvline(1,color=COL["gray"],ls="--",lw=.6)
    ax.axhline(-np.log10(.05),color=COL["gray"],ls="--",lw=.6)
    for gene in labels:
        if gene in data.index:
            row=data.loc[gene];dx,dy=offsets.get(gene,(3,3))
            ax.annotate(gene,(row.log2FC,-np.log10(max(row.FDR_BH,1e-300))),xytext=(dx,dy),textcoords="offset points",
                        fontsize=6,fontweight="bold",ha="right" if dx<0 else "left")
    ax.set(xlabel="Paired log2 fold change (lesional - non-lesional)",ylabel="-log10(FDR)",title=title)


def figure1(manifest):
    r1=pd.read_csv(RES/"differential_expression/GSE65127_paired_DE_all_genes.csv",index_col=0)
    r2=pd.read_csv(RES/"differential_expression/GSE75819_paired_DE_all_genes.csv",index_col=0)
    rep=pd.read_csv(RES/"differential_expression/cross_cohort_paired_DE.csv",index_col=0)
    genes14=rep.index[rep.replicated_FDR05_same_direction].tolist()
    genes6=rep.index[rep.replicated_large_effect].tolist()
    b=pd.read_csv(RES/"differential_expression/GSE65127_BHLHE41_sample_values.csv",index_col=0)
    summary=json.loads((RES/"differential_expression/differential_and_BHLHE41_summary.json").read_text())
    corr=pd.read_csv(RES/"differential_expression/GSE65127_BHLHE41_correlations.csv")
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.5),constrained_layout=True)
    panel(axs[0,0],"A","GSE65127: 10 donor pairs")
    volcano(axs[0,0],r1,"",genes6+["BHLHE41"],genes14,offsets={"MLANA":(4,8),"DCT":(4,-10)})
    panel(axs[0,1],"B","GSE75819: 15 donor pairs")
    volcano(axs[0,1],r2,"",["DCT","CA14","TYR"],genes14,offsets={"DCT":(4,8),"CA14":(4,-10),"TYR":(4,-10)})
    ax=axs[1,0];panel(ax,"C","BHLHE41 across clinical states")
    order=["Healthy","NonLesional","PeriLesional","Lesional"];labels=["Healthy","Non-lesional","Peri-lesional","Lesional"]
    positions=np.arange(4)
    for i,(status,label) in enumerate(zip(order,labels)):
        vals=b.loc[b.status==status,"expression"]
        ax.boxplot(vals,positions=[i],widths=.48,showfliers=False,patch_artist=True,
                   boxprops={"facecolor":COL["light"],"edgecolor":COL["blue"]},
                   medianprops={"color":COL["dark"],"lw":1.4},whiskerprops={"color":COL["blue"]},capprops={"color":COL["blue"]})
        jitter=np.linspace(-.10,.10,len(vals));ax.scatter(np.full(len(vals),i)+jitter,vals,s=13,c=COL["blue"],alpha=.75,zorder=3)
    for donor,g in b[b.status!="Healthy"].groupby("donor"):
        g=g.set_index("status").loc[order[1:]];ax.plot([1,2,3],g.expression,c=COL["gray"],alpha=.28,lw=.55,zorder=1)
    ax.set_xticks(positions,labels,rotation=18,ha="right");ax.set_ylabel("BHLHE41 log2 expression")
    ax.text(.02,.02,f"Donor random-intercept model: β = {summary['BHLHE41_mixed_model_beta_per_status']:.3f}/state; P = {summary['BHLHE41_mixed_model_p']:.3f}\nPaired lesional vs non-lesional: Δ = {summary['BHLHE41_paired_log2FC']:.3f}; P = {summary['BHLHE41_paired_p']:.3f}",
            transform=ax.transAxes,fontsize=6.8,va="bottom",bbox={"boxstyle":"round,pad=.3","fc":"white","ec":"#CCCCCC"})
    ax=axs[1,1];panel(ax,"D","BHLHE41 expression correlations in GSE65127")
    corr=corr.set_index("comparator").loc[["MITF","KIT","AURKA","mature_melanocyte_program"]].reset_index()
    names=["MITF","KIT","AURKA","Mature melanocyte\nprogram"]
    colors=[COL["blue"] if v>=0 else COL["red"] for v in corr.rho]
    y=np.arange(len(corr));ax.barh(y,corr.rho,color=colors,height=.58)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.axvline(0,c=COL["dark"],lw=.6);ax.set_xlabel("Spearman ρ with BHLHE41")
    ax.set_xlim(min(-.5,corr.rho.min()-.12),max(.55,corr.rho.max()+.12))
    for i,row in corr.iterrows():
        # Keep negative-bar labels out of the y-axis tick labels.
        x = row.rho + .018 if row.rho >= 0 else .018
        ax.text(x,i,f"ρ={row.rho:.3f}\nP={row.p:.3g}",va="center",ha="left",fontsize=6.5,
                bbox={"fc":"white","ec":"none","alpha":.85,"pad":.4})
    save(fig,"Figure_1",manifest,{"A":"GSE65127 paired volcano","B":"GSE75819 paired volcano","C":"BHLHE41 states and mixed model","D":"BHLHE41 correlations"})


def figure2(manifest):
    stability=pd.read_csv(RES/"machine_learning/six_gene_selection_stability.csv")
    pred=pd.read_csv(RES/"machine_learning/leave_one_pair_out_predictions.csv")
    cv=json.loads((RES/"machine_learning/lopocv_summary.json").read_text())
    model=json.loads((RES/"machine_learning/final_signature_model.json").read_text())
    ext=pd.read_csv(RES/"external_validation/GSE53146_sample_scores.csv")
    es=json.loads((RES/"external_validation/GSE53146_validation_summary.json").read_text())
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.2),constrained_layout=True)
    ax=axs[0,0];panel(ax,"A","Feature stability across 25 outer folds")
    top=stability.head(12).sort_values("selection_frequency")
    ax.barh(np.arange(len(top)),top.selection_frequency,color=[COL["red"] if g=="DCT" else COL["blue"] for g in top.gene])
    ax.set_yticks(np.arange(len(top)),top.gene);ax.set_xlim(0,1.05);ax.set_xlabel("Frequency in final six-gene fold model")
    for i,v in enumerate(top.selection_frequency):ax.text(v+.015,i,f"{v:.0%}",va="center",fontsize=6.5)
    ax.text(.98,.03,"DCT: 25/25 folds\nFull-data three-method intersection: DCT",transform=ax.transAxes,ha="right",va="bottom",fontsize=6.7,
            bbox={"boxstyle":"round,pad=.3","fc":"white","ec":"#CCCCCC"})
    ax=axs[0,1];panel(ax,"B","Leakage-controlled leave-one-pair-out ROC")
    fpr,tpr,_=roc_curve(pred.y,pred.decision_score);ax.plot(fpr,tpr,c=COL["blue"],lw=1.8);ax.plot([0,1],[0,1],"--",c=COL["gray"],lw=.8)
    ax.fill_between(fpr,tpr,fpr,color=COL["light"],alpha=.5);ax.set(xlabel="False-positive rate",ylabel="True-positive rate",xlim=(0,1),ylim=(0,1.02))
    ax.text(.98,.05,f"AUC = {cv['auc']:.4f}\n95% CI {cv['cluster_bootstrap_95ci'][0]:.4f}–{cv['cluster_bootstrap_95ci'][1]:.4f}\nDonor-pair bootstrap, 2,000×",transform=ax.transAxes,ha="right",va="bottom",fontsize=7)
    ax=axs[1,0];panel(ax,"C","External validation in GSE53146")
    rng=np.random.default_rng(20261117)
    for i,(group,color) in enumerate([("Control",COL["blue"]),("Vitiligo",COL["red"])]):
        vals=ext.loc[ext.group==group,"vitiligo_discrimination_score"];x=np.full(len(vals),i)+rng.uniform(-.10,.10,len(vals))
        ax.scatter(x,vals,s=28,c=color,edgecolor="white",linewidth=.4,zorder=3)
        ax.hlines(vals.mean(),i-.23,i+.23,color=COL["dark"],lw=1.3)
    ax.set_xticks([0,1],["Control\n(n=5)","Vitiligo\n(n=5)"]);ax.set_ylabel("Locked six-gene vitiligo score")
    ax.text(.03,.97,f"Mann–Whitney U = {es['mann_whitney_U']:.0f}\nTwo-sided exact P = {es['two_sided_exact_p']:.5f}\nHedges g = {es['hedges_g']:.3f}",transform=ax.transAxes,va="top",fontsize=7)
    ax=axs[1,1];panel(ax,"D","External ROC and effect size")
    fpr,tpr,_=roc_curve(ext.y,ext.vitiligo_discrimination_score);ax.plot(fpr,tpr,c=COL["red"],lw=1.8,drawstyle="steps-post");ax.plot([0,1],[0,1],"--",c=COL["gray"],lw=.8)
    ax.set(xlabel="False-positive rate",ylabel="True-positive rate",xlim=(-.02,1),ylim=(0,1.02))
    ax.text(.98,.05,f"AUC = {es['auc']:.3f}\n95% CI {es['auc_95ci_stratified_bootstrap'][0]:.3f}–{es['auc_95ci_stratified_bootstrap'][1]:.3f}\nCliff’s δ = {es['cliffs_delta']:.3f} (95% CI {es['cliffs_delta_95ci_stratified_bootstrap'][0]:.3f}–{es['cliffs_delta_95ci_stratified_bootstrap'][1]:.3f})",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=7)
    terms = [f"({model['coefficients'][g]:+.4f})×{g}" for g in model["genes"]] + [f"({model['intercept']:+.4f})"]
    fig.text(.5,-.035,"Final locked model: " + " + ".join(terms) + "; gene-wise z scores.",
             ha="center",fontsize=6.5)
    save(fig,"Figure_2",manifest,{"A":"Fold feature stability","B":"LOPOCV ROC","C":"External scores","D":"External ROC"})


def km_plot(ax,data,gene,title,color):
    for group,c in [("High",color),("Low",COL["blue"])]:
        d=data[data[gene+"_median_group"]==group]
        km=KaplanMeierFitter().fit(d.time_months,d.event,label=f"{group} (n={len(d)})")
        km.plot_survival_function(ax=ax,ci_show=True,color=c,lw=1.4)
    p=data.loc[0,gene+"_logrank"] if gene+"_logrank" in data else None
    ax.set(xlabel="Months",ylabel="Overall survival",ylim=(0,1.02),title=title)
    ax.legend(frameon=False,loc="lower left")


def figure3(manifest):
    gray=pd.read_csv(RES/"graying/Bian_vitiligo_31_gene_direction_comparison.csv")
    gs=json.loads((RES/"graying/graying_concordance_summary.json").read_text())
    tcga=pd.read_csv(RES/"melanoma/TCGA_SKCM_459_patient_survival_analysis.csv",index_col=0)
    corr=pd.read_csv(RES/"melanoma/TCGA_SKCM_472_sample_correlations.csv")
    surv=pd.read_csv(RES/"melanoma/TCGA_SKCM_survival_summary.csv").set_index("gene")
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.5),constrained_layout=True)
    ax=axs[0,0];panel(ax,"A","Graying versus vitiligo effect directions")
    colors=np.where(gray.direction_concordant,COL["green"],COL["orange"])
    ax.scatter(gray.log2_white_vs_black,gray.vitiligo_sample_size_weighted_log2FC,s=25,c=colors,alpha=.85,edgecolor="white",linewidth=.35)
    ax.axhline(0,c=COL["gray"],lw=.6);ax.axvline(0,c=COL["gray"],lw=.6)
    for gene in ["BHLHE41","KIT","TRPM1","PLP1"]:
        row=gray[gray.gene==gene]
        if len(row):
            r=row.iloc[0];ax.scatter(r.log2_white_vs_black,r.vitiligo_sample_size_weighted_log2FC,marker="*" if gene=="BHLHE41" else "o",s=85 if gene=="BHLHE41" else 28,c=COL["red"],zorder=4)
            ax.annotate(gene,(r.log2_white_vs_black,r.vitiligo_sample_size_weighted_log2FC),xytext=(4,3),textcoords="offset points",fontsize=6.5,fontweight="bold")
    ax.set(xlabel="Graying log2(white / black hair)",ylabel="Vitiligo weighted paired log2FC")
    ax.text(.02,.98,f"{gs['concordant']}/{gs['measurable_in_both_vitiligo_studies']} concordant ({gs['proportion']:.1%})\nExact two-sided P = {gs['two_sided_exact_binomial_p']:.4f}",transform=ax.transAxes,va="top",fontsize=7,
            bbox={"boxstyle":"round,pad=.25","fc":"white","ec":"none","alpha":.85})
    ax=axs[0,1];panel(ax,"B","BHLHE41 and the melanocytic program in TCGA-SKCM")
    # Draw exactly the 472 expression samples used for the annotated correlation.
    expression = pd.read_csv(SOURCE/"TCGA_SKCM_cBioPortal_expression_RSEM.csv",index_col=0)
    expression = np.log2(np.maximum(expression,0)+1)
    program_genes = ["MITF","DCT","MLANA","TYRP1","PMEL","TYR"]
    expression["melanocytic_program"] = expression[program_genes].apply(lambda c:(c-c.mean())/c.std(ddof=1)).mean(axis=1)
    ax.scatter(expression.melanocytic_program,expression.BHLHE41,s=8,c=COL["blue"],alpha=.32,rasterized=True)
    slope,intercept=np.polyfit(expression.melanocytic_program,expression.BHLHE41,1)
    xx=np.linspace(expression.melanocytic_program.min(),expression.melanocytic_program.max(),100)
    ax.plot(xx,slope*xx+intercept,c=COL["red"],lw=1.2)
    row=corr[(corr.predictor=="BHLHE41")&(corr.target=="melanocytic_program")].iloc[0]
    ax.set(xlabel="Melanocytic program (mean gene-wise z)",ylabel="BHLHE41 log2(RSEM + 1)")
    ax.text(.03,.97,f"All-expression-sample correlation:\nρ = {row.rho:.3f}; P = {row.p:.2e}; n = {int(row.n)}",transform=ax.transAxes,va="top",fontsize=7)
    for ax,gene,letter,title,color in [(axs[1,0],"BHLHE41","C","BHLHE41: overall survival",COL["green"]),(axs[1,1],"AURKA","D","AURKA: overall survival",COL["red"])]:
        panel(ax,letter,title);km_plot(ax,tcga,gene,"",color)
        s=surv.loc[gene];ax.text(.97,.94,f"HR/SD = {s.HR_per_SD:.3f} (95% CI {s.HR95_low:.3f}–{s.HR95_high:.3f})\nCox P = {s.cox_p:.4f}; log-rank P = {s.median_split_logrank_p:.4f}",
                                 transform=ax.transAxes,ha="right",va="top",fontsize=6.6)
    save(fig,"Figure_3",manifest,{"A":"Graying concordance","B":"TCGA correlation","C":"BHLHE41 survival","D":"AURKA survival"})


def figure4(manifest):
    d=pd.read_csv(SOURCE/"Bian_2019_Tables_III_IV_40_unique_genes.csv")
    top=d.reindex(d.log2_white_vs_black.abs().sort_values(ascending=False).index).head(11)
    top=pd.concat([top,d[d.gene=="BHLHE41"]]).drop_duplicates("gene").sort_values("log2_white_vs_black")
    fig,ax=plt.subplots(figsize=(7.2,4.4),constrained_layout=True)
    colors=[COL["red"] if g=="BHLHE41" else COL["blue"] if v<0 else COL["orange"] for g,v in zip(top.gene,top.log2_white_vs_black)]
    ax.barh(np.arange(len(top)),top.log2_white_vs_black,color=colors,height=.65)
    labels=[g+" ★" if g=="BHLHE41" else g for g in top.gene];ax.set_yticks(np.arange(len(top)),labels)
    ax.axvline(0,c=COL["dark"],lw=.7);ax.set_xlabel("Published log2(white-hair FPKM / black-hair FPKM)")
    ax.set_title("Bian et al. (2019): group-level gray-hair transcriptomic evidence",loc="left",fontweight="bold")
    for i,v in enumerate(top.log2_white_vs_black):ax.text(v+(.10 if v>=0 else -.10),i,f"{v:+.2f}",va="center",ha="left" if v>=0 else "right",fontsize=6.5)
    ax.text(.99,.02,"Published Tables III–IV; no donor-level raw data reanalysis",transform=ax.transAxes,ha="right",fontsize=7,color=COL["gray"])
    save(fig,"Figure_4",manifest,{"A":"Published graying effect sizes for 11 largest changes plus BHLHE41"})


def figure_s1(manifest):
    fig,ax=plt.subplots(figsize=(10,5.6));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis("off")
    def box(x,y,w,h,text,fc,ec=COL["dark"],fs=8):
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08,rounding_size=.08",fc=fc,ec=ec,lw=1.0);ax.add_patch(p)
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs);return p
    def arrow(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,lw=1.0,color=COL["gray"]))
    ax.text(5,5.75,"Reproducible Analysis Workflow",ha="center",fontsize=15,fontweight="bold",color=COL["dark"])
    box(.25,4.15,2.35,1.0,"GSE65127\n40 samples; 10 patient triads\n+ 10 healthy controls",COL["light"])
    box(.25,2.75,2.35,1.0,"GSE75819\n30 samples; 15 matched\nlesional/non-lesional pairs",COL["light"])
    box(.25,1.35,2.35,1.0,"GSE53146\n5 vitiligo + 5 controls\nExternal biological cohort", "#FCE4DE")
    box(3.35,3.55,2.55,1.35,"Study-stratified processing\nProbe annotation; median collapse\nWithin-study paired tests\n13,511 common genes", "#E8F3F1")
    box(3.35,1.65,2.55,1.15,"Leakage-controlled modeling\n25 outer donor-pair folds\nAll selection and scaling in-fold", "#E8F3F1")
    box(6.65,3.65,2.85,1.15,"Final locked six-gene model\nDCT, TYR, MC1R, FAM24B,\nWFS1, RPL31", "#FFF2CC")
    box(6.65,1.75,2.85,1.05,"External ranking validation\nAUC 0.960 (95% CI 0.760–1.000)\nExact P = 0.01587", "#FCE4DE")
    box(3.35,.15,2.55,.85,"Bian et al. Tables III–IV\n40 unique genes; 31 measurable\nDirection concordance", "#F0EAF5")
    box(6.65,.15,2.85,.85,"TCGA-SKCM + CCLE 2019\n472 expression / 459 survival\n58 melanoma cell lines", "#F0EAF5")
    arrow(2.6,4.65,3.35,4.3);arrow(2.6,3.25,3.35,3.9);arrow(4.6,3.55,4.6,2.8);arrow(5.9,2.25,6.65,2.25)
    arrow(5.9,4.2,6.65,4.2);arrow(7.95,3.65,7.95,2.8);arrow(2.6,1.85,3.35,.75);arrow(5.9,.58,6.65,.58)
    ax.text(5,5.25,"Public data → explicit sample design → fold-isolated fitting → locked external score → cross-disease context",ha="center",fontsize=9,color=COL["gray"])
    save(fig,"Figure_S1",manifest,{"A":"Complete analysis flowchart with explicit sample roles"})


def figure_s2(manifest):
    d=pd.read_csv(RES/"melanoma/CCLE_2019_58_melanoma_log2_expression.csv",index_col=0)
    raw=pd.read_csv(SOURCE/"CCLE_2019_cutaneous_melanoma_expression_RPKM.csv",index_col=0)
    corr=pd.read_csv(RES/"melanoma/CCLE_2019_58_melanoma_BHLHE41_correlations.csv")
    summary=json.loads((RES/"melanoma/CCLE_2019_summary.json").read_text())
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.6),constrained_layout=True)
    ax=axs[0];panel(ax,"A","BHLHE41 and MITF")
    ax.scatter(d.MITF,d.BHLHE41,s=19,c=COL["blue"],alpha=.7,edgecolor="white",linewidth=.25)
    slope,intercept=np.polyfit(d.MITF,d.BHLHE41,1);xx=np.linspace(d.MITF.min(),d.MITF.max(),100);ax.plot(xx,slope*xx+intercept,c=COL["red"],lw=1.2)
    row=corr[corr.gene=="MITF"].iloc[0];ax.set(xlabel="MITF log2(RPKM + 1)",ylabel="BHLHE41 log2(RPKM + 1)")
    ax.text(.03,.97,f"n = {int(row.n)}; ρ = {row.rho:.3f}\nP = {row.p:.2e}",transform=ax.transAxes,va="top",fontsize=7)
    ax=axs[1];panel(ax,"B","Melanocyte-lineage correlations")
    c=corr.copy();c["order"]=c.gene.map({g:i for i,g in enumerate(["MITF","PMEL","MLANA","TYR","TYRP1","DCT","KIT","AURKA"])});c=c.sort_values("order",ascending=False)
    ax.barh(np.arange(len(c)),c.rho,color=[COL["blue"] if p<.05 else COL["gray"] for p in c.p],height=.62)
    ax.set_yticks(np.arange(len(c)),c.gene);ax.axvline(0,c=COL["dark"],lw=.6);ax.set_xlim(-.15,.9);ax.set_xlabel("Spearman ρ with BHLHE41")
    for i,row in enumerate(c.itertuples()):ax.text(row.rho+.015,i,f"{row.rho:.3f}; P={row.p:.2g}",va="center",fontsize=6)
    ax=axs[2];panel(ax,"C","MITF median-split groups")
    high=raw.MITF>raw.MITF.median();vals=[np.log2(raw.loc[~high,"BHLHE41"]+1),np.log2(raw.loc[high,"BHLHE41"]+1)]
    bp=ax.boxplot(vals,tick_labels=["MITF low","MITF high"],patch_artist=True,showfliers=False,widths=.55,
                  boxprops={"edgecolor":COL["dark"]},medianprops={"color":COL["dark"],"lw":1.4})
    for patch,color in zip(bp["boxes"],[COL["blue"],COL["red"]]):patch.set_facecolor(color);patch.set_alpha(.35)
    rng=np.random.default_rng(20261117)
    for i,v in enumerate(vals):ax.scatter(np.full(len(v),i+1)+rng.uniform(-.12,.12,len(v)),v,s=13,c=[COL["blue"],COL["red"]][i],alpha=.65)
    ax.set_ylabel("BHLHE41 log2(RPKM + 1)")
    ax.text(.03,.97,f"Mean RPKM: {summary['BHLHE41_mean_MITF_low_RPKM']:.2f} vs {summary['BHLHE41_mean_MITF_high_RPKM']:.2f}\nFold ratio = {summary['mean_fold_ratio']:.2f}\nP = {summary['mann_whitney_two_sided_asymptotic_p']:.2e}",transform=ax.transAxes,va="top",fontsize=6.5)
    save(fig,"Figure_S2",manifest,{"A":"CCLE BHLHE41-MITF scatter, n=58","B":"CCLE lineage correlations","C":"MITF high/low BHLHE41"})


def graphical_abstract(manifest):
    # Exact journal canvas: 17 cm × 5 cm. Do not tighten the bounding box.
    fig = plt.figure(figsize=(17/2.54, 5/2.54))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0,17); ax.set_ylim(0,5); ax.axis("off")
    ax.add_patch(FancyBboxPatch((.05,.05),16.9,4.9,boxstyle="round,pad=.08,rounding_size=.12",fc="#F8FAFC",ec=COL["dark"],lw=.8))
    def box(x,y,w,h,title,body,fc):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08,rounding_size=.10",fc=fc,ec="white",lw=.8))
        ax.text(x+.16,y+h-.30,title,fontsize=6.5,fontweight="bold",va="top",color=COL["dark"])
        ax.text(x+.16,y+h-.82,body,fontsize=5.3,va="top",color=COL["dark"],linespacing=1.35)
    def arrow(x):ax.add_patch(FancyArrowPatch((x,2.5),(x+.48,2.5),arrowstyle="-|>",mutation_scale=9,lw=.9,color=COL["gray"]))
    box(.35,1.0,3.45,3.0,"PUBLIC TRANSCRIPTOMES","Vitiligo: 25 donor pairs\nExternal: 5 cases + 5 controls\nGraying, TCGA-SKCM and CCLE",COL["light"])
    arrow(3.9)
    box(4.45,1.0,3.65,3.0,"NO INFORMATION LEAKAGE","Leave one donor pair out\nRepeat scaling and selection\nFit coefficients in-fold only", "#E8F3F1")
    arrow(8.2)
    box(8.75,1.0,3.55,3.0,"LOCKED SIX-GENE SCORE","DCT • TYR • MC1R\nFAM24B • WFS1 • RPL31\nOuter-fold AUC 0.9872", "#FFF2CC")
    arrow(12.4)
    box(12.95,1.0,3.65,3.0,"EXTERNAL VALIDATION","AUC 0.960\n95% CI 0.760–1.000\nBHLHE41: lineage-associated;\nnot prognostic in TCGA", "#FCE4DE")
    ax.text(8.5,.55,"Expression associations support hypotheses; they do not establish cell abundance, causality or clinical utility.",ha="center",fontsize=5.0,color=COL["gray"])
    save(fig,"Graphical_Abstract",manifest,{"A":"Exact 17 cm × 5 cm graphical abstract canvas"},bbox_inches=None)
    (OUT/"Graphical_Abstract_caption.txt").write_text("Leakage-controlled analysis of public transcriptomes identifies a DCT-centered six-gene vitiligo score and situates BHLHE41 within melanocyte-lineage expression programs across vitiligo, hair graying and melanoma.")


def main():
    manifest={}
    figure1(manifest);figure2(manifest);figure3(manifest);figure4(manifest)
    figure_s1(manifest);figure_s2(manifest);graphical_abstract(manifest)
    (OUT/"figure_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Generated",len(manifest),"figures")


if __name__ == "__main__": main()
