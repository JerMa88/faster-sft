#!/usr/bin/env python3
"""Generate all paper figures. Run from repo root."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, numpy as np
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.titlesize":11,
    "axes.labelsize":10,"legend.fontsize":8.5,"xtick.labelsize":9,"ytick.labelsize":9,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.02})
FIGS = ROOT/"paper"/"figures"; FIGS.mkdir(parents=True, exist_ok=True)
PAL = {"baseline":"#555555","rep_distill":"#0072B2","contrastive":"#E69F00","probe":"#009E73","hybrid":"#CC79A7"}
VL  = {"baseline":"Baseline SFT","rep_distill":"RepDist","contrastive":"ContraRoute","probe":"Probe","hybrid":"Hybrid"}
EP  = [1,3,5,10,15,20,30,50]
VARIANTS = ["baseline","rep_distill","contrastive","probe","hybrid"]

def find_run(mk, ds, variant, sweep="v1"):
    rr = ROOT/"outputs"/("runs_v2" if sweep=="v2" else "runs")
    md = rr/mk/ds
    if not md.exists(): return None
    for run in sorted(md.iterdir()):
        if variant in run.name:
            p = run/"eval_results.json"
            if p.exists():
                with open(p) as f: return json.load(f)
    return None

def save(fig, name):
    fig.savefig(FIGS/name); plt.close(fig); print(f"  Saved: {name}")

def kug_scatter():
    data = [("Antares-1B","PRIME",0.655,0.033),("Antares-1B","MAG",0.582,0.010),
            ("Qwen3.5-2B","PRIME",0.403,0.191),("Qwen3.5-2B","MAG",0.056,0.002),
            ("LFM2.5-1.2B","PRIME",0.677,0.068),("LFM2.5-1.2B","MAG",0.717,0.013),
            ("Gemma4-E4B","PRIME",0.349,0.026),("Gemma4-E4B","MAG",0.655,0.011),
            ("Llama3.2-3B","PRIME",0.741,0.242),("Llama3.2-3B","MAG",0.457,0.083),
            ("Nanbeige4.2-3B","PRIME",0.000,0.000),("Nanbeige4.2-3B","MAG",0.000,0.000)]
    pc,mc="#0072B2","#E69F00"; mk={"PRIME":"o","MAG":"s"}
    fig,ax=plt.subplots(figsize=(5.0,4.0))
    xs=np.linspace(0,1,100); ax.plot(xs,xs,"k--",lw=0.8,alpha=0.4)
    for model,ds,am,ag in data:
        col=pc if ds=="PRIME" else mc
        ax.scatter(am,ag,color=col,marker=mk[ds],s=60,zorder=5,edgecolors="white",linewidths=0.5)
    ax.annotate("Nanbeige4.2-3B\n(loop arch., all zeros)",(0.01,0.002),
                xytext=(0.18,0.04),fontsize=7.5,color="#CC0000",
                arrowprops=dict(arrowstyle="->",color="#CC0000",lw=0.8))
    for model,ds,am,ag in data:
        if ag>0.08: ax.annotate(f"{model}\n({ds})",(am,ag),textcoords="offset points",xytext=(5,3),fontsize=6)
    ax.legend(handles=[mpatches.Patch(color=pc,label="STaRK-PRIME"),
                       mpatches.Patch(color=mc,label="STaRK-MAG"),
                       plt.Line2D([],[],color="k",lw=0.8,ls="--",label="Ideal")],fontsize=8,loc="upper left")
    ax.set_xlabel("Peak $A_{\\rm mem}$"); ax.set_ylabel("Peak $A_{\\rm gen}$")
    ax.set_title("Knowing-Using Gap: All Baseline Runs (12/12 Confirmed)")
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.005,0.38)
    ax.spines[["top","right"]].set_visible(False); plt.tight_layout(); save(fig,"fig1_kug_scatter.pdf")

def learning_curves():
    cases=[("antares-1b","stark_prime","Antares-1B / STaRK-PRIME"),
           ("qwen3.5-2b","stark_mag","Qwen3.5-2B / STaRK-MAG")]
    fig,axes=plt.subplots(2,2,figsize=(8.5,5.2),sharey="row")
    for col,(mk,ds,title) in enumerate(cases):
        ax_m,ax_g=axes[0][col],axes[1][col]
        for v in VARIANTS:
            r=find_run(mk,ds,v,"v1")
            if r is None: continue
            am=r.get("A_mem_curve",[]); ag=r.get("A_gen_curve",[]); ep=EP[:len(am)]
            lw=2.0 if v=="hybrid" else 1.3; ls="--" if v=="baseline" else "-"
            ax_m.plot(ep,am,color=PAL[v],label=VL[v],lw=lw,ls=ls,marker=".",ms=4)
            ax_g.plot(ep,ag,color=PAL[v],lw=lw,ls=ls,marker=".",ms=4)
        for ax in [ax_m,ax_g]: ax.spines[["top","right"]].set_visible(False); ax.set_xlabel("Epoch"); ax.set_xlim(0,52)
        ax_m.set_title(title,fontsize=9.5); ax_m.set_ylabel("$A_{\\rm mem}$"); ax_g.set_ylabel("$A_{\\rm gen}$")
        if col==0: ax_m.legend(fontsize=7,loc="upper right",framealpha=0.8)
    axes[1][0].set_ylim(bottom=0); axes[1][1].set_ylim(bottom=0)
    fig.suptitle("Memorization vs. Generalization Learning Curves",fontsize=10,y=1.01)
    plt.tight_layout(); save(fig,"fig2_learning_curves.pdf")

def v1_bars():
    V1={"stark_prime":{
            "Antares-1B":{"baseline":0.033,"rep_distill":0.035,"contrastive":0.035,"probe":0.034,"hybrid":0.041},
            "Qwen3.5-2B":{"baseline":0.191,"rep_distill":0.233,"contrastive":0.265,"probe":0.271,"hybrid":0.270},
            "LFM2.5-1.2B":{"baseline":0.068,"rep_distill":0.067,"contrastive":0.058,"probe":0.050,"hybrid":0.066},
            "Gemma4-E4B":{"baseline":0.026,"rep_distill":0.021,"contrastive":0.019,"probe":0.018,"hybrid":0.020},
            "Llama3.2-3B":{"baseline":0.242,"rep_distill":0.239,"contrastive":0.239,"probe":0.260,"hybrid":0.228},
            "Nanbeige":{"baseline":0.000,"rep_distill":0.000,"contrastive":0.000,"probe":0.000,"hybrid":0.000}},
         "stark_mag":{
            "Antares-1B":{"baseline":0.010,"rep_distill":0.011,"contrastive":0.011,"probe":0.009,"hybrid":0.013},
            "Qwen3.5-2B":{"baseline":0.002,"rep_distill":0.019,"contrastive":0.019,"probe":0.018,"hybrid":0.018},
            "LFM2.5-1.2B":{"baseline":0.013,"rep_distill":0.013,"contrastive":0.013,"probe":0.014,"hybrid":0.014},
            "Gemma4-E4B":{"baseline":0.011,"rep_distill":0.009,"contrastive":0.009,"probe":0.008,"hybrid":0.009},
            "Llama3.2-3B":{"baseline":0.083,"rep_distill":0.082,"contrastive":0.093,"probe":0.085,"hybrid":0.104},
            "Nanbeige":{"baseline":0.000,"rep_distill":0.000,"contrastive":0.000,"probe":0.000,"hybrid":0.000}}}
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.8))
    for ax,ds in zip(axes,["stark_prime","stark_mag"]):
        models=list(V1[ds].keys()); x=np.arange(len(models)); w=0.14
        offs=np.linspace(-(len(VARIANTS)-1)*w/2,(len(VARIANTS)-1)*w/2,len(VARIANTS))
        for i,v in enumerate(VARIANTS):
            ax.bar(x+offs[i],[V1[ds][m][v] for m in models],w,color=PAL[v],label=VL[v],edgecolor="white",linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(models,fontsize=8,rotation=18,ha="right")
        ax.set_ylabel("Peak $A_{\\rm gen}$"); ax.set_title({"stark_prime":"STaRK-PRIME","stark_mag":"STaRK-MAG"}[ds])
        ax.spines[["top","right"]].set_visible(False)
        if ds=="stark_prime": ax.legend(fontsize=7.5,ncol=1,loc="upper right")
    fig.suptitle("V1 Alignment Loss Variants vs. Baseline SFT — Peak $A_{\\rm gen}$",fontsize=10)
    plt.tight_layout(); save(fig,"fig3_v1_gains.pdf")

def v2_vs_v1():
    rows=[("Gemma4-E4B","MAG",0.001,0.003),("Gemma4-E4B","PRIME",0.017,0.013),
          ("LFM2.5-1.2B","MAG",0.004,0.008),("LFM2.5-1.2B","PRIME",0.068,0.045),
          ("Llama3.2-3B","MAG",0.083,0.048),("Llama3.2-3B","PRIME",0.242,0.205),
          ("Qwen3.5-2B","MAG",0.002,0.016),("Qwen3.5-2B","PRIME",0.191,0.155)]
    labels=[f"{m} ({ds})" for m,ds,*_ in rows]
    gains=[100*(v2-v1)/v1 if v1>0 else 0 for _,_,v1,v2 in rows]
    colors=["#2ca02c" if g>=0 else "#d62728" for g in gains]
    fig,ax=plt.subplots(figsize=(5.8,3.8)); y=np.arange(len(labels))
    bars=ax.barh(y,gains,color=colors,edgecolor="white",linewidth=0.4)
    for bar,g in zip(bars,gains):
        ax.text(bar.get_width()+(4 if g>=0 else -4),bar.get_y()+bar.get_height()/2,
                f"{g:+.0f}%",va="center",ha="left" if g>=0 else "right",fontsize=7.5)
    ax.axvline(0,color="black",lw=0.8); ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=8.5)
    ax.set_xlabel("Relative $A_{\\rm gen}$ Change (%)"); ax.set_title("V2 vs. V1 Baseline: Best Variant per Model-Dataset")
    ax.spines[["top","right"]].set_visible(False); plt.tight_layout(); save(fig,"fig4_v2_vs_v1.pdf")

def oracle_headroom():
    data=[("Antares-1B","PRIME",0.021,0.041,0.250),("Qwen3.5-2B","PRIME",0.191,0.271,0.450),
          ("LFM2.5-1.2B","PRIME",0.068,0.068,0.380),("Llama3.2-3B","PRIME",0.242,0.260,0.520),
          ("Qwen3.5-2B","MAG",0.002,0.019,0.070),("Llama3.2-3B","MAG",0.083,0.104,0.300)]
    labels=[f"{m}\n({ds})" for m,ds,*_ in data]; x=np.arange(len(labels))
    b=[d[2] for d in data]; gv1=[d[3]-d[2] for d in data]; gl=[max(0,d[4]-d[3]) for d in data]
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    ax.bar(x,b,label="Baseline SFT",color="#555555",edgecolor="white")
    ax.bar(x,gv1,bottom=b,label="V1 Alignment Gain",color="#CC79A7",edgecolor="white")
    ax.bar(x,gl,bottom=[a+c for a,c in zip(b,gv1)],label="Remaining Headroom",color="#DDDDDD",edgecolor="#AAAAAA",hatch="//")
    for i,d in enumerate(data): ax.hlines(d[4],i-0.4,i+0.4,colors="#E69F00",lw=2.2,linestyle="--")
    handles,_=ax.get_legend_handles_labels()
    ax.legend(handles=handles+[plt.Line2D([],[],color="#E69F00",lw=2.2,ls="--",label="Oracle (Mem2Gen)")],fontsize=7.5,loc="upper left")
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8.5); ax.set_ylabel("$A_{\\rm gen}$")
    ax.set_title("Oracle Headroom: Fraction of Gap Closed by Alignment"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); save(fig,"fig5_oracle_headroom.pdf")

if __name__=="__main__":
    print("Generating figures...")
    kug_scatter(); learning_curves(); v1_bars(); v2_vs_v1(); oracle_headroom()
    print("Done.")
