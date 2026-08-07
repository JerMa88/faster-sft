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
    # V1 re-evaluated baseline numbers (relaxed EM)
    data = [("Qwen3.5-2B",  "PRIME", 0.640, 0.244),
            ("Qwen3.5-2B",  "MAG",   0.384, 0.019),
            ("LFM2.5-1.2B", "PRIME", 0.682, 0.068),
            ("LFM2.5-1.2B", "MAG",   0.722, 0.013),
            ("Gemma4-E4B",  "MAG",   0.660, 0.009),
            ("Llama3.2-3B", "PRIME", 0.744, 0.254),
            ("Llama3.2-3B", "MAG",   0.458, 0.086)]
    pc, mc = "#0072B2", "#E69F00"; mk = {"PRIME": "^", "MAG": "o"}
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    xs = np.linspace(0, 1, 100); ax.plot(xs, xs, "k--", lw=0.8, alpha=0.4)
    for model, ds, am, ag in data:
        col = pc if ds == "PRIME" else mc
        ax.scatter(am, ag, color=col, marker=mk[ds], s=65, zorder=5, edgecolors="white", linewidths=0.5)
    for model, ds, am, ag in data:
        if ag > 0.05:
            ax.annotate(f"{model}\n({ds})", (am, ag), textcoords="offset points", xytext=(5, 3), fontsize=6)
    ax.legend(handles=[mpatches.Patch(color=pc, label="STaRK-PRIME"),
                       mpatches.Patch(color=mc, label="STaRK-MAG"),
                       plt.Line2D([], [], color="k", lw=0.8, ls="--", label="Ideal ($A_{gen}=A_{mem}$)")],
              fontsize=8, loc="upper left")
    ax.set_xlabel("Peak $A_{\\rm mem}$"); ax.set_ylabel("Peak $A_{\\rm gen}$")
    ax.set_title("Knowing-Using Gap: SFT Baseline Runs (7 confirmed, Relaxed EM)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.005, 0.38)
    ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); save(fig, "fig1_kug_scatter.pdf")

def learning_curves():
    cases=[("llama3.2-3b","stark_prime","Llama3.2-3B / STaRK-PRIME"),
           ("qwen3.5-2b","stark_prime","Qwen3.5-2B / STaRK-PRIME")]
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
    # V1 re-evaluated numbers (relaxed EM, from outputs/runs/)
    V1 = {"stark_prime": {
              "Qwen3.5-2B":  {"baseline": 0.244, "rep_distill": 0.233, "contrastive": 0.266, "probe": 0.271, "hybrid": 0.271},
              "LFM2.5-1.2B": {"baseline": 0.068, "rep_distill": 0.068, "contrastive": 0.060, "probe": 0.051, "hybrid": 0.069},
              "Gemma4-E4B":  {"baseline": None,  "rep_distill": None,  "contrastive": None,  "probe": None,  "hybrid": None},
              "Llama3.2-3B": {"baseline": 0.254, "rep_distill": 0.242, "contrastive": 0.239, "probe": 0.261, "hybrid": 0.226}},
           "stark_mag": {
              "Qwen3.5-2B":  {"baseline": 0.019, "rep_distill": 0.019, "contrastive": 0.019, "probe": 0.018, "hybrid": 0.018},
              "LFM2.5-1.2B": {"baseline": 0.013, "rep_distill": 0.013, "contrastive": 0.013, "probe": 0.014, "hybrid": 0.014},
              "Gemma4-E4B":  {"baseline": 0.009, "rep_distill": 0.000, "contrastive": 0.011, "probe": 0.009, "hybrid": 0.009},
              "Llama3.2-3B": {"baseline": 0.086, "rep_distill": 0.084, "contrastive": 0.093, "probe": 0.090, "hybrid": 0.104}}}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, ds in zip(axes, ["stark_prime", "stark_mag"]):
        models = list(V1[ds].keys()); x = np.arange(len(models)); w = 0.14
        offs = np.linspace(-(len(VARIANTS)-1)*w/2, (len(VARIANTS)-1)*w/2, len(VARIANTS))
        for i, v in enumerate(VARIANTS):
            vals = [V1[ds][m][v] if V1[ds][m][v] is not None else 0 for m in models]
            ax.bar(x+offs[i], vals, w, color=PAL[v], label=VL[v], edgecolor="white", linewidth=0.4)
        # Baseline dashed marker line per model
        for xi, m in enumerate(models):
            bl = V1[ds][m]["baseline"]
            if bl is not None:
                ax.hlines(bl, xi - 2.5*w, xi + 2.5*w, colors="#333333", lw=1.5, linestyle="--", zorder=6)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8, rotation=18, ha="right")
        ax.set_ylabel("Peak $A_{\\rm gen}$"); ax.set_title({"stark_prime": "STaRK-PRIME", "stark_mag": "STaRK-MAG"}[ds])
        ax.spines[["top", "right"]].set_visible(False)
        if ds == "stark_prime":
            ax.legend(fontsize=7.5, ncol=1, loc="upper right")
            ax.text(0.01, 0.97, "\u2012 \u2012 Baseline SFT", transform=ax.transAxes,
                    fontsize=7.5, va="top", color="#333333")
    fig.suptitle("AA-SFT Alignment Losses vs. SFT Baseline — Peak $A_{\\rm gen}$ (Relaxed EM)", fontsize=10)
    plt.tight_layout(); save(fig, "fig3_v1_gains.pdf")

def baseline_vs_aasft():
    """Fig 4: AA-SFT best variant vs SFT baseline, sorted by baseline."""
    # (model, dataset, baseline_agen, aasft_best_agen)
    rows = [("Llama3.2-3B", "PRIME", 0.254, 0.261),
            ("Qwen3.5-2B",  "PRIME", 0.244, 0.271),
            ("Llama3.2-3B", "MAG",   0.086, 0.104),
            ("LFM2.5-1.2B", "PRIME", 0.068, 0.069),
            ("Qwen3.5-2B",  "MAG",   0.019, 0.019),
            ("LFM2.5-1.2B", "MAG",   0.013, 0.014),
            ("Gemma4-E4B",  "MAG",   0.009, 0.011)]
    labels = [f"{m}\n({ds})" for m, ds, *_ in rows]
    bl_vals  = [r[2] for r in rows]
    aa_vals  = [r[3] for r in rows]
    x = np.arange(len(rows)); w = 0.32
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - w/2, bl_vals, w, color="#555555", label="SFT Baseline", edgecolor="white")
    bar_aa = ax.bar(x + w/2, aa_vals, w, label="AA-SFT (best loss)", edgecolor="white",
                    color=["#009E73" if a >= b else "#d62728" for a, b in zip(aa_vals, bl_vals)])
    # Annotate gain
    for xi, (b, a) in enumerate(zip(bl_vals, aa_vals)):
        pct = 100*(a-b)/b if b > 0 else 0
        sign = "+" if pct >= 0 else ""
        ax.text(xi + w/2, a + 0.003, f"{sign}{pct:.0f}%", ha="center", va="bottom", fontsize=7.5,
                color="#009E73" if pct >= 0 else "#d62728", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Peak $A_{\\rm gen}$ (Relaxed EM)")
    ax.set_title("AA-SFT vs. SFT Baseline — Peak $A_{\\rm gen}$ (6/7 pairs improved)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); save(fig, "fig4_baseline_vs_aasft.pdf")

def oracle_headroom():
    # (model, dataset, baseline_agen, aasft_best_agen, oracle_agen)
    data = [("Qwen3.5-2B",  "PRIME", 0.244, 0.271, 0.450),
            ("LFM2.5-1.2B", "PRIME", 0.068, 0.069, 0.380),
            ("Llama3.2-3B", "PRIME", 0.254, 0.261, 0.520),
            ("Qwen3.5-2B",  "MAG",   0.019, 0.019, 0.070),
            ("Llama3.2-3B", "MAG",   0.086, 0.104, 0.300)]
    labels = [f"{m}\n({ds})" for m, ds, *_ in data]; x = np.arange(len(labels))
    b    = [d[2] for d in data]
    gaa  = [d[3] - d[2] for d in data]
    gl   = [max(0, d[4] - d[3]) for d in data]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x, b,   label="Baseline SFT",    color="#555555", edgecolor="white")
    ax.bar(x, gaa, bottom=b, label="AA-SFT Gain", color="#009E73", edgecolor="white")
    ax.bar(x, gl,  bottom=[a+c for a, c in zip(b, gaa)],
           label="Remaining Headroom", color="#DDDDDD", edgecolor="#AAAAAA", hatch="//")
    for i, d in enumerate(data): ax.hlines(d[4], i-0.4, i+0.4, colors="#E69F00", lw=2.2, linestyle="--")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [plt.Line2D([], [], color="#E69F00", lw=2.2, ls="--", label="Oracle (Mem2Gen)")],
              fontsize=7.5, loc="upper left")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylabel("$A_{\\rm gen}$")
    ax.set_title("Oracle Headroom: Fraction of Gap Closed by AA-SFT (Relaxed EM)")
    ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); save(fig, "fig5_oracle_headroom.pdf")

if __name__=="__main__":
    print("Generating figures...")
    kug_scatter(); learning_curves(); v1_bars(); baseline_vs_aasft(); oracle_headroom()
    print("Done.")
