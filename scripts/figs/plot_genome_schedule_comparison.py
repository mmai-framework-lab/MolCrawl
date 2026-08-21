"""Regenerate a genome GPT-2 figure from the SLURM logs.

Run from a checkout that can see workflows/slurm-logs/ and
learning_source_genome_runs/. The PNG is a build artefact and is not committed;
this script and the CSVs under learning_source_genome_runs/_results/ are what
the figure is reproduced from.

Colours are the reference categorical palette, slots 1-4, used unchanged.
"""

import glob, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"
S1,S2,S3,S4="#2a78d6","#eb6834","#1baf7a","#eda100"

def curve(pat):
    f=sorted(glob.glob(pat))[0]; t=open(f,errors="ignore").read()
    return [(int(s),float(v)) for s,_,v in re.findall(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)",t)]

runs=[
 ("base", "peak 6e-4, decay 54,634", S2, curve("workflows/slurm-logs/mc-gen-stab-base-25495.out")),
 ("D",    "peak 6e-4, decay 13,658", S4, curve("workflows/slurm-logs/mc-gen-stab-D-decay025-25499.out")),
 ("B",    "peak 1e-4, decay 54,634", S1, curve("workflows/slurm-logs/mc-gen-stab-B-lr1e4-25497.out")),
 ("(c)",  "peak 6e-4, decay 1,802",  S3, curve("workflows/slurm-logs/mc-gen-stab2-H-6e4to1e4-*.out")),
]

fig,(axT,axB)=plt.subplots(2,1,figsize=(10.8,8.4),facecolor=SURFACE,sharex=True,
                           gridspec_kw={"height_ratios":[1,1.35],"hspace":0.13})
for ax in (axT,axB):
    ax.set_facecolor(SURFACE); ax.grid(True,color="#e8e7e1",linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("left","bottom"): ax.spines[sp].set_color("#d5d4cc")
    ax.tick_params(colors=INK2,labelsize=9)

for tag,desc,c,ev in runs:
    xs=[p[0] for p in ev]; ys=[p[1] for p in ev]
    b=min(ev,key=lambda r:r[1])
    for ax in (axT,axB):
        ax.plot(xs,ys,color=c,lw=2.0,zorder=3,label=f"{tag:<5} {desc}" if ax is axT else None)
        ax.plot([b[0]],[b[1]],"o",ms=9,color=c,mec=SURFACE,mew=2,zorder=4)

axT.set_ylim(1.05,1.97); axT.set_xlim(0,60000)
axT.set_title("Full range — base swings up to 1.91",fontsize=11,color=INK,loc="left",pad=8)
axT.set_ylabel("val loss",fontsize=9.5,color=INK2)
axT.legend(frameon=False,fontsize=9,labelcolor=INK,loc="upper right",ncol=2,columnspacing=1.6)
axT.annotate("base — final 1.6853",(6000,1.79),fontsize=9.5,color=S2,fontweight="bold",
             bbox=dict(fc=SURFACE,ec="none",alpha=0.9,pad=2))

axB.set_ylim(1.072,1.225)
axB.set_title("Same data, zoomed to where the other three separate",fontsize=11,color=INK,loc="left",pad=8)
axB.set_xlabel("optimizer step",fontsize=9.5,color=INK2)
axB.set_ylabel("val loss",fontsize=9.5,color=INK2)
for tag,desc,c,ev in runs:
    if tag=="base": continue
    xs=[p[0] for p in ev]; ys=[p[1] for p in ev]
    dy={"D":0,"B":10,"(c)":-10}[tag]
    axB.annotate(f"  {tag}  final {ys[-1]:.4f}",(xs[-1],ys[-1]),textcoords="offset points",
                 xytext=(6,dy),fontsize=9,color=c,va="center",fontweight="bold")
axB.annotate("base leaves the frame here",(20500,1.212),fontsize=8.8,color=S2,
             bbox=dict(fc=SURFACE,ec="none",alpha=0.9,pad=2))
axB.text(0.31,0.60,"Time at high LR decides the outcome, not the peak:\n"
                    "(c) hits the same 6e-4 peak as base but leaves it by step 1,802.",
         transform=axB.transAxes,fontsize=8.8,color=INK2,va="top",ha="left",
         bbox=dict(fc=SURFACE,ec="none",alpha=0.9,pad=3))

fig.suptitle("Shortening the high-LR stretch beats lowering the peak",fontsize=13,color=INK,x=0.01,ha="left",y=0.985)
fig.text(0.01,0.945,"Same subset (mammal_centered), four schedules — dots mark each run's best",
         fontsize=10,color=INK2,ha="left")
fig.tight_layout(rect=[0,0,1,0.935])
fig.savefig("tmp/figs/schedule-comparison.png",dpi=170,facecolor=SURFACE)
print("saved")
