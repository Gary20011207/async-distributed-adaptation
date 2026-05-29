
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "r13946001" / "presentation" / "assets"


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, fs=13):
    patch = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.018,rounding_size=0.024",facecolor=fc,edgecolor="#334155",linewidth=1.8)
    ax.add_patch(patch)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,weight="bold",color="#0f172a",linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color="#334155", lw=2.0):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=4, shrinkB=4))


def base(title):
    fig, ax = plt.subplots(figsize=(13.8,7.2))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(0.5,0.94,title,ha="center",fontsize=22,weight="bold",color="#0f172a")
    return fig, ax


def system_architecture():
    fig, ax = base("Event-Driven Clockless FL Simulator")
    xs=[0.06,0.28,0.50,0.72]
    clients=["Hospital A\nfast", "Hospital B\nnormal", "Hospital C\nstraggler", "Hospital D\nnon-IID"]
    for x,t in zip(xs,clients):
        box(ax,x,0.72,0.16,0.13,t,"#dbeafe",fs=13)
        arrow(ax,x+0.08,0.72,x+0.08,0.61,color="#2563eb")
    ax.plot([0.14,0.80],[0.60,0.60],color="#2563eb",lw=2.0)
    ax.text(0.47,0.625,"arrival events",ha="center",fontsize=13,weight="bold",color="#1d4ed8")
    stages=[(0.06,"Event Queue\nsimulated time", "#ede9fe"),(0.30,"Logical Version\nstaleness", "#fef3c7"),(0.54,"Async Aggregator\nNaive / FedBuff / CAA", "#dcfce7"),(0.78,"Global Model\nlogs + plots", "#ecfeff")]
    for x,t,c in stages:
        box(ax,x,0.36,0.17,0.14,t,c,fs=13)
    for i in range(len(stages)-1):
        arrow(ax,stages[i][0]+0.17,0.43,stages[i+1][0],0.43,color="#0f766e")
    box(ax,0.20,0.15,0.60,0.10,"No synchronized physical clock: ordering uses logical model versions and event arrivals.","#f8fafc",fs=14)
    save(fig,"system_architecture.png")


def method_flow():
    fig, ax = base("CAA-v2 Aggregation Pipeline")
    labels=["Receive\nclient delta","Buffer\nB updates","Score\nstale + agree","Normalize\nclip + fair","Update\nserver model"]
    xs=[0.04,0.24,0.44,0.64,0.82]
    for i,(x,t) in enumerate(zip(xs,labels),1):
        box(ax,x,0.59,0.14,0.15,f"{i}. {t}","#dbeafe",fs=13)
        if i < len(xs):
            arrow(ax,x+0.14,0.665,xs[i],0.665,color="#0f766e")
    signals=[("Logical staleness\ntau = v_server - v_start",0.12,"#fef3c7"),("Agreement\ncos(delta, reference)",0.39,"#dcfce7"),("Fairness credit\n1/(1+count)^p",0.66,"#fee2e2")]
    for t,x,c in signals:
        box(ax,x,0.28,0.22,0.14,t,c,fs=13)
    ax.plot([0.10,0.88],[0.52,0.52],color="#94a3b8",lw=1.6,linestyle="--")
    ax.text(0.5,0.485,"server-side clockless signals used during scoring",ha="center",fontsize=13,weight="bold",color="#475569")
    ax.text(0.5,0.11,"All decisions use logical versions, deltas, accepted trajectory, and contribution counts.",ha="center",fontsize=15,weight="bold",color="#1d4ed8")
    save(fig,"proposal_method_flow.png")


def decision_flow():
    fig, ax = base("CAA-v2 Server Decision Flow")
    xs=[0.06,0.30,0.54,0.76]
    labels=["Build reference\nbuffer + EMA","Score updates\nage x agreement","Apply safety\nclip / fair","Server step\nadaptive alpha"]
    cols=["#dbeafe","#dcfce7","#fef3c7","#ecfeff"]
    for i,(x,t,c) in enumerate(zip(xs,labels,cols)):
        box(ax,x,0.58,0.18,0.15,t,c,fs=13)
        if i < len(xs)-1:
            arrow(ax,x+0.18,0.655,xs[i+1],0.655,color="#0f766e")
    safety=[("Drop stale-conflicting\nonly when necessary",0.12,"#fee2e2"),("Fallback to FedBuff\nif all updates dropped",0.40,"#ede9fe"),("No client-side change\nserver aggregation only",0.68,"#f8fafc")]
    for t,x,c in safety:
        box(ax,x,0.28,0.22,0.14,t,c,fs=13)
    ax.text(0.5,0.12,"The flow is deterministic and clockless; no physical timestamp is required.",ha="center",fontsize=15,weight="bold",color="#1d4ed8")
    save(fig,"proposal_method_decision.png")


def challenge_map():
    fig, ax = base("Distributed-Systems Challenges Made Visible")
    items=[(0.08,0.62,"No global clock\nno total physical order","#dbeafe"),(0.56,0.62,"Stale updates\nold model version","#fef3c7"),(0.08,0.34,"Conflicting updates\nnon-IID hospital data","#fee2e2"),(0.56,0.34,"Fast-client bias\narrival dominance","#ede9fe")]
    for x,y,t,c in items:
        box(ax,x,y,0.34,0.15,t,c,fs=14)
    box(ax,0.29,0.12,0.42,0.11,"Report accuracy + staleness + simulated time + client imbalance","#ecfdf5",fs=14)
    save(fig,"proposal_challenge_map.png")


def component_stack():
    fig, ax = base("What CAA-v2 Adds Over FedBuff")
    items=[("1 FedBuff", "buffer async updates", "#ede9fe"),("2 Staleness", "weight by logical age", "#dbeafe"),("3 Agreement", "prefer aligned deltas", "#dcfce7"),("4 Trajectory", "server EMA reference", "#fef3c7"),("5 Fairness", "reduce fast-client dominance", "#fee2e2"),("6 Adaptive alpha", "aggressive only when safe", "#ecfeff")]
    xs=[0.06,0.38,0.70]; ys=[0.60,0.34]
    for i,(title,desc,c) in enumerate(items):
        box(ax,xs[i%3],ys[i//3],0.24,0.16,f"{title}\n{desc}",c,fs=13)
    ax.text(0.5,0.13,"Simple server rule: combine known signals into one clockless async aggregation policy.",ha="center",fontsize=15,weight="bold",color="#1d4ed8")
    save(fig,"proposal_component_stack.png")


def novelty_boundary():
    fig, ax = base("Novelty Boundary")
    cols=[(0.05,"Existing",["FedAvg","FedAsync","FedBuff","staleness decay","cosine aggregation"],"#dbeafe"),(0.38,"Ours",["clockless simulator","fair-budget protocol","agreement weighting","server trajectory EMA","client fairness credit"],"#dcfce7"),(0.71,"Not claimed",["SOTA theorem","secure aggregation","hospital deployment","full proof"],"#fee2e2")]
    for x,title,items,c in cols:
        box(ax,x,0.23,0.25,0.52,title+"\n\n"+"\n".join(items),c,fs=13)
    ax.text(0.5,0.10,"Contribution: reproducible systems integration and evidence, not a wholly new FL primitive.",ha="center",fontsize=15,weight="bold",color="#1d4ed8")
    save(fig,"proposal_novelty_position.png")

if __name__ == "__main__":
    system_architecture(); component_stack(); method_flow(); decision_flow(); novelty_boundary(); challenge_map()
