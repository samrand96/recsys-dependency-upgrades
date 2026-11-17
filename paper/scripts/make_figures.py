"""Read saved evidence; write publication figures, extra tables and provenance only in paper/.

Run: uv run --locked python paper/scripts/make_figures.py
No training, inference, bootstrap resampling or mutation of research records occurs.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_tables import ROOT, PRIMARY, HOSTS, _load, _fmt

OUT = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
INPUTS: dict[str, str] = {}


def track(path: Path) -> None:
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    track(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, lines: list[str]) -> None:
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    track(PRIMARY / "ledger.duckdb")
    cases = _load(PRIMARY)
    primary = {k: v for k, v in cases.items() if v["dataset"] == "ml1m"}
    reports = {cid: read(PRIMARY / "compare" / f"{cid}.json") for cid in cases}
    hosts = {}
    for job, label in HOSTS.items():
        root = PRIMARY / "hosts" / job / "results"
        track(root / "ledger.duckdb")
        for p in sorted((root / "compare").glob("*.json")):
            track(p)
        hosts.setdefault(label, {}).update(_load(root))
    outcome = Counter(v["outcome"] for v in primary.values())
    list_changed = sum(v["detail"].get("changed_frac", 0) > 0 for v in primary.values())
    assert len(primary) == 201 and outcome["changed"] == 14 and list_changed == 9
    assert outcome["regression"] == outcome["contract"] == outcome["uncertain"] == 0
    write("evidence_numbers.tex", [
        rf"\newcommand{{\nListChanged}}{{{list_changed}}}",
        rf"\newcommand{{\nCheckpointOnly}}{{{outcome['changed'] - list_changed}}}",
        rf"\newcommand{{\nCompleted}}{{{outcome['identical'] + outcome['changed']}}}",
    ])

    # Disjoint case outcomes, retaining the declared collection and split.
    labels = ["Single / development", "Single / held-out", "Snapshot / development", "Snapshot / held-out"]
    groups = [(c, s) for c in ("systematic_single", "systematic_snapshot") for s in ("dev", "heldout")]
    cats = ["identical", "checkpoint_only", "top10_change", "failure", "not_applicable"]
    names = ["Identical checkpoints", "Checkpoint change only", "Top-10 change", "Runtime failure", "Not applicable"]
    colors = ["#d6dce4", "#67a9cf", "#2166ac", "#b77a43", "#a78cb6"]
    matrix = []
    for coll, split in groups:
        c = Counter()
        for v in primary.values():
            if (v["collection"], v["split"]) != (coll, split):
                continue
            key = v["outcome"]
            if key == "changed":
                key = "top10_change" if v["detail"]["changed_frac"] > 0 else "checkpoint_only"
            c[key] += 1
        matrix.append([c[k] for k in cats])
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    left = np.zeros(4)
    for j, (name, color) in enumerate(zip(names, colors)):
        vals = np.array(matrix)[:, j]
        ax.barh(range(4), vals, left=left, color=color, label=name, height=.65)
        for i, val in enumerate(vals):
            if val:
                ax.text(left[i] + val/2, i, str(val), ha="center", va="center", fontsize=8,
                        color="white" if j == 2 else "#182430")
        left += vals
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Recorded case configurations (not independent mechanisms)")
    ax.set_xlim(0, 70)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.35), ncol=3, frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    save(fig, "outcome_flow")

    points = []
    for cid, case in cases.items():
        for mode, m in reports[cid].get("modes", {}).items():
            ab = m.get("AB", {})
            q = ((ab.get("quality") or {}).get("ndcg10") or {}).get("primary")
            behavior = ab.get("behavior") or {}
            if m.get("status") != "ok" or q is None or behavior.get("users_changed_frac") is None:
                continue
            points.append({"case": cid, "mode": mode, "dataset": case["dataset"],
                           "churn_pct": 100 * behavior["users_changed_frac"], **q})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True)
    sets = [("ml1m", "MovieLens 1M", "#2166ac", "o"),
            ("ml20m", "MovieLens 20M", "#b57030", "s"),
            ("ml1m_per_user_70_10_20", "1M split sensitivity", "#5b8a5a", "^")]
    for ax in axes:
        for dataset, label, color, marker in sets:
            ps = [p for p in points if p["dataset"] == dataset]
            ax.errorbar([p["churn_pct"] for p in ps], [p["estimate"] for p in ps],
                        yerr=[[p["estimate"]-p["low"] for p in ps], [p["high"]-p["estimate"] for p in ps]],
                        fmt=marker, ms=4, color=color, alpha=.65, elinewidth=.7, capsize=2, label=label)
        ax.axhline(0, color="#444444", lw=.7)
        ax.set_xlabel("Changed top-10 lists (%)")
        ax.set_xlim(-.3, 9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=.15)
    axes[0].axhspan(-.002, .002, color="#d9e9d8", alpha=.45)
    axes[0].axhline(.002, color="#5b8a5a", ls="--", lw=.8)
    axes[0].axhline(-.002, color="#5b8a5a", ls="--", lw=.8)
    axes[0].set_ylim(-.00225, .00225)
    axes[0].set_title("(a) Smallest equivalence bounds: +/-0.002")
    axes[0].set_ylabel("NDCG@10 loss (old - new)")
    axes[1].set_ylim(-.0003, .00038)
    axes[1].set_title("(b) Same observations, enlarged vertical scale")
    axes[1].annotate("ALS retraining", (8.48, -.00003354), (4.1, -.00019),
                     arrowprops={"arrowstyle": "-", "lw": .6}, fontsize=8)
    axes[1].annotate("20M tie selection", (.25, .00011), (2.3, .00023),
                     arrowprops={"arrowstyle": "-", "lw": .6}, fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.07), fontsize=8)
    fig.tight_layout(w_pad=2)
    save(fig, "churn_quality")

    # Group intervals for cases with nonzero quality effects; no pooling across cases.
    select = ["N2-P2-als", "N2-P2-als-ml20m"]
    select.extend(k for k, v in cases.items() if v["dataset"] == "ml1m_per_user_70_10_20")
    fig, axes = plt.subplots(1, len(select), figsize=(7.2, 2.7), sharex=True)
    group_records = []
    for ax, cid in zip(np.atleast_1d(axes), select):
        m = reports[cid]["modes"]["retrain"]["AB"]["quality"]["ndcg10"]
        gs = m["activity_groups"]
        for i, group in enumerate(("low", "medium", "high")):
            q = gs[group]["primary"]
            group_records.append({"case": cid, "group": group, **q})
            ax.errorbar(1e4*q["estimate"], i,
                        xerr=[[1e4*(q["estimate"]-q["low"])], [1e4*(q["high"]-q["estimate"])]],
                        fmt="o", ms=4, color="#2166ac", capsize=3)
        ax.axvline(0, color="#777777", lw=.7)
        ax.set_yticks(range(3), [f"{g.title()} (n={gs[g]['n_users']})" for g in ("low", "medium", "high")], fontsize=7)
        ax.invert_yaxis()
        ax.set_title({"N2-P2-als": "1M: ALS", "N2-P2-als-ml20m": "20M: ALS"}.get(cid, "1M sensitivity: item-kNN"), fontsize=9)
        ax.set_xlabel(r"NDCG@10 loss ($\times 10^{-4}$)", fontsize=8)
        ax.set_xlim(-5.5, 5.5)
        ax.grid(axis="x", alpha=.15)
    fig.tight_layout(w_pad=1.5)
    save(fig, "activity_intervals")

    # Synthetic controls: see make_extras.py, which reads results/synthetic_v2/summary.json.
    tab = []
    for cid in sorted(k for k, v in cases.items() if v["dataset"] != "ml1m"):
        for mode, m in sorted(reports[cid].get("modes", {}).items()):
            ab = m.get("AB", {})
            p = ((ab.get("quality") or {}).get("ndcg10") or {}).get("primary")
            if not p:
                continue
            changed = any(r.get("changed_stages") for r in ab.get("runs", []))
            tab.append(r"\texttt{" + cid.replace("_", r"\_") + "} & " + mode +
                       f" & {'yes' if changed else 'no'} & {100*ab['behavior']['users_changed_frac']:.2f} & " +
                       rf"${_fmt(p['estimate'])}\ [{_fmt(p['low'])}, {_fmt(p['high'])}]$" + r" \\")
    write("tab_extensions.tex", tab)
    ease = read(PRIMARY / "diagnosis" / "N5-P6-ease" / "summary.json")
    tab = []
    for phase in ("native-235", "native-246", "235-preload-246", "246-preload-235", "rollback-235", "rollback-246"):
        r = ease["runs"][phase]
        label = "native" if phase.startswith("native") else "rollback" if phase.startswith("rollback") else "preload"
        tab.append(f"{r['numpy']} & {label} & {r['donor_wheel_numpy']} & " +
                   r"\texttt{" + ease["s3_sha256"][phase][:16] + r"} \\")
    assert ease["checks"]["replicated_donor_s3_in_both_directions"]
    assert ease["checks"]["native_s3_restored_after_removing_preload"]
    write("tab_ease_intervention.tex", tab)
    dataset = read(ROOT / "data" / "manifest.json")
    tab = []
    for split in ("train", "validation", "test"):
        s = dataset["splits"][split]
        nums = [s[k] for k in ("interactions", "warm_interactions", "warm_users", "warm_relevant_users")]
        tab.append(split.title() + " & " + " & ".join(f"{n:,}".replace(",", "{,}") for n in nums) + r" \\")
    write("tab_data_counts.tex", tab)
    audit = read(ROOT / "paper" / "phase-audit-2026-11-15.json")
    payload = {"method": "Read-only extraction of saved reports; no experimental reruns.",
               "primary_case_count": len(primary), "outcomes": dict(outcome),
               "changed_top10_cases": list_changed, "outcome_chart": matrix,
               "quality_points": points, "group_points": group_records,
               "synthetic_status": "Reported from analysis version 2 (results/synthetic_v2) by make_extras.py.",
               "sources_sha256": dict(sorted(INPUTS.items()))}
    (OUT / "figure_evidence.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": 3, "primary_cases": len(primary), "top10_changes": list_changed,
                      "quality_points": len(points), "source_files_hashed": len(INPUTS)}))


if __name__ == "__main__":
    main()
