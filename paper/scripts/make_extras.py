"""Additional tables and figures for the paper, read-only from saved evidence.

Run from the project root, after make_tables.py and make_figures.py:

    uv run --locked python paper/scripts/make_extras.py

Reads the ledgers, comparison reports, manifests and results/synthetic_v2/summary.json.
Writes paper/generated/tab_x_*.tex, paper/generated/extra_numbers.tex and paper/figures/*.pdf.
No training, inference or resampling happens here.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.patches
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch

from make_tables import HOSTS, PRIMARY, ROOT, _fmt, _load
from tuning_evidence import audit

OUT = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
MACROS: dict[str, str] = {}

PIPELINES = ["P0", "P1-float32", "P1-float64", "P2-als", "P2-bm25", "P2-cosine",
             "P3-svd", "P4-sar", "P6-ease", "P6-knn", "P7-bpr"]
PIPELINE_INFO = {  # label, library, model
    "P0": ("pandas", "preprocessing"),
    "P1-float32": ("reference", "item-kNN, float32"),
    "P1-float64": ("reference", "item-kNN, float64"),
    "P2-als": ("implicit", "ALS"),
    "P2-bm25": ("implicit", "BM25 item-kNN"),
    "P2-cosine": ("implicit", "cosine item-kNN"),
    "P3-svd": ("Surprise", "SVD"),
    "P4-sar": ("Recommenders", "SAR"),
    "P6-ease": ("Cornac", "EASE"),
    "P6-knn": ("Cornac", "ItemKNN"),
    "P7-bpr": ("reference", "BPR (PyTorch)"),
}
# Declared transition-pipeline cells that could not be created (protocol/exclusions.md).
EXCLUDED = {("D2", "P4-sar"), ("D3", "P4-sar")} | {(t, p) for t in ("N1", "S1", "S2") for p in ("P6-ease", "P6-knn")}
COLORS = {"identical": "#d6dce4", "checkpoint": "#67a9cf", "list": "#2166ac", "failure": "#b77a43",
          "not_applicable": "#a78cb6", "excluded": "#f0f0f0"}
BLUE, ORANGE, GREEN, GREY = "#2166ac", "#b57030", "#5b8a5a", "#666666"


def write(name: str, lines: list[str]) -> None:
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def num(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def tt(text: str) -> str:
    return r"\texttt{" + text.replace("_", r"\_") + "}"


def natural(key: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", key)]


def split_case(case_id: str) -> tuple[str, str]:
    transition, rest = case_id.split("-", 1)
    return transition, rest


def cell_state(case: dict) -> str:
    if case["outcome"] == "changed":
        return "list" if case["detail"].get("changed_frac", 0) > 0 else "checkpoint"
    return case["outcome"]


def versions(freeze) -> dict:
    if isinstance(freeze, str):
        freeze = json.loads(freeze)
    if isinstance(freeze, list):
        return dict(line.split("==", 1) for line in freeze if "==" in line)
    return freeze


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    cases = _load(PRIMARY)
    primary = {k: v for k, v in cases.items() if v["dataset"] == "ml1m"}
    reports = {cid: json.loads((PRIMARY / "compare" / f"{cid}.json").read_text(encoding="utf-8")) for cid in cases}
    con = duckdb.connect(str(PRIMARY / "ledger.duckdb"), read_only=True)
    transitions = con.execute(
        "SELECT transition_id, family, collection, split, direct_change_json, held_fixed_json FROM transitions").fetchall()
    env_rows = con.execute("SELECT env_id, status, freeze_json, runtime_json FROM envs").fetchall()
    suites = con.execute("SELECT package, status, count(*) FROM testsuite GROUP BY ALL").fetchall()
    cost = con.execute("""SELECT mode, count(*), median(wall_s), max(wall_s), median(cpu_s), max(peak_rss_mb)
                          FROM runs WHERE status = 'ok' GROUP BY mode ORDER BY mode""").fetchall()
    total_hours = con.execute("SELECT sum(wall_s) / 3600 FROM runs").fetchone()[0]
    configs = {cid: json.loads(cfg) for cid, cfg in con.execute("SELECT case_id, config_json FROM cases").fetchall()}
    tuning = audit(ROOT, con, primary, reports, configs)
    con.close()
    (OUT / "tuning_evidence.json").write_text(json.dumps(tuning, indent=2) + "\n", encoding="utf-8")
    # Stability of the validation selection across baseline environments.
    by_value, gaps = defaultdict(list), []
    for cid, case in tuning["cases"].items():
        means = []
        for candidate in case["validation"]:
            scores = [trial["mean_ndcg10"] for trial in candidate["trials"]]
            by_value[(split_case(cid)[1], str(candidate["value"]))].extend(scores)
            if scores:
                means.append(sum(scores) / len(scores))
        if len(means) > 1:
            ordered = sorted(means, reverse=True)
            gaps.append(ordered[0] - ordered[1])
    MACROS["tuningSpread"] = f"${_fmt(max(max(v) - min(v) for v in by_value.values() if v))}$"
    MACROS["tuningGap"] = f"${_fmt(min(gaps))}$"
    selected = tuning["cases"]
    inconsistent = [cid for cid, row in selected.items()
                    if row["status"] not in ("verified", "no_successful_comparison")]
    if inconsistent:
        raise ValueError(f"Validation and test settings need review: {', '.join(inconsistent)}")

    by_cell = {split_case(cid): v for cid, v in primary.items()}

    # ------------------------------------------------------------- single-dependency transitions
    rows = []
    order = {"numpy": 0, "scipy": 1, "pandas": 2, "torch": 3}
    single = [t for t in transitions if t[2] == "systematic_single"]
    for tid, family, _, split, direct, held in sorted(single, key=lambda t: (order[t[1]], natural(t[0]))):
        direct, held = json.loads(direct), json.loads(held)
        mine = [v for (t, _), v in by_cell.items() if t == tid]
        c = Counter(cell_state(v) for v in mine)
        short = {"numpy": "np", "scipy": "sp", "pandas": "pd"}
        fixed = ", ".join(f"{short[k]} {v}" for k, v in sorted(held.items())) or "np 2.2.6, sp 1.16.3"
        name = {"numpy": "NumPy", "scipy": "SciPy", "pandas": "pandas", "torch": "PyTorch"}[family]
        rows.append(f"{tid} & {name} & {direct['old']} $\\rightarrow$ {direct['new']} & {fixed} & "
                    f"{'dev' if split == 'dev' else 'held-out'} & {len(mine)} & {c['identical']} & "
                    f"{c['checkpoint'] + c['list']} & {c['failure']} & {c['not_applicable']} \\\\")
    write("tab_x_transitions.tex", rows)

    # ------------------------------------------------------------- snapshots
    snaps = {}
    for env_id, status, freeze, _ in env_rows:
        if env_id.startswith("py312-snapshot-") and status == "ok":
            snaps[env_id[-10:]] = versions(freeze)
    into = {}
    for tid, _, coll, split, direct, _ in transitions:
        if coll == "systematic_snapshot":
            d = json.loads(direct)
            into[d["new_as_of"]] = (tid, split, d["old_as_of"])
    rows = []
    for date in sorted(snaps):
        v = snaps[date]
        tid, split, _ = into.get(date, ("--", "", ""))
        mine = [c for (t, _), c in by_cell.items() if t == tid]
        c = Counter(cell_state(x) for x in mine)
        tail = (f"{tid} & {len(mine)} & {c['checkpoint'] + c['list']} & {c['failure']}" if mine else "-- & -- & -- & --")
        rows.append(f"{date} & {v.get('numpy')} & {v.get('scipy')} & {v.get('pandas')} & {v.get('implicit')} & {tail} \\\\")
    write("tab_x_snapshots.tex", rows)

    # ------------------------------------------------------------- BLAS kernels by NumPy version and host
    def blas_table(ledger: Path) -> dict:
        c = duckdb.connect(str(ledger), read_only=True)
        try:
            table = {}
            for env_id, runtime in c.execute("SELECT env_id, runtime_json FROM envs WHERE status = 'ok'").fetchall():
                match = re.search(r"-np([\d.]+)-", env_id)
                runtime = json.loads(runtime)
                if not match or match.group(1) in table:
                    continue
                pools = [p for p in runtime.get("threadpools", [])
                         if p.get("internal_api") == "openblas" and "/numpy.libs/" in p.get("filepath", "")]
                if pools:
                    table[match.group(1)] = (pools[0]["version"], pools[0]["architecture"])
            return table
        finally:
            c.close()

    blas = {"A": blas_table(PRIMARY / "ledger.duckdb")}
    for job, label in HOSTS.items():
        if label in ("B", "C"):
            blas.setdefault(label, {}).update(blas_table(PRIMARY / "hosts" / job / "results" / "ledger.duckdb"))
    np_versions = sorted(blas["A"], key=natural)
    rows = []
    for v in np_versions:
        version = blas["A"][v][0]
        short = ".".join(version.split(".")[:3])
        rows.append(f"{v} & {short} & " + " & ".join(blas[h].get(v, ("", "--"))[1] for h in ("A", "B", "C")) + r" \\")
        for h in ("B", "C"):
            assert v not in blas[h] or blas[h][v][0] == version, "hosts must bundle the same OpenBLAS"
    write("tab_x_blas.tex", rows)

    # ------------------------------------------------------------- pipelines and selected settings
    rows = []
    for p in PIPELINES:
        mine = sorted((cid for cid in primary if split_case(cid)[1] == p), key=natural)
        libs = set()
        for cid in mine:
            m = re.search(r"-(implicit|surprise|recommenders|cornac|torch)([\d.]+)$", configs[cid]["old"])
            if m:
                libs.add(m.group(2))
        chosen = Counter()
        for cid in mine:
            value = selected.get(cid, {}).get("selected")
            if value is not None:
                chosen[f"{value:g}" if isinstance(value, float) else str(value)] += 1
        cfg = configs[mine[0]]
        grid = next((f"{k.replace('_grid', '')} & {', '.join(f'{x:g}' if isinstance(x, float) else str(x) for x in cfg[k])}"
                     for k in ("neighbors_grid", "factors_grid", "lamb_grid", "similarity_grid") if cfg.get(k)), "-- & --")
        library, model = PIPELINE_INFO[p]
        model = model.split(",")[0]
        lib = library + (" " + "/".join(sorted(libs, key=natural)) if libs and library != "reference" else "")
        picks = ", ".join(f"{k} ({n})" for k, n in sorted(chosen.items(), key=lambda kv: -kv[1])) or "--"
        rows.append(f"{p} & {lib} & {model} & {cfg.get('dtype', '--') if p != 'P0' else '--'} & {grid} & {picks} & {len(mine)} \\\\")
    write("tab_x_pipelines.tex", [r.replace("lamb", "$\\lambda$").replace("_", r"\_") for r in rows])

    rows = []
    for p in PIPELINES[1:]:
        for mode in ("inference", "retrain"):
            scores = [row["quality"][mode]["baseline_ndcg10"] for cid, row in selected.items()
                      if split_case(cid)[1] == p and mode in row["quality"]]
            if scores:
                low, high = min(scores), max(scores)
                def span(a, b, places):
                    first, last = f"{a:.{places}f}", f"{b:.{places}f}"
                    return first if first == last else f"{first}--{last}"

                relative = [span(100*d/high, 100*d/low, 2) for d in (.002, .005, .010)]
                rows.append(f"{tt(p)} & {mode} & {len(scores)} & {span(low, high, 5)} & "
                            + " & ".join(relative) + r" \\")
    if sum(len(row["quality"]) for row in selected.values()) != sum(
        int(row.split(" & ")[2]) for row in rows
    ):
        raise ValueError("Baseline table omitted one or more completed case-mode comparisons")
    write("tab_x_baseline.tex", rows)

    # ------------------------------------------------------------- pipeline outcomes
    rows = []
    for p in PIPELINES:
        c = Counter(cell_state(v) for (t, q), v in by_cell.items() if q == p)
        excluded = sum(q == p for _, q in EXCLUDED)
        rows.append(f"{p} & {PIPELINE_INFO[p][0]} {PIPELINE_INFO[p][1]} & {sum(c.values())} & {c['identical']} & "
                    f"{c['checkpoint']} & {c['list']} & {c['failure']} & {c['not_applicable']} & {excluded or '--'} \\\\")
    write("tab_x_pipeline_outcomes.tex", rows)

    # ------------------------------------------------------------- upstream suites and cost
    suite = defaultdict(Counter)
    for package, status, n in suites:
        suite[package][status] = n
    rows = [f"{p} & " + " & ".join(str(suite[p][s]) for s in ("passed", "failed", "unavailable", "not_runnable")) + r" \\"
            for p in ("numpy", "scipy", "pandas", "implicit", "scikit-surprise", "recommenders", "cornac", "torch")]
    write("tab_x_suites.tex", rows)
    rows = [f"{mode} & {num(n)} & {med:.1f} & {mx:.0f} & {cpu:.1f} & {rss:.0f} \\\\" for mode, n, med, mx, cpu, rss in cost]
    write("tab_x_cost.tex", rows)
    MACROS["computeHours"] = f"{total_hours:.1f}"

    # ------------------------------------------------------------- numerical magnitude of every changed comparison
    points, rows, tie_rows = [], [], []
    mech_of = {"P1": "M1", "P2-als": "M2", "P6-ease": "M3", "P2-bm25": "M4"}
    for cid in sorted((k for k, v in primary.items() if v["outcome"] == "changed"), key=natural):
        pipeline = split_case(cid)[1]
        mech = mech_of.get(pipeline, mech_of.get(pipeline.split("-")[0]))
        for mode, m in sorted(reports[cid]["modes"].items()):
            ab = m["AB"]
            if not any(r.get("changed_stages") for r in ab["runs"]):
                continue
            beh = ab["behavior"]
            shifted = sum(r.get("shared_items_changed_position10", 0) for r in ab["runs"])
            shared = sum(r.get("shared_items10", 0) for r in ab["runs"])
            points.append({"case": cid, "mode": mode, "mech": mech, "rel": beh["score_rel_l2"], "maxabs": beh["score_maxabs"],
                           "churn": 100 * beh["users_changed_frac"]})
            rows.append(f"{tt(cid)} & {'retrain' if mode == 'retrain' else 'infer'} & {mech} & ${_fmt(beh['score_maxabs'])}$ & "
                        f"${_fmt(beh['score_rel_l2'])}$ & {100 * beh['users_changed_frac']:.2f} & {beh['overlap10_mean']:.6f} & "
                        f"{100 * shifted / shared if shared else 0:.2f} & {'yes' if beh['canonical_equal'] else 'no'} \\\\")
            if mech == "M1" and mode == "inference":
                r0 = ab["runs"][0]
                ties = r0["crossing_ties"]["new"]["k10"]
                sizes = sorted(int(s) for s in ties["tie_sizes"])
                tie_rows.append(f"{tt(cid)} & {r0['n_users']} & {ties['users']} & {num(min(sizes))}--{num(max(sizes))} & "
                                f"{round(r0['users_changed_frac'] * r0['n_users'])} & {'yes' if r0['canonical_equal'] else 'no'} \\\\")
    write("tab_x_numeric.tex", rows)
    write("tab_x_ties.tex", tie_rows)

    # ------------------------------------------------------------- secondary metrics and interval precision
    widths, degenerate, n_modes, worst = [], 0, 0, {"ndcg10": 0.0, "recall10": 0.0, "ndcg20": 0.0}
    for cid, case in primary.items():
        if case["outcome"] not in ("identical", "changed"):
            continue
        for mode, m in reports[cid]["modes"].items():
            quality = (m.get("AB") or {}).get("quality") or {}
            if "ndcg10" not in quality:
                continue
            n_modes += 1
            p = quality["ndcg10"]["primary"]
            half = (p["high"] - p["low"]) / 2
            widths.append(half)
            degenerate += half == 0
            for metric in worst:
                q = quality[metric]["primary"]
                worst[metric] = max(worst[metric], abs(q["low"]), abs(q["high"]))
    MACROS.update(nCaseModes=str(n_modes), nDegenerate=str(degenerate), nNonDegenerate=str(n_modes - degenerate),
                  maxHalfWidth=f"${_fmt(max(widths))}$",
                  worstNdcgTen=f"${_fmt(worst['ndcg10'])}$", worstRecallTen=f"${_fmt(worst['recall10'])}$",
                  worstNdcgTwenty=f"${_fmt(worst['ndcg20'])}$")
    rows = []
    als = [reports[cid]["modes"]["retrain"]["AB"]["quality"] for cid in ("N2-P2-als", "Q3-P2-als", "B1-P2-als")]
    for metric, label in (("ndcg10", "NDCG@10 (primary)"), ("recall10", "Recall@10"), ("ndcg20", "NDCG@20")):
        p = als[0][metric]["primary"]
        assert all(q[metric]["primary"] == p for q in als), "the three ALS cases must reproduce the same intervals"
        s = als[0][metric]["sensitivity"]
        rows.append(f"{label} & ${_fmt(p['estimate'])}$ & $[{_fmt(p['low'])}, {_fmt(p['high'])}]$ & "
                    f"$[{_fmt(s['low'])}, {_fmt(s['high'])}]$ \\\\")
    write("tab_x_secondary.tex", rows)

    # ------------------------------------------------------------- issue-guided leads
    leads = yaml.safe_load((ROOT / "protocol" / "cases" / "issue_guided.yaml").read_text(encoding="utf-8"))
    leads = leads["leads"] if isinstance(leads, dict) else leads
    table = defaultdict(Counter)
    for lead in leads:
        table[lead["project"].split("/")[-1]][lead["status"]] += 1
    statuses = ("reproduced", "visible_failure", "not_reproducible", "not_tested")
    rows = [f"{project} & " + " & ".join(str(c[s]) for s in statuses) + f" & {sum(c.values())} \\\\"
            for project, c in sorted(table.items(), key=lambda kv: -sum(kv[1].values()))]
    rows.append(r"\midrule")
    rows.append("Total & " + " & ".join(str(sum(c[s] for c in table.values())) for s in statuses) + f" & {len(leads)} \\\\")
    write("tab_x_leads.tex", rows)

    # ------------------------------------------------------------- figure: the complete case matrix
    groups = [("NumPy", [t for t in "N1 N2 N3 N4 N5 N6".split()]), ("SciPy", "S1 S2 S3 S4 S5 S6 S7".split()),
              ("pandas", "D0 D1 D2 D3".split()), ("PyTorch", "T1 T2 T3 T4 T5 T6".split()),
              ("Snapshot", ["B1"] + [f"Q{i}" for i in range(1, 12)])]
    columns = [t for _, ts in groups for t in ts]
    split_of = {t[0]: t[3] for t in transitions}
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for j, t in enumerate(columns):
        for i, p in enumerate(PIPELINES):
            case = by_cell.get((t, p))
            state = cell_state(case) if case else ("excluded" if (t, p) in EXCLUDED else None)
            if state is None:
                continue
            ax.add_patch(plt.Rectangle((j + .06, i + .06), .88, .88, facecolor=COLORS[state],
                                       edgecolor="#888888" if state == "excluded" else "none",
                                       hatch="////" if state == "excluded" else None, lw=.4))
    edge = 0
    for name, ts in groups:
        ax.text(edge + len(ts) / 2, -1.25, name, ha="center", va="center", fontsize=10, fontweight="bold")
        edge += len(ts)
        if edge < len(columns):
            ax.axvline(edge, color="#333333", lw=.7)
    ax.set_xlim(0, len(columns))
    ax.set_ylim(len(PIPELINES), 0)
    ax.set_xticks(np.arange(len(columns)) + .5, columns, fontsize=7.2)
    for label, t in zip(ax.get_xticklabels(), columns):
        label.set_color("#111111" if split_of[t] == "heldout" else "#9a5b00")
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(PIPELINES)) + .5, [f"{p.split('-')[0]}  {PIPELINE_INFO[p][1]}" for p in PIPELINES], fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Patch(facecolor=COLORS[k], label=l) for k, l in (
        ("identical", "Identical checkpoints"), ("checkpoint", "Checkpoint change only"), ("list", "Top-10 lists change"),
        ("failure", "Runtime failure"), ("not_applicable", "Not applicable"))]
    handles.append(Patch(facecolor=COLORS["excluded"], edgecolor="#888888", hatch="////", label="Cannot be installed"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.42, -.03), ncol=3, frameon=False, fontsize=9)
    save(fig, "case_matrix")

    # ------------------------------------------------------------- figure: perturbation size versus list churn
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    floor_x = 1e-18
    style = {"M1": ("o", BLUE, "M1 exact ties (item-kNN)"), "M2": ("s", ORANGE, "M2 BLAS kernel (ALS)"),
             "M3": ("^", GREEN, "M3 BLAS binary (EASE)"), "M4": ("D", "#8a5a9e", "M4 log1p (BM25)")}
    clusters = defaultdict(list)
    for pt in points:
        clusters[(pt["mech"], pt["mode"], f"{pt['maxabs']:.2g}", f"{pt['churn']:.2f}")].append(pt)
    seen = set()
    for (mech, mode, _, _), members in sorted(clusters.items()):
        marker, color, label = style[mech]
        x = members[0]["maxabs"] if members[0]["maxabs"] > 0 else floor_x
        y = members[0]["churn"]
        ax.scatter(x, y, marker=marker, s=46, color=color, edgecolor="white", lw=.6,
                   label=label if mech not in seen else None, zorder=3)
        seen.add(mech)
        lift = -12 if (y >= 5 or (mech == "M1" and mode == "retrain" and y < .35)) else 5
        ax.annotate(f"{len(members)}x {'retrain' if mode == 'retrain' else 'inference'}", (x, y), textcoords="offset points",
                    xytext=(-34 if mech == "M1" and mode == "retrain" else 7, lift), fontsize=6.6, color=color)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=.1, linscale=.6)
    ax.set_ylim(-.04, 30)
    ax.set_yticks([0, .1, .3, 1, 3, 10], ["0", "0.1", "0.3", "1", "3", "10"])
    ax.set_xlim(floor_x / 5, 30)
    ticks = [floor_x, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1]
    ax.set_xticks(ticks, ["0\n(identical)", r"$10^{-15}$", r"$10^{-12}$", r"$10^{-9}$", r"$10^{-6}$", r"$10^{-3}$", "1"])
    ax.axvline(floor_x * 40, color="#999999", lw=.6, ls=":")
    ax.set_xlabel("Largest absolute difference of an eligible score between sides")
    ax.set_ylabel("Changed top-10 lists (%)")
    ax.grid(alpha=.15)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(.2, 1.0))
    save(fig, "perturbation_churn")

    # ------------------------------------------------------------- figure: what an unpinned project installed
    dates = sorted(d for d in snaps if d != "2024-06-01" and d != "2026-11-01")
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    packages = ["numpy", "scipy", "pandas", "implicit"]
    for i, package in enumerate(packages):
        previous = None
        for j, date in enumerate(dates):
            version = snaps[date].get(package)
            minor = ".".join(version.split(".")[:2])
            changed = previous is not None and minor != previous
            ax.text(j, i, version, ha="center", va="center", fontsize=6.6,
                    color="#111111", fontweight="bold" if changed else "normal",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "#dbe9f6" if changed else "#f3f3f3", "edgecolor": "none"})
            previous = minor
    quarter = {into[d][0]: j for j, d in enumerate(dates) if d in into and into[d][0].startswith("Q")}
    notes = {}
    for (t, p), case in by_cell.items():
        if t in quarter and case["outcome"] in ("changed", "failure"):
            notes.setdefault(t, Counter())[cell_state(case)] += 1
    for t, j in quarter.items():
        c = notes.get(t, Counter())
        x = j - .5
        text, color = "identical", "#9a9a9a"
        if c["failure"]:
            text, color = f"{c['failure']} crash", COLORS["failure"]
        elif c["list"] or c["checkpoint"]:
            text, color = f"{c['list'] + c['checkpoint']} changed", BLUE
        ax.annotate("", (j - .1, 4.0), (j - .9, 4.0), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
        ax.text(x, 4.38, t, ha="center", va="center", fontsize=7, color=color, fontweight="bold")
        ax.text(x, 4.72, text, ha="center", va="center", fontsize=5.6, color=color)
    ax.set_xlim(-.6, len(dates) - .4)
    ax.set_ylim(5.0, -.6)
    ax.set_xticks(range(len(dates)), [d[:7] for d in dates], fontsize=7)
    ax.set_yticks(range(4), ["NumPy", "SciPy", "pandas", "implicit"], fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "snapshot_timeline")

    # ------------------------------------------------------------- figure: BLAS kernel by NumPy version and host
    kernels = ["Zen", "Haswell", "Cooperlake", "SkylakeX"]
    kcolor = {"Zen": "#f1d9b5", "Haswell": "#e0b073", "Cooperlake": "#bcd4ea", "SkylakeX": "#6c9dc9"}
    hosts_cmp = {label: {} for label in ("B", "C")}
    for job, label in HOSTS.items():
        if label in hosts_cmp:
            hosts_cmp[label].update(_load(PRIMARY / "hosts" / job / "results"))
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    host_names = {"A": "A  Zen 4, AVX-512+BF16", "B": "B  Zen 3, AVX2", "C": "C  Cascade Lake, AVX-512"}
    for i, h in enumerate(("A", "B", "C")):
        for j, v in enumerate(np_versions):
            kernel = blas[h].get(v, (None, None))[1]
            if kernel:
                ax.add_patch(plt.Rectangle((j + .05, i + .12), .9, .76, facecolor=kcolor[kernel], edgecolor="none"))
                ax.text(j + .5, i + .5, kernel, ha="center", va="center", fontsize=7)
    marks = [("N1-P1-float32", 0, "M1"), ("N1-P2-bm25", 0, "M4"), ("N2-P2-als", 1, "M2"),
             ("N5-P6-ease", 4, "M3"), ("N6-P6-ease", 5, "M3")]
    offsets = defaultdict(int)
    for cid, j, mech in marks:
        for i, (h, source) in enumerate((("A", primary), ("B", hosts_cmp["B"]), ("C", hosts_cmp["C"]))):
            case = source.get(cid)
            if case and case["outcome"] == "changed":
                k = offsets[(i, j)]
                offsets[(i, j)] += 1
                ax.text(j + 1, i + .2 + .32 * k, mech, ha="center", va="center", fontsize=6.3, color="white", fontweight="bold",
                        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#b2182b", "edgecolor": "none"})
    ax.set_xlim(0, len(np_versions))
    ax.set_ylim(3, 0)
    ax.set_xticks(np.arange(len(np_versions)) + .5,
                  [f"NumPy {v}\nOpenBLAS {'.'.join(blas['A'][v][0].split('.')[:3])}" for v in np_versions], fontsize=6.2)
    ax.set_yticks(np.arange(3) + .5, [host_names[h] for h in ("A", "B", "C")], fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "blas_map")

    # ------------------------------------------------------------- figure: study design
    fig, ax = plt.subplots(figsize=(7.4, 3.3))

    def box(x, y, w, h, text, face, size=7.2, weight="normal"):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                                       facecolor=face, edgecolor="#555555", lw=.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "->", "color": "#444444", "lw": .9})

    ax.text(3.05, 5.72, "Environments under test (runner container, one thread)", ha="center", fontsize=7.6, fontweight="bold")
    ax.text(9.2, 5.72, "Pinned orchestrator", ha="center", fontsize=7.6, fontweight="bold")
    for y, name, face in ((4.25, "A: old lock", "#e8eef5"), (2.55, "B: new lock", "#dbe9f6")):
        box(.1, y, 1.25, 1.1, name, face, weight="bold")
        for i, stage in enumerate(("S2\ndata", "S3\nmodel", "S5-S6\nscores", "S7\ntop-K")):
            box(1.6 + i * 1.12, y + .12, 1.0, .86, stage, "white", size=6.4)
            if i:
                arrow(1.6 + i * 1.12 - .12, y + .55, 1.6 + i * 1.12, y + .55)
        arrow(1.35, y + .55, 1.6, y + .55)
        arrow(6.08, y + .55, 7.25, 3.95 if y > 3 else 3.75)
    box(.1, .75, 5.95, 1.1, "Frozen, checksummed inputs shared by both sides:\nprepared data (S0-S1), tuned settings, seeds, initial factors,\n"
        "minibatch orders, negatives; in inference the old side's model", "#f3f0e6", size=6.6)
    arrow(3.05, 1.85, 3.05, 2.55)
    box(7.25, 3.2, 1.75, 1.3, "Stage hashes\nS0-S7:\nfirst divergence", "white", size=6.4)
    box(9.25, 3.2, 1.9, 1.3, "S7c canonical rank\nS8 evaluator:\nNDCG@10, contracts", "white", size=6.2)
    arrow(9.0, 3.85, 9.25, 3.85)
    box(7.25, 1.55, 3.9, 1.15, "Per case: A/A control, A/B inference,\nA/B retraining (5 paired seeds);\n"
        "paired interval, verdict per tolerance", "#e7f1e6", size=6.4)
    arrow(10.2, 3.2, 10.2, 2.7)
    arrow(8.1, 3.2, 8.1, 2.7)
    box(7.25, .1, 3.9, .95, "Diagnosis: replay first diverging stage;\n"
        "rollback, forced kernel, swapped binary", "#f6e6e2", size=6.4)
    arrow(9.2, 1.55, 9.2, 1.05)
    ax.plot([6.65, 6.65], [.1, 5.5], color="#999999", lw=.8, ls="--")
    ax.set_xlim(0, 11.3)
    ax.set_ylim(0, 6)
    ax.axis("off")
    save(fig, "study_design")

    # ------------------------------------------------------------- synthetic controls, analysis version 2
    path = PRIMARY / "synthetic_v2" / "summary.json"
    if path.is_file():
        synthetic(json.loads(path.read_text(encoding="utf-8")))

    path = PRIMARY / "hosts" / "rerun_verification.json"
    if path.is_file():
        reruns(json.loads(path.read_text(encoding="utf-8")))

    write("extra_numbers.tex", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())])
    print(json.dumps(MACROS, indent=1))


def reruns(report: dict) -> None:
    """Cloud reruns with raw arrays retained (protocol/cloud_reruns.md)."""
    labels = {"second-cpu-2": "B (EPYC 7R13)", "intel-cpu-2": "C (Xeon 8259CL)", "gpu-t4-2": "GPU (Tesla T4)"}
    rows, total = [], Counter()
    for job in ("second-cpu-2", "intel-cpu-2", "gpu-t4-2"):
        r = report[job]
        ok = r["runs"].get("ok", 0)
        same_outcomes = r["cases"] - len(r["outcome_differences"])
        rows.append(f"{labels[job]} & {r['cases']} & {ok} & {num(r['stage_hashes_equal'])}/{num(r['stage_hashes_compared'])} & "
                    f"{same_outcomes}/{r['cases']} & {r['archive_bytes'] / 1e9:.1f} & "
                    f"{num(r['files_checked'] - r['files_failing_checksum'])}/{num(r['files_checked'])} & "
                    f"{r['runs_reevaluated'] - r['runs_with_metric_mismatch']}/{r['runs_reevaluated']} \\\\")
        # The paper's wording assumes every declared check passed; fail loudly otherwise.
        assert r["same_cpu_model"], f"{job} ran on a different CPU model than its first Job"
        assert r["stage_hashes_equal"] == r["stage_hashes_compared"] and not r["runs_without_counterpart"], job
        assert not r["outcome_differences"] and not r["archives_failing_checksum"] and not r["files_failing_checksum"], job
        assert r["runs_reevaluated"] == ok and not r["runs_with_metric_mismatch"], job
        for key in ("stage_hashes_equal", "stage_hashes_compared", "files_checked", "files_failing_checksum",
                    "runs_reevaluated", "runs_with_metric_mismatch", "archive_bytes", "cases", "runs_with_counterpart"):
            total[key] += r[key]
        total["runs"] += ok
        total["bad_archives"] += len(r["archives_failing_checksum"])
        total["outcome_differences"] += len(r["outcome_differences"])
    write("tab_x_reruns.tex", rows)
    MACROS.update(
        rerunRuns=num(total["runs"]), rerunCases=str(total["cases"]),
        rerunHashes=num(total["stage_hashes_compared"]), rerunHashesEqual=num(total["stage_hashes_equal"]),
        rerunFiles=num(total["files_checked"]), rerunFilesBad=str(total["files_failing_checksum"] + total["bad_archives"]),
        rerunEvaluated=num(total["runs_reevaluated"]), rerunEvaluatedBad=str(total["runs_with_metric_mismatch"]),
        rerunGigabytes=f"{total['archive_bytes'] / 1e9:.0f}", rerunOutcomeDifferences=str(total["outcome_differences"]),
        rerunGpuDeviceRuns=str(report["gpu-t4-2"]["runs_with_device_record"]),
    )


DETECTORS = [
    ("uniform:100", "Panel, uniform, 100"), ("activity:100", "Panel, activity, 100"), ("score_margin:100", "Panel, score margin, 100"),
    ("uniform:300", "Panel, uniform, 300"), ("activity:300", "Panel, activity, 300 (primary)"),
    ("score_margin:300", "Panel, score margin, 300"), ("uniform:1000", "Panel, uniform, 1{,}000"),
    ("activity:1000", "Panel, activity, 1{,}000"), ("score_margin:1000", "Panel, score margin, 1{,}000"),
    ("exact_snapshot", "Exact snapshot"), ("score_tolerance", "Score tolerance"), ("overlap_strict", "Overlap, strict"),
    ("overlap_lenient", "Overlap, lenient"), ("full_check", "Full labelled check"),
]


def synthetic(syn: dict) -> None:
    primary_delta = syn["primary"]["delta"]
    at = {d: {r["detector"]: r for r in syn["rows"] if r["delta"] == d} for d in (0.002, 0.005, 0.01)}
    rows = at[primary_delta]
    faults = ("F1", "F2", "F3", "F4")
    names = {"F0": "F0 no fault, full rebuild (control)", "F1": "F1 item-ID swap, 1\\% of items", "F2": "F2 seen mask dropped",
             "F3": "F3 scores rounded to 2 decimals", "F4": "F4 reversed ranking, 1\\% of users"}

    lines = []
    for f in ("F0",) + faults:
        c = syn["reference_labels"][f][repr(primary_delta)]
        fp = syn["footprint"][f]
        loss = "--" if fp["median_full_loss"] is None else f"${_fmt(fp['median_full_loss'])}$"
        lines.append(f"{names[f]} & {c['contract_failure']} & {c['harmful']} & {c['inconclusive']} & {c['equivalent']} & "
                     f"{fp['median_users_changed']:.0f} & {fp['max_users_changed']} & {loss} \\\\")
        if f == "F0":
            lines.append(r"\midrule")
    write("tab_x_syn_labels.tex", lines)

    lines = []
    for key, label in DETECTORS:
        r = rows[key]
        pf = r["per_fault"]
        cells = " & ".join(f"{pf[f]['detected']}/{pf[f]['positives']}" if pf[f]["positives"] else "--" for f in faults)
        lines.append(f"{label} & {r['recall']:.3f} & {cells} & {r['aa_alarms']}/{r['aa_controls']} & "
                     f"{r['benign_ab_alarms']}/{r['benign_ab_cases']} & {r['control_alarms']}/{r['aa_controls']} & "
                     f"{num(r['paired_user_requests'])} \\\\")
    write("tab_x_syn_detect.tex", lines)

    lines = []
    for key, label in (("activity:300", "Activity, 300 (primary)"), ("score_margin:300", "Score margin, 300"),
                       ("uniform:300", "Uniform, 300"), ("score_margin:100", "Score margin, 100"),
                       ("exact_snapshot", "Exact snapshot"), ("score_tolerance", "Score tolerance"),
                       ("overlap_lenient", "Overlap, lenient"), ("full_check", "Full labelled check")):
        cells = []
        for d in (0.002, 0.005, 0.01):
            r = at[d][key]
            cells.append(f"{r['detected']}/{r['synthetic_positives']} ({r['recall']:.3f})")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    write("tab_x_syn_deltas.tex", lines)

    pipes = sorted(rows["full_check"]["per_pipeline"], key=natural)
    lines = []
    for p in pipes:
        cells = []
        for key in ("activity:300", "score_margin:300", "uniform:300", "overlap_lenient", "score_tolerance"):
            slot = rows[key]["per_pipeline"].get(p, {"positives": 0, "detected": 0})
            cells.append(f"{slot['detected']}/{slot['positives']}")
        lines.append(f"{p} & {PIPELINE_INFO[p][0]} {PIPELINE_INFO[p][1]} & " + " & ".join(cells) + r" \\")
    write("tab_x_syn_pipeline.tex", lines)

    full = rows["full_check"]
    MACROS.update(
        synPositives=str(full["synthetic_positives"]), synBase=str(syn["n_base_cases"]),
        synQualityPositives=str(full["per_fault"]["F1"]["positives"] + full["per_fault"]["F4"]["positives"]),
        synContractPositives=str(full["per_fault"]["F2"]["positives"] + full["per_fault"]["F3"]["positives"]),
        synFOne=str(full["per_fault"]["F1"]["positives"]), synFTwo=str(full["per_fault"]["F2"]["positives"]),
        synFThree=str(full["per_fault"]["F3"]["positives"]), synFFour=str(full["per_fault"]["F4"]["positives"]),
        synRecallPrimary=f"{rows['activity:300']['recall']:.3f}", synRecallMargin=f"{rows['score_margin:300']['recall']:.3f}",
        synRecallUniform=f"{rows['uniform:300']['recall']:.3f}", synRecallScoreTol=f"{rows['score_tolerance']['recall']:.3f}",
        synRecallLenient=f"{rows['overlap_lenient']['recall']:.3f}", synRecallSnapshot=f"{rows['exact_snapshot']['recall']:.3f}",
        synPrimaryFOne=f"{rows['activity:300']['per_fault']['F1']['detected']}",
        synMarginFOne=f"{rows['score_margin:300']['per_fault']['F1']['detected']}",
        synSnapshotBenign=str(rows["exact_snapshot"]["benign_ab_alarms"]), synBenign=str(full["benign_ab_cases"]),
        synPanelBenign=str(max(rows[k]["benign_ab_alarms"] for k, _ in DETECTORS[:9])),
        synPanelAA=str(max(rows[k]["aa_alarms"] for k, _ in DETECTORS)),
        synControlChangedCases=str(syn["footprint"]["F0"]["cases_with_changed_lists"]),
        synControlMaxUsers=str(syn["footprint"]["F0"]["max_users_changed"]),
        synControlSnapshotAlarms=str(rows["exact_snapshot"]["control_alarms"]),
        synControlPanelAlarms=str(max(rows[k]["control_alarms"] for k, _ in DETECTORS[:9])),
        synControlFullAlarms=str(full["control_alarms"]),
        synFFourMaxUsers=str(syn["footprint"]["F4"]["max_users_changed"]),
        synFFourMedianLoss=f"${_fmt(syn['footprint']['F4']['median_full_loss'])}$",
        synPositivesLow=str(at[0.002]["full_check"]["synthetic_positives"]),
        synPositivesHigh=str(at[0.01]["full_check"]["synthetic_positives"]),
    )

    # figure: recall against the request proxy, split by quality and contract positives
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    strategies = [("uniform", "Uniform", GREY, "o"), ("activity", "Activity strata", ORANGE, "s"), ("score_margin", "Score margin", BLUE, "^")]
    for ax, (title, fs) in zip(axes, (("(a) Quality positives", ("F1", "F4")),
                                     ("(b) Contract positives", ("F2", "F3")))):
        total = sum(rows["full_check"]["per_fault"][f]["positives"] for f in fs)
        positive_faults = ", ".join(f for f in fs if rows["full_check"]["per_fault"][f]["positives"])
        for key, label, color, marker in strategies:
            xs = [200, 600, 2000]
            ys = [sum(rows[f"{key}:{n}"]["per_fault"][f]["detected"] for f in fs) / total for n in (100, 300, 1000)]
            ax.plot(xs, ys, marker=marker, color=color, label=label, lw=1.2, ms=5)
        for key, label, marker in (("exact_snapshot", "Exact snapshot", "x"), ("score_tolerance", "Score tolerance", "+"),
                                   ("overlap_lenient", "Overlap < 0.99", "d")):
            y = sum(rows[key]["per_fault"][f]["detected"] for f in fs) / total
            ax.scatter([2014], [y], marker=marker, color="#222222", s=34, zorder=4, label=label)
        ax.set_xscale("log")
        ax.set_xticks([200, 600, 2000], ["200", "600", "2,000"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(150, 2700)
        ax.set_ylim(-.04, 1.06)
        ax.set_title(f"{title} ({positive_faults}), n = {total}", fontsize=9)
        ax.set_xlabel("Paired user requests (proxy)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Recall on synthetic positives")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.2)
    fig.legend(handles, labels, frameon=False, fontsize=7.5, loc="lower center", ncol=6, bbox_to_anchor=(.5, -.07))
    save(fig, "detection_cost")

    # figure: detected / positives per fault class
    chosen = ["activity:300", "score_margin:300", "uniform:300", "exact_snapshot", "score_tolerance", "overlap_lenient", "full_check"]
    labels = ["Activity 300 (primary)", "Score margin 300", "Uniform 300", "Exact snapshot", "Score tolerance", "Overlap < 0.99",
              "Full labelled check"]
    shown = [f for f in faults if rows["full_check"]["per_fault"][f]["positives"]]
    vals = np.array([[rows[d]["per_fault"][f]["detected"] / rows[d]["per_fault"][f]["positives"] for f in shown] for d in chosen])
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.imshow(vals, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    kind = {"F1": "quality", "F2": "contract", "F3": "contract", "F4": "quality"}
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xticks(range(len(shown)), [f"{f}: {kind[f]}\n{rows['full_check']['per_fault'][f]['positives']} positives" for f in shown], fontsize=8)
    for i, d in enumerate(chosen):
        for j, f in enumerate(shown):
            a = rows[d]["per_fault"][f]
            ax.text(j, i, f"{a['detected']}/{a['positives']}", ha="center", va="center",
                    color="white" if vals[i, j] > .6 else "#182430", fontsize=8.5)
    ax.tick_params(length=0)
    save(fig, "synthetic_recall_v2")


if __name__ == "__main__":
    main()
