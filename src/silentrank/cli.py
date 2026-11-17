"""Command-line entry point (operations in service.py)."""

from __future__ import annotations

import argparse
import json
import sys

from silentrank import __version__, service
from silentrank.orchestrator import ledger

def _print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print("(none)")
        return
    cells = [[("" if row.get(col) is None else str(row.get(col))) for col in columns] for row in rows]
    widths = [max(len(col), *(len(line[i]) for line in cells)) for i, col in enumerate(columns)]
    print("  ".join(col.ljust(width) for col, width in zip(columns, widths)))
    print("  ".join("-" * width for width in widths))
    for line in cells:
        print("  ".join(cell.ljust(width) for cell, width in zip(line, widths)))


def _change_text(change) -> str:
    if not isinstance(change, dict):
        return ""
    if "package" in change:
        return f"{change['package']} {change.get('old')} -> {change.get('new')}"
    return ", ".join(f"{key}={value}" for key, value in change.items())


def _cmd_init(_args) -> int:
    print(json.dumps(service.init(), indent=2))
    return 0


def _cmd_case_list(args) -> int:
    rows = service.list_cases(collection=args.collection, status=args.status)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        row["what"] = _change_text(row.get("change")) or (row.get("project") or "")
        progress = row.get("ledger")
        row["runs"] = progress["n_runs"] if isinstance(progress, dict) else (progress or "")
    _print_table(rows, ["id", "kind", "collection", "status", "what", "runs"])
    return 0


def _cmd_case_show(args) -> int:
    try:
        print(json.dumps(service.get_case(args.case_id), indent=2))
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 1
    return 0


def _cmd_query(args) -> int:
    try:
        result = service.query_ledger(args.sql, max_rows=args.max_rows)
    except (service.QueryRejected, ledger.LedgerMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_table([dict(zip(result["columns"], row)) for row in result["rows"]], result["columns"])
    if result["truncated"]:
        print(f"(truncated at {result['row_count']} rows)")
    return 0


def _cmd_env(args) -> int:
    from silentrank.orchestrator import envs, runner  # imported lazily: needs Docker

    try:
        if args.env_cmd == "screen-pilot":
            result = envs.screen_pilot(args.group, args.split)
        elif args.env_cmd == "prepare-cases":
            result = envs.prepare_cases(args.case_ids)
            result["status"] = "ok" if all(v == "ok" for v in result["environments"].values()) else "env_failed"
        elif args.env_cmd == "recover-suites":
            result = envs.recover_suites(run_missing=args.run_missing)
        elif args.env_cmd == "screen" and args.package:
            result = envs.screen(args.env_id, packages=args.package)
        else:
            result = getattr(envs, args.env_cmd)(args.env_id)
    except (FileNotFoundError, ValueError, runner.DockerUnavailable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.env_cmd == "probe" and result["status"] == "ok":
        report = result.pop("report")
        numpy_info = report["numpy"]
        result.update(
            python=report["python"].split()[0],
            numpy=numpy_info.get("version"),
            default_int=numpy_info.get("default_int"),
            cpu=report["cpu"]["model"],
            logical_cpus=report["cpu"]["logical_cpus"],
            avx512_flags=[flag for flag in report["cpu"]["flags"] if flag.startswith("avx512")],
            numpy_simd_enabled=numpy_info.get("simd_enabled"),
            thread_env=report["thread_env"],
        )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


def _cmd_run(args) -> int:
    from silentrank.orchestrator import pilot

    try:
        result = pilot.run_case(args.case, inference_only=args.inference_only, timing_context=args.timing_context)
    except (ValueError, OSError, ledger.LedgerMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"ok", "not_applicable"} else 1


def _cmd_analysis(args) -> int:
    from silentrank.orchestrator import comparisons

    try:
        analysis = comparisons.compare_case if args.cmd == "compare" else comparisons.verdict_case
        result = analysis(args.case)
    except (ValueError, OSError, ledger.LedgerMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_diagnose(args) -> int:
    from silentrank.orchestrator.diagnosis import diagnose_case

    try:
        if args.stage == "S3":
            from silentrank.orchestrator.neighbor_diagnosis import diagnose_neighbors

            result = diagnose_neighbors(args.case, seed=args.seed)
        else:
            result = diagnose_case(args.case)
    except (ValueError, OSError, ledger.LedgerMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] == "reproduced" else 1


def _cmd_adjudicate(args) -> int:
    from silentrank.orchestrator.adjudication import record

    try:
        result = record(args.case, args.mechanism, args.cause, args.evidence, args.adjudicator)
    except (ValueError, ledger.LedgerMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def _cmd_synthetic(args) -> int:
    from silentrank.detect.synthetic import run_all

    try:
        summary = run_all(args.case)
    except (ValueError, OSError, ledger.LedgerMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    primary = summary["primary"]
    for row in summary["rows"]:
        if row["delta"] == primary["delta"]:
            print(f"{row['detector']:<22} recall={row['recall']!s:<6} aa_alarms={row['aa_alarms']}/{row['aa_controls']} "
                  f"benign_alarms={row['benign_ab_alarms']}/{row['benign_ab_cases']} requests={row['paired_user_requests']}")
    return 0


def _cmd_probes(args) -> int:
    from silentrank.detect.analysis import analyze_probes

    try:
        result = analyze_probes(args.case)
    except (ValueError, OSError, ledger.LedgerMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = {key: result[key] for key in ("status", "artifact", "summary_by_collection")}
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_report(_args) -> int:
    from silentrank.orchestrator.pilot_report import summarize_pilot

    try:
        print(json.dumps(summarize_pilot(), indent=2))
    except (ValueError, OSError, ledger.LedgerMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="silentrank", description=__doc__)
    parser.add_argument("--version", action="version", version=f"silentrank {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create the ledger and sync manifests into it").set_defaults(func=_cmd_init)

    case = sub.add_parser("case", help="list or show transitions, leads and cases")
    case_sub = case.add_subparsers(dest="case_cmd", required=True)
    case_list = case_sub.add_parser("list")
    case_list.add_argument("--collection")
    case_list.add_argument("--status")
    case_list.add_argument("--json", action="store_true")
    case_list.set_defaults(func=_cmd_case_list)
    case_show = case_sub.add_parser("show")
    case_show.add_argument("case_id")
    case_show.set_defaults(func=_cmd_case_show)

    query = sub.add_parser("query", help="run one read-only SELECT against the ledger")
    query.add_argument("sql")
    query.add_argument("--max-rows", type=int, default=200)
    query.set_defaults(func=_cmd_query)

    env = sub.add_parser("env", help="build, probe or screen an environment under test (needs Docker)")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    for name in ("build", "probe", "screen"):
        command = env_sub.add_parser(name)
        command.add_argument("env_id", help="folder name under envs/")
        if name == "screen":
            command.add_argument("--package", action="append", help="run only the named upstream test suite")
    env_sub.add_parser("screen-pilot").add_argument("group", nargs="?", default="base",
                                                   choices=["base", "implicit", "surprise", "sar", "cornac", "torch", "torch_gpu", "snapshot", "all"])
    env_sub.choices["screen-pilot"].add_argument("--split", choices=["dev", "heldout"])
    prepare = env_sub.add_parser("prepare-cases", help="build and probe only the environments some cases need (second host)")
    prepare.add_argument("case_ids", nargs="+")
    recover = env_sub.add_parser("recover-suites", help="recover completed upstream suite outcomes from saved logs")
    recover.add_argument("--run-missing", action="store_true", help="run suites still missing after log recovery")
    env.set_defaults(func=_cmd_env)

    run = sub.add_parser("run", help="run a screened pilot case with baseline tuning and A/A controls")
    run.add_argument("--case", required=True)
    run.add_argument("--inference-only", action="store_true", help="stop after A/B inference")
    run.add_argument("--timing-context", help="record host workload context in timing metadata")
    run.set_defaults(func=_cmd_run)

    for name in ("compare", "verdict"):
        analysis = sub.add_parser(name, help=f"{name} a pilot case from its logged runs")
        analysis.add_argument("--case", required=True)
        analysis.set_defaults(func=_cmd_analysis)

    diagnose = sub.add_parser("diagnose", help="replay a localized reference ranking or neighbor selection divergence")
    diagnose.add_argument("--case", required=True)
    diagnose.add_argument("--stage", default="S7", choices=["S3", "S7"])
    diagnose.add_argument("--seed", type=int, default=17, help="paired training seed for the S3 replay")
    diagnose.set_defaults(func=_cmd_diagnose)

    adjudicate = sub.add_parser("adjudicate", help="record a mechanism and cause class for a real case (append-only)")
    adjudicate.add_argument("--case", required=True)
    adjudicate.add_argument("--mechanism", required=True)
    adjudicate.add_argument("--cause", required=True, choices=["upstream_defect", "downstream_incompatibility",
                                                               "intended_semantic_change", "numerical_variation", "unresolved"])
    adjudicate.add_argument("--evidence", action="append", required=True, help="repo path or URL (repeatable)")
    adjudicate.add_argument("--adjudicator", default="pending researcher review")
    adjudicate.set_defaults(func=_cmd_adjudicate)
    synthetic = sub.add_parser("synthetic", help="run the declared synthetic positive controls on held-out cases")
    synthetic.add_argument("--case", action="append", help="limit to one base case (repeatable)")
    synthetic.set_defaults(func=_cmd_synthetic)
    probes = sub.add_parser("probes", help="analyze development probe panels from logged inference runs")
    probes.add_argument("--case", action="append", help="limit analysis to the specified pilot case")
    probes.set_defaults(func=_cmd_probes)

    report = sub.add_parser("report", help="summarize logged pilot comparisons")
    report.add_argument("phase", choices=["pilot"])
    report.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Ledger text (uv resolver messages, paths) is not ASCII; Windows consoles default to cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
