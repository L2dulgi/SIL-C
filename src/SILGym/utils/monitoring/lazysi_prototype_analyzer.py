#!/usr/bin/env python3
"""Offline analyzer for LazySI decoder and policy prototypes."""

from __future__ import annotations

import csv
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cloudpickle
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    Console = None
    Table = None
    Panel = None


@dataclass
class EvalEntry:
    phase_index: int
    phase_name: Optional[str]
    overall_avg_reward: Optional[float]
    detailed: Optional[Dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LazySI prototypes from saved logs.")
    parser.add_argument(
        "--log-dir",
        required=True,
        type=Path,
        help="Experiment log directory containing eval_results.json and skills/ pickles.",
    )
    parser.add_argument(
        "--output",
        default="prototype_report.json",
        help="Report filename to create inside the log directory.",
    )
    parser.add_argument(
        "--csv-output",
        default="prototype_prototypes.csv",
        help="CSV filename for exporting per-prototype statistics.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of decoder/policy entries to highlight in detail.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console summary output.",
    )
    return parser.parse_args()


def ensure_log_dir(path: Path) -> Path:
    log_dir = path.expanduser().resolve()
    if not log_dir.exists() or not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    return log_dir


def load_eval_results(log_dir: Path) -> Tuple[Optional[Path], List[EvalEntry]]:
    candidates = sorted(log_dir.glob("eval_results*.json"))
    if not candidates:
        return None, []
    eval_path = candidates[0]
    with eval_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    entries: List[EvalEntry] = []
    for key, payload in raw.items():
        payload = payload or {}
        phase_index = int(key)
        phase_name = payload.get("phase_name")
        overall_avg = payload.get("overall_avg_reward")
        detailed = payload.get("detailed")
        entries.append(EvalEntry(phase_index, phase_name, overall_avg, detailed))
    entries.sort(key=lambda e: e.phase_index)
    return eval_path, entries


def collect_interface_snapshots(log_dir: Path) -> List[Tuple[str, Path]]:
    skills_dir = log_dir / "skills"
    if not skills_dir.exists():
        return []
    snapshots: List[Tuple[str, Path]] = []
    for path in sorted(skills_dir.glob("interface_*.pkl")):
        tag = path.stem.split("interface_", 1)[-1]
        snapshots.append((tag, path))
    return snapshots


def collect_policy_snapshots(log_dir: Path) -> List[Tuple[str, Path]]:
    policy_dir = log_dir / "policy"
    if not policy_dir.exists():
        return []
    snapshots: List[Tuple[str, Path]] = []
    for subdir in sorted(policy_dir.glob("policy_*")):
        if not subdir.is_dir():
            continue
        for path in sorted(subdir.glob("*.pkl")):
            tag = f"{subdir.name}/{path.stem}"
            snapshots.append((tag, path))
    return snapshots


def load_interface(path: Path):
    with path.open("rb") as f:
        return cloudpickle.load(f)


def load_policy(path: Path):
    with path.open("rb") as f:
        return cloudpickle.load(f)


def safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def format_float(value: Optional[float], digits: int = 4) -> str:
    """Format float with readable scientific notation when appropriate."""
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if np.isnan(value):
        return "-"
    if value == 0:
        return "0"

    abs_value = abs(value)
    small_threshold = 1e-3
    large_threshold = 1e4
    if abs_value < small_threshold or abs_value >= large_threshold:
        formatted = f"{value:.{digits}e}"
        mantissa, exp = formatted.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        if not mantissa:
            mantissa = "0"
        exp_sign = "+" if exp.startswith("+") else "-"
        exp_value = exp[1:].lstrip("0") or "0"
        return f"{mantissa}e{exp_sign}{exp_value}"

    formatted = f"{value:.{digits}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def format_int(value: Optional[int]) -> str:
    """Format integer or return dash."""
    if value is None:
        return "-"
    try:
        return f"{int(value)}"
    except (TypeError, ValueError):
        return "-"


def format_variance_mean(stats: Optional[Dict[str, object]]) -> str:
    if not stats:
        return "-"
    return format_float(stats.get("variance_mean"))


def snapshot_variance(snapshot: Optional[Dict[str, object]]) -> str:
    if not snapshot:
        return "-"
    return format_float(snapshot.get("variance_mean"))


def ma_proto_stats(protos: Iterable) -> Optional[Dict[str, object]]:
    proto_list = [p for p in protos if p is not None]
    if not proto_list:
        return None

    counts: List[int] = []
    dims: List[int] = []
    variance_means: List[float] = []
    variance_maxes: List[float] = []
    threshold_means: List[float] = []
    threshold_maxes: List[float] = []

    for proto in proto_list:
        mean = np.asarray(proto.mean)
        if mean.ndim == 1:
            counts.append(int(mean.shape[0]))
            dims.append(int(mean.shape[0]))
        else:
            counts.append(int(mean.shape[0]))
            dims.append(int(mean.shape[1]))
        variance = getattr(proto, "variance", None)
        if variance is not None:
            var_arr = np.asarray(variance)
            if var_arr.size:
                variance_means.append(float(np.mean(var_arr)))
                variance_maxes.append(float(np.max(var_arr)))
        threshold = getattr(proto, "threshold", None)
        if threshold is not None:
            thr_arr = np.asarray(threshold)
            if thr_arr.size:
                threshold_means.append(float(np.mean(thr_arr)))
                threshold_maxes.append(float(np.max(thr_arr)))

    stats: Dict[str, object] = {
        "prototype_count": {
            "avg": float(np.mean(counts)),
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
        },
        "feature_dim_mode": int(np.bincount(dims).argmax()) if dims else None,
    }
    if variance_means:
        stats["variance_mean"] = float(np.mean(variance_means))
        stats["variance_max"] = float(np.max(variance_maxes))
    if threshold_means:
        stats["threshold_mean"] = float(np.mean(threshold_means))
        stats["threshold_max"] = float(np.max(threshold_maxes))

    first = proto_list[0]
    stats["distance_type"] = getattr(first, "distance_type", None)
    stats["threshold_type"] = getattr(first, "threshold_type", None)
    stats["confidence_interval"] = getattr(first, "confidence_interval", None)
    return stats


def single_ma_snapshot(proto) -> Optional[Dict[str, object]]:
    if proto is None:
        return None
    mean = np.asarray(proto.mean)
    if mean.ndim == 1:
        count = int(mean.shape[0])
        dim = int(mean.shape[0])
    else:
        count = int(mean.shape[0])
        dim = int(mean.shape[1])
    variance = getattr(proto, "variance", None)
    variance_mean = None
    variance_max = None
    if variance is not None:
        var_arr = np.asarray(variance)
        if var_arr.size:
            variance_mean = float(np.mean(var_arr))
            variance_max = float(np.max(var_arr))
    threshold = getattr(proto, "threshold", None)
    threshold_mean = None
    if threshold is not None:
        thr_arr = np.asarray(threshold)
        if thr_arr.size:
            threshold_mean = float(np.mean(thr_arr))
    return {
        "count": count,
        "feature_dim": dim,
        "variance_mean": variance_mean,
        "variance_max": variance_max,
        "threshold_mean": threshold_mean,
    }


def populate_snapshot_metrics(row: Dict[str, object], prefix: str, snapshot: Optional[Dict[str, object]]) -> None:
    row[f"{prefix}_count"] = snapshot.get("count") if snapshot else None
    row[f"{prefix}_dim"] = snapshot.get("feature_dim") if snapshot else None
    row[f"{prefix}_var_mean"] = snapshot.get("variance_mean") if snapshot else None
    row[f"{prefix}_var_max"] = snapshot.get("variance_max") if snapshot else None
    row[f"{prefix}_threshold_mean"] = snapshot.get("threshold_mean") if snapshot else None


def l2_norm(array: Optional[np.ndarray]) -> Optional[float]:
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.size == 0:
        return None
    return float(np.linalg.norm(arr))

def analyze_decoder(interface, top_k: int) -> Dict[str, object]:
    entries = list((skill_id, entry) for skill_id, entry in (interface.entry_skill_map or {}).items())
    entries.sort(key=lambda item: item[0])
    summary: Dict[str, object] = {"count": len(entries)}

    if not entries:
        return summary

    data_counts = [int(entry.data_count) for _, entry in entries]
    summary["data_points"] = {
        "total": int(np.sum(data_counts)),
        "avg": float(np.mean(data_counts)),
        "min": int(np.min(data_counts)),
        "max": int(np.max(data_counts)),
    }

    summary["state_prototypes"] = ma_proto_stats(entry.state_prototypes for _, entry in entries)
    summary["action_prototypes"] = ma_proto_stats(entry.action_prototypes for _, entry in entries)
    summary["subgoal_prototypes"] = ma_proto_stats(entry.subgoal_prototypes for _, entry in entries)

    top_entries = sorted(entries, key=lambda item: item[1].data_count, reverse=True)[:max(0, top_k)]
    summary["top_skills"] = [
        {
            "skill_id": int(skill_id),
            "decoder_id": int(entry.decoder_id),
            "data_count": int(entry.data_count),
            "state": single_ma_snapshot(entry.state_prototypes),
            "action": single_ma_snapshot(entry.action_prototypes),
            "subgoal": single_ma_snapshot(entry.subgoal_prototypes),
        }
        for skill_id, entry in top_entries
    ]
    return summary



def collect_decoder_rows(tag: str, snapshot_path: str, interface) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not getattr(interface, "entry_skill_map", None):
        return rows
    for skill_id, entry in interface.entry_skill_map.items():
        state_snapshot = single_ma_snapshot(entry.state_prototypes)
        action_snapshot = single_ma_snapshot(entry.action_prototypes)
        subgoal_snapshot = single_ma_snapshot(entry.subgoal_prototypes)
        row: Dict[str, object] = {
            "phase": tag,
            "scope": "decoder",
            "skill_id": int(skill_id),
            "prototype_id": None,
            "decoder_id": int(entry.decoder_id),
            "data_count": int(entry.data_count),
            "distance_type": getattr(entry.state_prototypes, "distance_type", None),
            "threshold_type": getattr(entry.state_prototypes, "threshold_type", None),
            "confidence_interval": getattr(entry.state_prototypes, "confidence_interval", None),
            "subgoal_norm": None,
            "snapshot": snapshot_path,
        }
        populate_snapshot_metrics(row, "state", state_snapshot)
        populate_snapshot_metrics(row, "action", action_snapshot)
        populate_snapshot_metrics(row, "subgoal", subgoal_snapshot)
        rows.append(row)
    return rows

CSV_FIELDNAMES = [
    "phase",
    "scope",
    "skill_id",
    "prototype_id",
    "decoder_id",
    "data_count",
    "distance_type",
    "threshold_type",
    "confidence_interval",
    "state_count",
    "state_dim",
    "state_var_mean",
    "state_var_max",
    "state_threshold_mean",
    "action_count",
    "action_dim",
    "action_var_mean",
    "action_var_max",
    "action_threshold_mean",
    "subgoal_count",
    "subgoal_dim",
    "subgoal_var_mean",
    "subgoal_var_max",
    "subgoal_threshold_mean",
    "subgoal_norm",
    "snapshot",
]


def collect_policy_rows(tag: str, snapshot_path: str, prototypes: Optional[Dict[int, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not prototypes:
        return rows
    for proto_id, entry in prototypes.items():
        state_snapshot = single_ma_snapshot(entry.state_prototypes)
        row: Dict[str, object] = {
            "phase": tag,
            "scope": "policy",
            "skill_id": None,
            "prototype_id": int(proto_id),
            "decoder_id": None,
            "data_count": int(entry.data_count),
            "distance_type": getattr(entry.state_prototypes, "distance_type", None),
            "threshold_type": getattr(entry.state_prototypes, "threshold_type", None),
            "confidence_interval": getattr(entry.state_prototypes, "confidence_interval", None),
            "subgoal_norm": l2_norm(getattr(entry, 'subgoal', None)),
            "snapshot": snapshot_path,
        }
        populate_snapshot_metrics(row, "state", state_snapshot)
        populate_snapshot_metrics(row, "action", None)
        populate_snapshot_metrics(row, "subgoal", None)
        rows.append(row)
    return rows



def analyze_policy_prototypes(prototypes: Optional[Dict[int, object]], top_k: int) -> Dict[str, object]:
    entries = []
    if prototypes:
        entries = list((pid, entry) for pid, entry in prototypes.items())
        entries.sort(key=lambda item: item[0])
    summary: Dict[str, object] = {"count": len(entries)}

    if not entries:
        return summary

    data_counts = [int(entry.data_count) for _, entry in entries]
    summary["data_points"] = {
        "total": int(np.sum(data_counts)),
        "avg": float(np.mean(data_counts)),
        "min": int(np.min(data_counts)),
        "max": int(np.max(data_counts)),
    }

    summary["state_prototypes"] = ma_proto_stats(entry.state_prototypes for _, entry in entries)

    top_entries = sorted(entries, key=lambda item: item[1].data_count, reverse=True)[:max(0, top_k)]
    summary["top_prototypes"] = [
        {
            "prototype_id": int(pid),
            "data_count": int(entry.data_count),
            "subgoal_norm": l2_norm(getattr(entry, 'subgoal', None)),
            "state": single_ma_snapshot(entry.state_prototypes),
        }
        for pid, entry in top_entries
    ]
    return summary


def gather_eval_summary(entries: List[EvalEntry], phase_tag: str) -> Optional[Dict[str, object]]:
    if not entries:
        return None
    matches = [e for e in entries if e.phase_name and e.phase_name.split("/")[-1] == phase_tag]
    if not matches:
        return None

    overall_values = [safe_float(e.overall_avg_reward) for e in matches if e.overall_avg_reward is not None]

    tasks: Dict[str, List[float]] = defaultdict(list)
    for entry in matches:
        if not entry.detailed:
            continue
        for task_name, payload in entry.detailed.items():
            avg = payload.get("avg_reward") if isinstance(payload, dict) else None
            if avg is not None:
                tasks[task_name].append(float(avg))

    task_summary = {
        task: {
            "avg": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }
        for task, values in tasks.items()
        if values
    }

    summary: Dict[str, object] = {
        "num_eval_phases": len(matches),
        "overall_avg_reward": {
            "avg": float(np.mean(overall_values)) if overall_values else None,
            "min": float(np.min(overall_values)) if overall_values else None,
            "max": float(np.max(overall_values)) if overall_values else None,
            "count": len(overall_values),
        },
        "tasks": task_summary,
    }
    return summary


def build_report(log_dir: Path, top_k: int) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    eval_path, eval_entries = load_eval_results(log_dir)
    decoder_snapshots = collect_interface_snapshots(log_dir)
    policy_snapshots = collect_policy_snapshots(log_dir)

    decoder_phases: List[Dict[str, object]] = []
    policy_phases: List[Dict[str, object]] = []

    report: Dict[str, object] = {
        "log_dir": str(log_dir),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "eval_results_path": str(eval_path) if eval_path else None,
        "num_snapshots": len(decoder_snapshots),
        "num_decoder_snapshots": len(decoder_snapshots),
        "num_policy_snapshots": len(policy_snapshots),
        "phases": decoder_phases,
        "decoder_phases": decoder_phases,
        "policy_phases": policy_phases,
    }
    csv_rows: List[Dict[str, object]] = []

    for tag, interface_path in decoder_snapshots:
        interface = load_interface(interface_path)
        decoder_stats = analyze_decoder(interface, top_k)
        eval_summary = gather_eval_summary(eval_entries, tag)
        rel_snapshot = str(interface_path.relative_to(log_dir))

        phase_info = {
            "phase": tag,
            "interface_snapshot": rel_snapshot,
            "decoder": decoder_stats,
            "eval": eval_summary,
        }
        decoder_phases.append(phase_info)

        csv_rows.extend(collect_decoder_rows(tag, rel_snapshot, interface))

    for tag, policy_path in policy_snapshots:
        try:
            policy_obj = load_policy(policy_path)
        except Exception as exc:
            policy_stats = {"count": 0, "error": str(exc)}
            prototypes = None
        else:
            prototypes = getattr(policy_obj, "subtask_prototypes", None)
            policy_stats = analyze_policy_prototypes(prototypes, top_k)
        eval_summary = gather_eval_summary(eval_entries, tag)
        rel_snapshot = str(policy_path.relative_to(log_dir))

        phase_info = {
            "phase": tag,
            "policy_snapshot": rel_snapshot,
            "policy": policy_stats,
            "eval": eval_summary,
        }
        policy_phases.append(phase_info)

        csv_rows.extend(collect_policy_rows(tag, rel_snapshot, prototypes))

    return report, csv_rows


def print_console_summary(report: Dict[str, object]) -> None:
    decoder_phases = report.get("decoder_phases") or report.get("phases", [])
    policy_phases = report.get("policy_phases", [])

    if Console is None or Table is None:
        print(f"Analyzed log directory: {report['log_dir']}")
        print(f"Decoder snapshots: {len(decoder_phases)}")
        for phase in decoder_phases:
            decoder_count = phase.get("decoder", {}).get("count", 0)
            print(f"  - {phase.get('phase', '-')}: decoder_skills={decoder_count}")
        if policy_phases:
            print(f"Policy snapshots: {len(policy_phases)}")
            for phase in policy_phases:
                policy_count = phase.get("policy", {}).get("count", 0)
                print(f"  - {phase.get('phase', '-')}: policy_prototypes={policy_count}")
        return

    console = Console()
    console.print(f"[bold]LazySI prototype analysis[/bold] — [cyan]{report['log_dir']}[/cyan]")
    console.print(
        f"Decoder snapshots: {len(decoder_phases)} | Policy snapshots: {len(policy_phases)}",
        style="dim",
    )
    console.print()

    if decoder_phases:
        overview = Table(title="Decoder Overview", header_style="bold cyan")
        overview.add_column("Phase", style="cyan", no_wrap=True)
        overview.add_column("Decoder#", justify="right")
        overview.add_column("Decoder VarMean", justify="right")
        overview.add_column("Eval Avg", justify="right")
        overview.add_column("Snapshot", style="dim")

        for phase in decoder_phases:
            decoder = phase.get("decoder", {})
            eval_summary = phase.get("eval") or {}
            overall = (eval_summary.get("overall_avg_reward") or {})
            overview.add_row(
                phase.get("phase", "-"),
                format_int(decoder.get("count")),
                format_variance_mean(decoder.get("state_prototypes")),
                format_float(overall.get("avg")),
                phase.get("interface_snapshot", "-"),
            )
        console.print(overview)

        for phase in decoder_phases:
            phase_name = phase.get("phase", "-")
            decoder_top = phase.get("decoder", {}).get("top_skills") or []
            if decoder_top:
                decoder_table = Table(show_header=True, header_style="bold magenta")
                decoder_table.title = f"{phase_name} · Decoder Top {len(decoder_top)}"
                decoder_table.add_column("Skill", justify="right")
                decoder_table.add_column("Decoder", justify="right")
                decoder_table.add_column("Samples", justify="right")
                decoder_table.add_column("State VarMean", justify="right")
                decoder_table.add_column("Action VarMean", justify="right")
                decoder_table.add_column("Subgoal VarMean", justify="right")
                for item in decoder_top:
                    decoder_table.add_row(
                        format_int(item.get("skill_id")),
                        format_int(item.get("decoder_id")),
                        format_int(item.get("data_count")),
                        snapshot_variance(item.get("state")),
                        snapshot_variance(item.get("action")),
                        snapshot_variance(item.get("subgoal")),
                    )
                console.print(Panel(decoder_table, border_style="magenta"))

            eval_summary = phase.get("eval")
            if eval_summary:
                eval_table = Table(show_header=False)
                eval_table.title = f"{phase_name} · Evaluation"
                eval_table.add_column("Metric", style="bold yellow")
                eval_table.add_column("Value", justify="right")
                eval_table.add_row("Eval phases", format_int(eval_summary.get("num_eval_phases")))
                overall = eval_summary.get("overall_avg_reward") or {}
                eval_table.add_row("Overall avg", format_float(overall.get("avg")))
                eval_table.add_row("Overall min/max", f"{format_float(overall.get('min'))} / {format_float(overall.get('max'))}")
                tasks = eval_summary.get("tasks") or {}
                if tasks:
                    task_items = sorted(
                        tasks.items(),
                        key=lambda item: item[1].get("avg") if item[1].get("avg") is not None else float('-inf'),
                        reverse=True,
                    )
                    for task, stats in task_items[:3]:
                        eval_table.add_row(f"Task {task} avg", format_float(stats.get("avg")))
                console.print(Panel(eval_table, border_style="yellow"))

    if policy_phases:
        overview = Table(title="Policy Overview", header_style="bold cyan")
        overview.add_column("Phase", style="green", no_wrap=True)
        overview.add_column("Policy#", justify="right")
        overview.add_column("Policy VarMean", justify="right")
        overview.add_column("Eval Avg", justify="right")
        overview.add_column("Snapshot", style="dim")

        for phase in policy_phases:
            policy = phase.get("policy", {})
            eval_summary = phase.get("eval") or {}
            overall = (eval_summary.get("overall_avg_reward") or {})
            overview.add_row(
                phase.get("phase", "-"),
                format_int(policy.get("count")),
                format_variance_mean(policy.get("state_prototypes")),
                format_float(overall.get("avg")),
                phase.get("policy_snapshot", "-"),
            )
        console.print(overview)

        for phase in policy_phases:
            phase_name = phase.get("phase", "-")
            policy_top = phase.get("policy", {}).get("top_prototypes") or []
            if policy_top:
                policy_table = Table(show_header=True, header_style="bold green")
                policy_table.title = f"{phase_name} · Policy Top {len(policy_top)}"
                policy_table.add_column("Prototype", justify="right")
                policy_table.add_column("Samples", justify="right")
                policy_table.add_column("State VarMean", justify="right")
                policy_table.add_column("‖Subgoal‖", justify="right")
                for item in policy_top:
                    policy_table.add_row(
                        format_int(item.get("prototype_id")),
                        format_int(item.get("data_count")),
                        snapshot_variance(item.get("state")),
                        format_float(item.get("subgoal_norm")),
                    )
                console.print(Panel(policy_table, border_style="green"))

            eval_summary = phase.get("eval")
            if eval_summary:
                eval_table = Table(show_header=False)
                eval_table.title = f"{phase_name} · Evaluation"
                eval_table.add_column("Metric", style="bold yellow")
                eval_table.add_column("Value", justify="right")
                eval_table.add_row("Eval phases", format_int(eval_summary.get("num_eval_phases")))
                overall = eval_summary.get("overall_avg_reward") or {}
                eval_table.add_row("Overall avg", format_float(overall.get("avg")))
                eval_table.add_row("Overall min/max", f"{format_float(overall.get('min'))} / {format_float(overall.get('max'))}")
                tasks = eval_summary.get("tasks") or {}
                if tasks:
                    task_items = sorted(
                        tasks.items(),
                        key=lambda item: item[1].get("avg") if item[1].get("avg") is not None else float('-inf'),
                        reverse=True,
                    )
                    for task, stats in task_items[:3]:
                        eval_table.add_row(f"Task {task} avg", format_float(stats.get("avg")))
                console.print(Panel(eval_table, border_style="yellow"))


def main() -> None:
    args = parse_args()
    log_dir = ensure_log_dir(args.log_dir)
    report, csv_rows = build_report(log_dir, top_k=args.top_k)
    output_path = log_dir / args.output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = log_dir / args.csv_output
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    if not args.quiet:
        print_console_summary(report)
        print(f"Report written to: {output_path}")
        print(f"CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
