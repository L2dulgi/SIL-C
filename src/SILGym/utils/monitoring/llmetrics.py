import argparse
import os
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from SILGym.utils.logger import get_logger

logger = get_logger(__name__)

def calculate_metrics(tasks_data, base_fwt=None):
    """
    Calculate per-task and overall (FWT, BWT, AUC) metrics based on the given tasks_data.
    """
    per_task = {}
    for task, values in tasks_data.items():
        values.sort(key=lambda x: x[0])
        if base_fwt is not None and task in base_fwt:
            # When base_fwt is provided, calculate differences relative to the base FWT value.
            fwt_value = base_fwt[task]
            differences = [val - fwt_value for _, val in values]
            bwt = sum(differences) / len(differences) if differences else 0.0
            auc = (fwt_value + sum(val for _, val in values)) / (len(values) + 1)
        else:
            # FWT is the avg_reward from the first evaluation.
            fwt_value = values[0][1]
            if base_fwt is not None:
                logger.warning(f"Task {task} found in base FWT, using first evaluation value as FWT: {fwt_value}")
            if len(values) > 1:
                differences = [val - fwt_value for _, val in values[1:]]
                bwt = sum(differences) / len(differences)
            else:
                bwt = 0.0
            auc = sum(val for _, val in values) / len(values)
        per_task[task] = (fwt_value, bwt, auc)
    
    if per_task:
        overall_fwt = sum(fwt for fwt, _, _ in per_task.values()) / len(per_task)
        overall_bwt = sum(bwt for _, bwt, _ in per_task.values()) / len(per_task)
        overall_auc = sum(auc for _, _, auc in per_task.values()) / len(per_task)
    else:
        overall_fwt = overall_bwt = overall_auc = 0.0
    overall = (overall_fwt, overall_bwt, overall_auc)
    return per_task, overall

def calculate_last_metric(tasks_data):
    """
    For each task, take the last (i.e. highest phase) evaluation's avg_reward as the "Last" metric,
    and return per-task and overall (average) Last.
    """
    per_task_last = {}
    for task, values in tasks_data.items():
        values.sort(key=lambda x: x[0])
        per_task_last[task] = values[-1][1]  
    overall_last = sum(per_task_last.values()) / len(per_task_last) if per_task_last else 0.0
    return per_task_last, overall_last

def build_tasks_data(data, group_filter=None):
    """
    Build tasks_data from the raw data applying the given group_filter.
    Returns a dictionary mapping each task (or policy) to a list of (phase, avg_reward) tuples.
    """
    tasks_data = {}
    for phase_key in sorted(data.keys(), key=lambda x: int(x)):
        entry = data[phase_key]
        targets = entry.get("train_targets", [])
        if group_filter and not group_filter(targets):
            continue
        detailed = entry.get("detailed")
        if not detailed:
            continue
        phase = int(phase_key)
        for key, info in detailed.items():
            avg_reward = info.get("avg_reward")
            if avg_reward is None:
                continue
            tasks_data.setdefault(key, []).append((phase, avg_reward))
    return tasks_data

def compute_task_metrics_old(data, group_filter=None, base_fwt=None):
    """
    Process the old version of the data.
    """
    tasks_data = {}
    for phase_key in sorted(data.keys(), key=lambda x: int(x)):
        entry = data[phase_key]
        targets = entry.get("train_targets", [])
        if group_filter and not group_filter(targets):
            continue
        detailed = entry.get("detailed")
        if not detailed:
            continue
        phase = int(phase_key)
        for task, task_info in detailed.items():
            avg_reward = task_info.get("avg_reward")
            if avg_reward is None:
                continue
            tasks_data.setdefault(task, []).append((phase, avg_reward))
    return calculate_metrics(tasks_data, base_fwt)

def compute_policy_metrics(data, group_filter=None, base_fwt=None):
    """
    Process the new version of the data.
    """
    tasks_data = {}
    for phase_key in sorted(data.keys(), key=lambda x: int(x)):
        entry = data[phase_key]
        targets = entry.get("train_targets", [])
        if group_filter and not group_filter(targets):
            continue
        detailed = entry.get("detailed")
        if not detailed:
            continue
        phase = int(phase_key)
        for policy_id, policy_info in detailed.items():
            avg_reward = policy_info.get("avg_reward")
            if avg_reward is None:
                continue
            tasks_data.setdefault(policy_id, []).append((phase, avg_reward))
    return calculate_metrics(tasks_data, base_fwt)

def auto_select_compute_metrics(data, group_filter=None, base_fwt=None):
    """
    Automatically select which function to use based on the keys in the first non-empty "detailed" entry.
    """
    for entry in data.values():
        detailed = entry.get("detailed")
        if detailed and isinstance(detailed, dict) and len(detailed) > 0:
            first_key = next(iter(detailed.keys()))
            if first_key.startswith("policy"):
                return compute_policy_metrics(data, group_filter, base_fwt)
            else:
                return compute_task_metrics_old(data, group_filter, base_fwt)
    # Default to old version if no detailed data is found.
    return compute_task_metrics_old(data, group_filter, base_fwt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AsyncSIL Metric Calculator with FWT, BWT, AUC, and Last.')
    parser.add_argument('-al', '--algo', type=str, help='Algorithm', default='ptgm')
    parser.add_argument('-e', '--env', type=str, help='Environment', default='kitchen')
    parser.add_argument('-p', '--path', type=str, help='Path to logs directory', default='none')
    parser.add_argument('-g', '--grep', type=str, nargs='*', help='Grep filters for evaluation files (all must be present in the full path including filename)', default=[])
    parser.add_argument('-i', '--exp_id', type=str, help='Experiment ID', default='DEF')
    parser.add_argument('-u', '--unit_trace', action='store_true', help='Display a table of average reward per task per phase with decoder phase')
    parser.add_argument('--detailed_auc', action='store_true', help='Display per-task AUC metrics')
    args = parser.parse_args()

    # Use the provided path; otherwise, default to logs/{env}/ as the base path.
    if args.path != 'none':
        base_path = args.path
    else:
        base_path = f'logs/{args.env}/'

    # Recursively search for any JSON files starting with eval_results
    eval_paths = []
    for root, dirs, files in os.walk(base_path):
        for file_name in files:
            # Check if file starts with eval_results and ends with .json
            if file_name.startswith('eval_results') and file_name.endswith('.json'):
                full_path = os.path.join(root, file_name)
                # Apply grep filter to both directory path and filename
                if args.grep:
                    # Check if all grep terms are present in either the path or filename
                    if not all(g in full_path for g in args.grep):
                        continue
                eval_paths.append(full_path)
    eval_paths.sort()

    console = Console()
    if not eval_paths:
        console.print(f"[red]No eval_results*.json files found under {base_path} with grep filters: {args.grep}[/red]")
        exit(1)

    console.print(Panel("\n".join(eval_paths), title="Evaluation File Paths", style="cyan"))
    
    # Define group filters
    group_all = lambda targets: True
    group_policy = lambda targets: targets is not None and "policy" in targets
    group_decoder = lambda targets: targets is not None and "decoder" in targets

    # Process metrics for each evaluation file (for ALL, Policy, and Decoder groups)
    for path in eval_paths:
        try:
            console.print(Panel(f"Processing file: {path}", style="magenta"))
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Compute metrics for each group
            per_task_all, overall_all = auto_select_compute_metrics(data, group_filter=group_all)
            per_task_policy, overall_policy = auto_select_compute_metrics(data, group_filter=group_policy)
            # For the Decoder group, use the FWT from the ALL group as the base.
            base_fwt_dict = {task: fwt for task, (fwt, _, _) in per_task_all.items()}
            per_task_decoder, overall_decoder = auto_select_compute_metrics(data, group_filter=group_decoder, base_fwt=base_fwt_dict)
            
            # Compute "Last" metric only for the ALL group
            tasks_data_all = build_tasks_data(data, group_filter=group_all)
            per_task_last, overall_last = calculate_last_metric(tasks_data_all)

            # Build summary table (add Last column for ALL; use '-' for others)
            metric_table = Table(title="Task-level Metrics (FWT, BWT, AUC, Last)")
            metric_table.add_column("Group", style="cyan")
            metric_table.add_column("FWT", justify="right", style="magenta")
            metric_table.add_column("BWT", justify="right", style="magenta")
            metric_table.add_column("AUC", justify="right", style="magenta")
            metric_table.add_column("Last", justify="right", style="magenta")
            
            metric_table.add_row("Overall", f"{overall_all[0]:.3f}", f"{overall_all[1]:.3f}", f"{overall_all[2]:.3f}", f"{overall_last:.3f}")
            metric_table.add_row("FwSC", f"{overall_policy[0]:.3f}", f"{overall_policy[1]:.3f}", f"{overall_policy[2]:.3f}", "-")
            metric_table.add_row("BwSC", f"{overall_decoder[0]:.3f}", f"{overall_decoder[1]:.3f}", f"{overall_decoder[2]:.3f}", "-")
            
            console.print(metric_table)

            for key in per_task_all.keys():
                if key not in per_task_decoder:
                    console.print(f"[yellow]Warning: Task {key} is missing in Decoder group metrics.[/yellow]")
            logger.info(f"Tasks in ALL group: {len(per_task_all)}")
            logger.info(f"Tasks in Policy group: {len(per_task_policy)}")
            logger.info(f"Tasks in Decoder group: {len(per_task_decoder)}")
            
            if args.detailed_auc:
                # Detailed tables for each group; include "Last" column only for ALL group.
                for group_name, group_filter, extra_param in [
                    ("ALL", group_all, None),
                    ("Policy", group_policy, None),
                    ("Decoder", group_decoder, base_fwt_dict)
                ]:
                    per_task, _ = auto_select_compute_metrics(data, group_filter=group_filter, base_fwt=extra_param)
                    if not per_task:
                        console.print(f"[yellow]No tasks for group {group_name}[/yellow]")
                        continue
                    if group_name == "ALL":
                        # For ALL, get the detailed Last values from tasks_data_all.
                        _, overall_last_detailed = calculate_last_metric(build_tasks_data(data, group_filter=group_all))
                        task_table = Table(title=f"Detailed Metrics for Group: {group_name}")
                        task_table.add_column("Task", style="yellow")
                        task_table.add_column("FWT", justify="right")
                        task_table.add_column("BWT", justify="right")
                        task_table.add_column("AUC", justify="right")
                        task_table.add_column("Last", justify="right")
                        for task, (fwt, bwt, auc) in sorted(per_task.items(), key=lambda x: x[0]):
                            last_val = per_task_last.get(task, None)
                            last_str = f"{last_val:.3f}" if last_val is not None else "-"
                            task_table.add_row(task, f"{fwt:.3f}", f"{bwt:.3f}", f"{auc:.3f}", last_str)
                    else:
                        task_table = Table(title=f"Detailed Metrics for Group: {group_name}")
                        task_table.add_column("Task", style="yellow")
                        task_table.add_column("FWT", justify="right")
                        task_table.add_column("BWT", justify="right")
                        task_table.add_column("AUC", justify="right")
                        for task, (fwt, bwt, auc) in sorted(per_task.items(), key=lambda x: x[0]):
                            task_table.add_row(task, f"{fwt:.3f}", f"{bwt:.3f}", f"{auc:.3f}")
                    console.print(task_table)
                    
                    if group_name == "Decoder" and extra_param is not None:
                        missing_tasks = [task for task in per_task if task not in extra_param]
                        if missing_tasks:
                            console.print(f"[yellow]Note: The following tasks are not present in ALL/Policy (fallback applied): {', '.join(sorted(missing_tasks))}[/yellow]")
            
            # Append summary to file including Last (only for ALL group)
            output_lines = []
            output_lines.append(f"File: {path}\n")
            output_lines.append("Group\tFWT\tBWT\tAUC\tLast\n")
            output_lines.append(f"ALL\t{overall_all[0]:.3f}\t{overall_all[1]:.3f}\t{overall_all[2]:.3f}\t{overall_last:.3f}\n")
            output_lines.append(f"Policy\t{overall_policy[0]:.3f}\t{overall_policy[1]:.3f}\t{overall_policy[2]:.3f}\t-\n")
            output_lines.append(f"Decoder\t{overall_decoder[0]:.3f}\t{overall_decoder[1]:.3f}\t{overall_decoder[2]:.3f}\t-\n")
            output_lines.append("\n")
            output_str = "".join(output_lines)
            with open("data/metric_summary.txt", "a") as out_f:
                out_f.write(output_str)
            
        except Exception as e:
            console.print(f"[red]Error processing file {path}: {e}[/red]")
