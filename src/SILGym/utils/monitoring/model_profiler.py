import os

# Force JAX to run on CPU so that deserializing checkpoints does not require GPUs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, unfreeze
from flax.traverse_util import flatten_dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    # Present in ModelAppender V1/V2.
    from SILGym.models.skill_decoder.appender import LoraWeightPool
except ImportError:  # pragma: no cover - optional dependency
    LoraWeightPool = None


BYTES_PER_PARAM_FP32 = 4
CONSOLE = Console()


@dataclass
class ProfileResult:
    path: str
    model_type: str
    model_class: str
    param_count: int
    param_memory_mb: float
    forward_flops: Optional[float]
    sampling_steps: Optional[int]
    sampling_flops: Optional[float]
    lora_param_count: Optional[int]
    lora_rank_set: Optional[Iterable[int]]
    lora_pool_size: Optional[int]
    warnings: List[str]


def human_readable(value: float, unit: str = "", base: int = 1000) -> str:
    """Pretty-print large numeric values using SI prefixes."""
    if value is None:
        return "n/a"

    abs_value = abs(value)
    suffixes = ["", "K", "M", "G", "T", "P"]
    for suffix in suffixes:
        if abs_value < base or suffix == suffixes[-1]:
            return f"{value:.3f}{suffix}{unit}"
        value /= base
        abs_value /= base
    return f"{value:.3f}{suffixes[-1]}{unit}"


def _to_mutable(tree: Any) -> Any:
    """Convert FrozenDict to mutable dictionaries recursively."""
    if isinstance(tree, FrozenDict):
        return {k: _to_mutable(v) for k, v in tree.items()}
    if isinstance(tree, dict):
        return {k: _to_mutable(v) for k, v in tree.items()}
    return tree


def count_parameters(params: Any) -> int:
    """Count scalar parameters in a nested pytree."""
    if params is None:
        return 0

    mutable = _to_mutable(params)
    if "params" in mutable and isinstance(mutable, dict) and len(mutable) == 1:
        mutable = mutable["params"]

    flat = flatten_dict(mutable, keep_empty_nodes=False)
    total = 0
    for value in flat.values():
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            total += int(np.prod(value.shape))
        elif hasattr(value, "shape") and hasattr(value, "size"):
            try:
                total += int(np.prod(tuple(int(dim) for dim in value.shape)))
            except Exception:  # pragma: no cover - defensive
                continue
        elif np.isscalar(value):
            total += 1
    return total


def extract_cost_metric(cost_analysis: Any, key: str) -> float:
    """Aggregate a metric (e.g., FLOPs) from JAX cost analysis output."""
    if cost_analysis is None:
        return 0.0
    items: Iterable[Dict[str, float]]
    if isinstance(cost_analysis, dict):
        items = [cost_analysis]
    else:
        items = [entry for entry in cost_analysis if isinstance(entry, dict)]
    total = 0.0
    for entry in items:
        value = entry.get(key)
        if value is not None:
            total += float(value)
    return total


def estimate_policy_flops(policy_obj) -> Tuple[Optional[float], Optional[str]]:
    params = policy_obj.train_state.params
    mutable = _to_mutable(params)
    if "params" in mutable:
        mutable = mutable["params"]

    flat = flatten_dict(mutable, keep_empty_nodes=False)
    input_dim = None
    for path, value in flat.items():
        if path[-1] == "kernel" and hasattr(value, "shape"):
            input_dim = int(value.shape[0])
            break

    if input_dim is None:
        return None, "Unable to infer policy input dimension."

    sample_input = jnp.ones((1, input_dim), dtype=jnp.float32)

    def forward(p, x):
        return policy_obj.model_eval.apply(p, x, rngs=None)

    try:
        compiled = jax.jit(forward).lower(policy_obj.train_state.params, sample_input).compile()
        flops = extract_cost_metric(compiled.cost_analysis(), "flops")
        return flops, None
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return None, f"FLOPs estimation failed: {exc}"


def _build_decoder_inputs(base_model) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cfg = base_model.input_config
    model_cfg = getattr(base_model, "model_config", {})
    model_kwargs = model_cfg.get("model_kwargs", {})

    default_x_shape = (1, 1, int(getattr(base_model, "out_dim", 1)))
    default_cond_last = int(model_kwargs.get("context_emb_dim", 64))
    default_cond_shape = (1, 1, default_cond_last)
    default_time_shape = (1, 1, int(getattr(base_model, "dim_time_embedding", 64)))

    x_shape = tuple(int(dim) for dim in cfg.get("x", default_x_shape))
    cond_shape = tuple(int(dim) for dim in cfg.get("cond", default_cond_shape))
    time_shape = tuple(int(dim) for dim in cfg.get("time", default_time_shape))

    x = jnp.ones(x_shape, dtype=jnp.float32)
    cond = jnp.ones(cond_shape, dtype=jnp.float32)
    t = jnp.ones(time_shape, dtype=jnp.float32)
    return x, t, cond


def estimate_decoder_flops(base_model, base_params) -> Tuple[Optional[float], Optional[str]]:
    x, t, cond = _build_decoder_inputs(base_model)

    def forward(p, x_val, t_val, cond_val):
        return base_model.model_eval.apply(p, x_val, t_val, cond_val, deterministic=True)

    try:
        compiled = jax.jit(forward).lower(base_params, x, t, cond).compile()
        flops = extract_cost_metric(compiled.cost_analysis(), "flops")
        return flops, None
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return None, f"FLOPs estimation failed: {exc}"


def compute_lora_stats(model_obj) -> Tuple[Optional[int], Optional[Iterable[int]], Optional[int]]:
    """Estimate LoRA parameter footprint if available."""
    def stats_from_weight_pool(tree) -> Tuple[int, List[int], Optional[int]]:
        mutable = _to_mutable(tree)
        if "params" in mutable:
            mutable = mutable["params"]
        flat = flatten_dict(mutable, keep_empty_nodes=False)
        total = 0
        ranks: List[int] = []
        pool_size: Optional[int] = None
        for value in flat.values():
            if LoraWeightPool is not None and isinstance(value, LoraWeightPool):
                a = np.asarray(value.a)
                b = np.asarray(value.b)
                total += int(a.size + b.size)
                ranks.append(int(a.shape[-2]))
                pool_size = int(a.shape[0])
        return total, ranks, pool_size

    def stats_from_lora_mats(lora_a, lora_b) -> Tuple[int, List[int], Optional[int]]:
        total = 0
        ranks: List[int] = []
        flat_a = flatten_dict(_to_mutable(lora_a), keep_empty_nodes=False)
        flat_b = flatten_dict(_to_mutable(lora_b), keep_empty_nodes=False)
        for path, a_val in flat_a.items():
            if a_val is None:
                continue
            a_arr = np.asarray(a_val)
            total += int(a_arr.size)
            b_val = flat_b.get(path)
            if b_val is not None:
                b_arr = np.asarray(b_val)
                total += int(b_arr.size)
                if b_arr.ndim >= 2:
                    ranks.append(int(b_arr.shape[-1]))
                elif a_arr.ndim >= 1:
                    ranks.append(int(a_arr.shape[0]))
        pool_size = None
        return total, ranks, pool_size

    if hasattr(model_obj, "lora_param_template"):
        total, ranks, pool = stats_from_weight_pool(model_obj.lora_param_template)
        return total or None, sorted(set(ranks)) or None, pool
    if hasattr(model_obj, "init_lora_params"):
        total, ranks, pool = stats_from_weight_pool(model_obj.init_lora_params)
        return total or None, sorted(set(ranks)) or None, pool
    if hasattr(model_obj, "lora_template_a") and hasattr(model_obj, "lora_template_b"):
        total, ranks, pool = stats_from_lora_mats(model_obj.lora_template_a, model_obj.lora_template_b)
        return total or None, sorted(set(ranks)) or None, pool
    return None, None, None


def infer_model_type(path: str, obj: Any, requested: str) -> str:
    if requested in {"decoder", "policy"}:
        return requested

    lower = path.lower()
    if "decoder" in lower:
        return "decoder"
    if "policy" in lower:
        return "policy"

    class_name = obj.__class__.__name__.lower()
    if "policy" in class_name:
        return "policy"
    return "decoder"


def load_decoder_components(obj: Any) -> Tuple[Any, Any]:
    if hasattr(obj, "base_model"):
        base_model = obj.base_model
        base_params = getattr(obj, "base_params", None)
        if base_params is None:
            base_params = base_model.train_state.params
        return base_model, base_params
    if hasattr(obj, "train_state"):
        return obj, obj.train_state.params
    raise ValueError("Unsupported decoder object; missing base model or train state.")


def profile_path(path: str, requested_type: str) -> ProfileResult:
    warnings: List[str] = []
    with open(path, "rb") as f:
        model_obj = cloudpickle.load(f)

    model_type = infer_model_type(path, model_obj, requested_type)

    if model_type == "policy":
        params = model_obj.train_state.params
        param_count = count_parameters(params)
        forward_flops, err = estimate_policy_flops(model_obj)
        if err:
            warnings.append(err)
        memory_mb = param_count * BYTES_PER_PARAM_FP32 / (1024 ** 2)

        return ProfileResult(
            path=path,
            model_type=model_type,
            model_class=type(model_obj).__name__,
            param_count=param_count,
            param_memory_mb=memory_mb,
            forward_flops=forward_flops,
            sampling_steps=None,
            sampling_flops=None,
            lora_param_count=None,
            lora_rank_set=None,
            lora_pool_size=None,
            warnings=warnings,
        )

    base_model, base_params = load_decoder_components(model_obj)
    param_count = count_parameters(base_params)
    forward_flops, err = estimate_decoder_flops(base_model, base_params)
    if err:
        warnings.append(err)

    sampling_steps = getattr(base_model, "num_sampling_steps", None)
    sampling_flops = forward_flops * sampling_steps if forward_flops is not None and sampling_steps is not None else None
    lora_param_count, lora_ranks, lora_pool_size = compute_lora_stats(model_obj)
    memory_mb = param_count * BYTES_PER_PARAM_FP32 / (1024 ** 2)

    return ProfileResult(
        path=path,
        model_type=model_type,
        model_class=type(base_model).__name__,
        param_count=param_count,
        param_memory_mb=memory_mb,
        forward_flops=forward_flops,
        sampling_steps=sampling_steps,
        sampling_flops=sampling_flops,
        lora_param_count=lora_param_count,
        lora_rank_set=lora_ranks,
        lora_pool_size=lora_pool_size,
        warnings=warnings,
    )


def find_model_paths(base_path: str, grep_terms: List[str], requested_type: str) -> List[str]:
    matches: List[str] = []
    for root, _, files in os.walk(base_path):
        for file_name in files:
            if not file_name.endswith(".pkl"):
                continue
            full_path = os.path.join(root, file_name)
            lower = full_path.lower()
            if requested_type == "policy" and "policy" not in lower:
                continue
            if requested_type == "decoder" and "decoder" not in lower:
                continue
            if grep_terms and not all(term in lower for term in grep_terms):
                continue
            matches.append(full_path)
    matches.sort()
    return matches


def render_result(result: ProfileResult) -> None:
    table = Table(show_header=False, title=result.path, title_style="cyan bold")
    table.add_row("Model Type", result.model_type)
    table.add_row("Model Class", result.model_class)
    table.add_row("Parameters", f"{human_readable(result.param_count, unit='')} ({result.param_count:,})")
    table.add_row("FP32 Memory", f"{result.param_memory_mb:.2f} MB")
    table.add_row("Forward FLOPs", human_readable(result.forward_flops or 0.0, unit="FLOPs") if result.forward_flops is not None else "n/a")

    if result.model_type == "decoder":
        if result.sampling_steps is not None:
            table.add_row("Sampling Steps", str(result.sampling_steps))
        if result.sampling_flops is not None:
            table.add_row("Full Sampling FLOPs", human_readable(result.sampling_flops, unit="FLOPs"))
        if result.lora_param_count is not None:
            ranks = ", ".join(str(rank) for rank in result.lora_rank_set) if result.lora_rank_set else "n/a"
            table.add_row("LoRA Params", f"{human_readable(result.lora_param_count)} ({result.lora_param_count:,})")
            table.add_row("LoRA Rank", ranks)
            if result.lora_pool_size is not None:
                table.add_row("LoRA Pool Size", str(result.lora_pool_size))

    if result.warnings:
        warning_panel = Panel("\n".join(result.warnings), title="Warnings", style="yellow")
        CONSOLE.print(warning_panel)

    CONSOLE.print(table)


def render_summary(results: List[ProfileResult]) -> None:
    if not results:
        return

    summary = Table(title="Aggregate Summary", title_style="green bold")
    summary.add_column("Type")
    summary.add_column("Models", justify="right")
    summary.add_column("Total Params", justify="right")
    summary.add_column("Forward FLOPs", justify="right")

    for model_type in ["decoder", "policy"]:
        subset = [res for res in results if res.model_type == model_type]
        if not subset:
            continue
        model_count = len(subset)
        total_params = sum(res.param_count for res in subset)
        total_flops = sum(res.forward_flops or 0.0 for res in subset)
        summary.add_row(
            model_type,
            str(model_count),
            human_readable(total_params, unit=""),
            human_readable(total_flops, unit="FLOPs"),
        )

    CONSOLE.print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile model checkpoints to report parameter counts and FLOPs."
    )
    parser.add_argument("-e", "--env", type=str, default="kitchen", help="Environment name under logs/ to inspect.")
    parser.add_argument("-p", "--path", type=str, default="none", help="Override logs root path.")
    parser.add_argument("-g", "--grep", type=str, nargs="*", default=[], help="Additional path filters that must all match.")
    parser.add_argument("-t", "--type", type=str, choices=["decoder", "policy", "all"], default="all", help="Restrict profiling to a specific model type.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = args.path if args.path != "none" else os.path.join("logs", args.env)

    if not os.path.exists(base_path):
        CONSOLE.print(f"[red]Base path '{base_path}' does not exist.[/red]")
        raise SystemExit(1)

    model_paths = find_model_paths(base_path, [term.lower() for term in args.grep], args.type)
    if not model_paths:
        CONSOLE.print(f"[red]No model checkpoints found under {base_path} matching filters {args.grep}.[/red]")
        raise SystemExit(1)

    if args.limit is not None:
        model_paths = model_paths[: args.limit]

    results: List[ProfileResult] = []
    for path in model_paths:
        try:
            result = profile_path(path, args.type)
        except Exception as exc:
            warning_panel = Panel(f"{exc}", title=f"Failed to profile {path}", style="red")
            CONSOLE.print(warning_panel)
            continue
        results.append(result)
        render_result(result)

    render_summary(results)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
