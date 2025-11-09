import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


console = Console()

OBS_PRIORITY = [
    "agentview_rgb_dinov3",
    "eye_in_hand_rgb_dinov3",
    "agentview_rgb",
    "eye_in_hand_rgb",
    "joint_states",
    "ee_states",
    "gripper_states",
    "robot_states",
    "state",
    "proprio",
]

OBS_COMBOS: Dict[str, List[str]] = {
    "aggregate": OBS_PRIORITY,
    "vision": [
        "agentview_rgb_dinov3",
        "eye_in_hand_rgb_dinov3",
        "agentview_rgb",
        "eye_in_hand_rgb",
    ],
    "proprio": [
        "joint_states",
        "ee_states",
        "gripper_states",
        "robot_states",
        "state",
        "proprio",
    ],
}


def discover_files(base_path: Path) -> List[Path]:
    if base_path.is_file():
        return [base_path]
    patterns = ("*.h5", "*.hdf5", "*.hdf")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(base_path.rglob(pattern)))
    return files


def decode_labels(raw_labels: np.ndarray) -> np.ndarray:
    if raw_labels.dtype.kind in {"S", "O", "U"}:
        return np.array([str(label, "utf-8") if isinstance(label, bytes) else str(label) for label in raw_labels])
    return raw_labels.astype(str)


def _flatten_observation(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.reshape(-1, 1)
    if data.ndim > 2:
        return data.reshape(data.shape[0], -1)
    return data


def parse_obs_spec(obs_spec: str) -> Tuple[str, List[str]]:
    """Return a canonical name and list of keys to concatenate."""
    obs_spec = obs_spec.strip()
    if obs_spec in OBS_COMBOS:
        return obs_spec, OBS_COMBOS[obs_spec]

    # Support delimiter-based combinations, e.g. "key1+key2" or "key1,key2"
    if any(delim in obs_spec for delim in ("+", ",")):
        keys = [part.strip() for part in re.split(r"[+,]", obs_spec) if part.strip()]
        if keys:
            name = "combo_" + "_".join(keys)
            return name, keys

    # Default to single key lookup
    return obs_spec, [obs_spec]


def collect_embeddings(file_path: Path, obs_spec: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    embeddings: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    used_keys: List[str] = []
    canonical_name, key_candidates = parse_obs_spec(obs_spec)

    with h5py.File(file_path, "r") as h5file:
        if "data" not in h5file:
            raise KeyError("Missing 'data' group in file.")

        data_group = h5file["data"]
        for demo_name in data_group:
            demo = data_group[demo_name]
            if "obs" not in demo:
                console.log(f"[yellow]{file_path.name}: skipping {demo_name} (missing obs group).[/yellow]")
                continue

            obs_group = demo["obs"]
            aggregated_parts: List[np.ndarray] = []
            aggregated_keys: List[str] = []

            for candidate in key_candidates:
                if candidate not in obs_group:
                    continue
                obs_dataset = _flatten_observation(np.array(obs_group[candidate][:]))
                aggregated_parts.append(obs_dataset.astype(np.float32))
                aggregated_keys.append(candidate)

            if not aggregated_parts:
                if len(key_candidates) == 1:
                    console.log(
                        f"[yellow]{file_path.name}: skipping {demo_name} (missing obs/{key_candidates[0]}).[/yellow]"
                    )
                else:
                    console.log(
                        f"[yellow]{file_path.name}: skipping {demo_name} (none of keys {key_candidates} present).[/yellow]"
                    )
                continue

            obs_dataset = np.concatenate(aggregated_parts, axis=-1)
            skills_dataset = demo["skills"][:]

            if len(obs_dataset) != len(skills_dataset):
                console.log(
                    f"[yellow]{file_path.name}: length mismatch in {demo_name} "
                    f"(obs {len(obs_dataset)} vs skills {len(skills_dataset)}).[/yellow]"
                )
                count = min(len(obs_dataset), len(skills_dataset))
                obs_dataset = obs_dataset[:count]
                skills_dataset = skills_dataset[:count]

            embeddings.append(obs_dataset.astype(np.float32))
            labels.append(decode_labels(skills_dataset))
            used_keys.extend(aggregated_keys)

    if not embeddings:
        raise ValueError(f"No embeddings found for obs spec '{obs_spec}'.")

    stacked_embeddings = np.concatenate(embeddings, axis=0)
    stacked_labels = np.concatenate(labels, axis=0)
    deduplicated_keys = sorted(set(used_keys), key=lambda k: key_candidates.index(k) if k in key_candidates else 0)
    return stacked_embeddings, stacked_labels, deduplicated_keys


def sample_per_label(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_samples_per_label: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples_per_label <= 0:
        return embeddings, labels

    rng = np.random.default_rng(seed)
    selected_indices: List[int] = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) > max_samples_per_label:
            indices = rng.choice(indices, max_samples_per_label, replace=False)
        selected_indices.extend(indices.tolist())

    selected_indices.sort()
    return embeddings[selected_indices], labels[selected_indices]


def run_tsne(embeddings: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    max_perplexity = max(5.0, min(perplexity, max(2.0, (len(scaled) - 1) / 3)))

    tsne = TSNE(
        n_components=2,
        init="pca",
        perplexity=max_perplexity,
        random_state=seed,
        learning_rate="auto",
    )
    return tsne.fit_transform(scaled)


def plot_embeddings(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20")

    total_plots = len(unique_labels) + 1
    cols = min(3, total_plots)
    rows = math.ceil(total_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    # Combined view
    combined_ax = axes[0]
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        combined_ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.7,
            color=cmap(idx % cmap.N),
            label=label,
        )
    combined_ax.set_title(f"{title} - all tasks")
    combined_ax.set_xlabel("t-SNE dim 1")
    combined_ax.set_ylabel("t-SNE dim 2")
    combined_ax.legend(fontsize="small", markerscale=2, loc="best")

    # Per-task plots
    x_limits = (coords[:, 0].min(), coords[:, 0].max())
    y_limits = (coords[:, 1].min(), coords[:, 1].max())

    for axis, label in zip(axes[1:], unique_labels):
        mask = labels == label
        axis.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.8,
            color=cmap(np.where(unique_labels == label)[0][0] % cmap.N),
        )
        axis.set_title(label)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_xticks([])
        axis.set_yticks([])

    # Hide unused axes
    for axis in axes[1 + len(unique_labels) :]:
        axis.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize(labels: np.ndarray) -> Table:
    table = Table(title="Samples per task", show_lines=False)
    table.add_column("Task")
    table.add_column("Count", justify="right")
    for label in np.unique(labels):
        count = np.sum(labels == label)
        table.add_row(label, f"{count:,}")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate t-SNE visualisations of observation embeddings grouped by task."
    )
    parser.add_argument("dataset_path", type=str, help="Path to an HDF5 file or directory containing demos.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store generated figures (defaults to <dataset_path>/tsne_plots).",
    )
    parser.add_argument(
        "--obs-key",
        type=str,
        default="aggregate",
        help=(
            "Observation spec to load. Use a single key (e.g. agentview_rgb_dinov3), "
            "a comma/plus-separated list (e.g. agentview_rgb_dinov3+ee_states), "
            "or a named combo such as 'aggregate', 'vision', or 'proprio' (default: aggregate)."
        ),
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=2000,
        help="Maximum number of samples per task for t-SNE (0 disables subsampling).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity value (adjusted automatically when sample counts are low).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling and t-SNE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser()
    if not dataset_path.exists():
        console.print(f"[red]Dataset path '{dataset_path}' does not exist.[/red]")
        raise SystemExit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        base_for_output = dataset_path.parent if dataset_path.is_file() else dataset_path
        output_dir = base_for_output / "tsne_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(dataset_path)
    if not files:
        console.print(f"[red]No HDF5 files found under '{dataset_path}'.[/red]")
        raise SystemExit(1)

    console.print(f"[cyan]Found {len(files)} file(s). Generating t-SNE plots using obs spec '{args.obs_key}'.[/cyan]")

    for file_path in track(files, description="Processing datasets"):
        try:
            embeddings, labels, resolved_keys = collect_embeddings(file_path, args.obs_key)
        except Exception as exc:  # pragma: no cover - report and continue
            console.log(f"[red]{file_path.name}: failed to load embeddings ({exc}).[/red]")
            continue

        if len(np.unique(labels)) == 0:
            console.log(f"[yellow]{file_path.name}: no task labels found, skipping.[/yellow]")
            continue

        embeddings, labels = sample_per_label(
            embeddings,
            labels,
            args.max_samples_per_task,
            args.seed,
        )

        console.log(
            f"[green]{file_path.name}: {len(embeddings):,} samples across {len(np.unique(labels))} tasks.[/green]"
        )
        console.print(f"Resolved observation keys: {', '.join(resolved_keys)}")
        console.print(summarize(labels))

        if len(embeddings) < 10:
            console.log(f"[yellow]{file_path.name}: not enough samples for t-SNE, skipping.[/yellow]")
            continue

        coords = run_tsne(embeddings, args.perplexity, args.seed)

        safe_obs_name = parse_obs_spec(args.obs_key)[0]
        title = f"{file_path.stem} ({safe_obs_name})"
        output_path = output_dir / f"{file_path.stem}_{safe_obs_name}_tsne.png"
        plot_embeddings(coords, labels, title, output_path)
        console.log(f"[blue]Saved plot to {output_path}[/blue]")


if __name__ == "__main__":
    main()
