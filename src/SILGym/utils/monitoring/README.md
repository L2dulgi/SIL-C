# Monitoring Utilities

Standalone scripts for post-run inspection, metric aggregation, and checkpoint introspection live in this folder. Each module can be launched directly with `python` and is designed to work against artifacts produced by the SILGym training stack.

| Script | Purpose | Primary Output |
| --- | --- | --- |
| `inspect_dataset.py` | Visualise dataset embeddings per task via t-SNE. | PNG scatter plots + task statistics. |
| `lazysi_prototype_analyzer.py` | Summarise LazySI prototypes and evaluation traces. | JSON/CSV reports, optional console tables. |
| `llmetrics.py` | Compute FWT/BWT/AUC metrics from evaluation logs. | Rich tables per run. |
| `model_profiler.py` | Profile decoder/policy checkpoints for params & FLOPs. | Rich tables + summary. |

## Quick Usage

### Dataset Inspection (`inspect_dataset.py`)

Generates t-SNE plots from LeRobot-style HDF5 demos, grouping points by task label.

```bash
python src/SILGym/utils/monitoring/inspect_dataset.py \
  data/kitchen_lerobot_embed/base/raw \
  --obs-key aggregate \
  --max-samples-per-task 1500 \
  --perplexity 35
```

Key notes:
- Observation specs accept single keys (`agentview_rgb_dinov3`), custom combos (`key1+key2` or comma-separated), or the built-ins:
  - `aggregate` (default): prioritized list matching the training dataloader (vision embeddings, then proprio).
  - `vision`: only the camera embeddings/raw pixels.
  - `proprio`: joint, end-effector, gripper, robot state channels.
- Outputs land in `<path>/tsne_plots/…_tsne.png` when a directory is provided, or alongside a single file.

### LazySI Prototype Analyzer (`lazysi_prototype_analyzer.py`)

Aggregates decoder/policy prototype statistics for LazySI experiments.

```bash
python src/SILGym/utils/monitoring/lazysi_prototype_analyzer.py \
  --log-dir logs/kitchen/.../1004appenderv3seed3maha \
  --top-k 10
```

Expect JSON (`prototype_report.json`) and CSV (`prototype_prototypes.csv`) summaries inside the log directory, plus optional Rich tables unless `--quiet` is set.

### Log Metric Aggregator (`llmetrics.py`)

Matches the canonical log layout (`logs/<env>/…/eval_results*.json`) and computes per-task / policy metrics.

```bash
python src/SILGym/utils/monitoring/llmetrics.py \
  -e kitchen \
  -g appenderv3 seed3 \
  --unit_trace
```

The script prints tables for all tasks, policy-only, and decoder-only groupings, including FWT/BWT/AUC and optional per-phase traces with `--unit_trace`.

### Model Profiler (`model_profiler.py`)

Reports parameter counts, LoRA ranks/pool sizes (when present), and forward-path FLOPs for decoder/policy checkpoints without needing GPUs (enforces `JAX_PLATFORMS=cpu`).

```bash
python src/SILGym/utils/monitoring/model_profiler.py \
  -e kitchen \
  --type decoder \
  --grep appenderv3 \
  --limit 3
```

Each checkpoint yields a Rich table plus an aggregate summary over the selection.

---

## Appendix: CLI Reference

### `inspect_dataset.py`

- `dataset_path` (positional): HDF5 file or directory containing demos.
- `-o, --output-dir`: Override output directory (default: `<path>/tsne_plots` or `<file_parent>/tsne_plots`).
- `--obs-key`: Observation spec (`aggregate`, `vision`, `proprio`, single key, or combination). **Default:** `aggregate`.
- `--max-samples-per-task`: Cap samples per task before t-SNE (0 = no limit). **Default:** `2000`.
- `--perplexity`: Target t-SNE perplexity (auto-clamped for small sample counts). **Default:** `30.0`.
- `--seed`: RNG seed for subsampling and t-SNE init. **Default:** `42`.

### `lazysi_prototype_analyzer.py`

- `--log-dir`: Experiment directory containing `eval_results*.json` and `skills/`, `policy/`. **Required**.
- `--output`: JSON report filename stored in the log dir. **Default:** `prototype_report.json`.
- `--csv-output`: CSV export filename. **Default:** `prototype_prototypes.csv`.
- `--top-k`: Number of decoder/policy entries to highlight. **Default:** `5`.
- `--quiet`: Suppress console output (only files written). **Flag**.

### `llmetrics.py`

- `-al, --algo`: Algorithm tag used in output headers. **Default:** `ptgm`.
- `-e, --env`: Environment subdirectory under `logs/`. **Default:** `kitchen`.
- `-p, --path`: Explicit path override (skips `logs/<env>`). **Default:** `none`.
- `-g, --grep`: Sequence of substrings that must appear in file paths. **Default:** `[]`.
- `-i, --exp_id`: Identifier to display in tables. **Default:** `DEF`.
- `-u, --unit_trace`: Show per-phase unit traces including decoder phases. **Flag**.
- `--detailed_auc`: Include per-task AUC breakdown tables. **Flag**.

### `model_profiler.py`

- `-e, --env`: Environment subdirectory under `logs/`. **Default:** `kitchen`.
- `-p, --path`: Explicit path override. **Default:** `none`.
- `-g, --grep`: Lower-case substrings required in checkpoint paths. **Default:** `[]`.
- `-t, --type`: Restrict to `decoder`, `policy`, or `all`. **Default:** `all`.
- `--limit`: Stop after N checkpoints. **Default:** unrestricted.

All profilers log missing/malformed artifacts and continue with remaining entries, making them safe to run across large log trees.
