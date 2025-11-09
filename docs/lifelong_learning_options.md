# Lifelong Learning Algorithm Options

This document provides a comprehensive guide to all available lifelong learning algorithm options in the SILGym codebase.

## Table of Contents

1. [BUDS](#1-buds-behavior-unsupervised-discovery-of-skills)
2. [PTGM](#2-ptgm-prototype-based-task-generation-and-matching)
3. [IsCiL](#3-iscil-incremental-skill-continual-learning)
4. [Imanip](#4-imanip-instruction-manipulation)
5. [LazySI / SILC](#5-lazysi--silc-lazy-skill-interface--skill-incremental-learning-with-clustering)
6. [Summary Table](#summary-table)
7. [Usage Examples](#usage-examples)

---

## 1. BUDS (Behavior Unsupervised Discovery of Skills)

BUDS uses a simple lifelong learning configuration with three main strategies.

### Options

| Option Format | Description | Examples |
|--------------|-------------|----------|
| `ft` | Fine-tuning (default, no memory) | `ft` |
| `er[N]` or `er[N%]` | Experience Replay (keep N% of buffer) | `er`, `er10`, `er20%` |
| `append[N]` | LoRA-based expansion (N = LoRA dimension) | `append4`, `append16` |

### Details

- **Fine-tuning (`ft`)**: Standard training without memory replay
- **Experience Replay**: Maintains a buffer of previous data
  - Default: `er` → 10% buffer
  - Custom: `erN` or `erN%` → N% buffer (must be 0-100)
- **Append**: Uses LoRA (Low-Rank Adaptation) for parameter-efficient expansion
  - Default LoRA dimension: 4
  - Pool length: 10

### Examples

```bash
# Fine-tuning only
--algorithm buds --lifelong_algo ft

# Experience replay with 20% buffer
--algorithm buds --lifelong_algo er20

# LoRA expansion with dimension 8
--algorithm buds --lifelong_algo append8
```

---

## 2. PTGM (Prototype-based Task Generation and Matching)

PTGM supports cluster configuration along with lifelong learning strategies.

### Options

| Option Format | Description | Examples |
|--------------|-------------|----------|
| `[prefix]-{base}` | Prefix: `notsne`, `sXX`, `gXX` combinations<br>Base: `ft`, `er[N]`, `append[N]` | `s20-ft`, `notsne-er10`, `s20g40-append4` |

### Prefix Options

- **`notsne`**: Disable t-SNE dimensionality reduction
- **`sXX`**: Set number of clusters (e.g., `s20` = 20 clusters)
- **`gXX`**: Set number of groups (e.g., `g40` = 40 groups)

### Base Options

Same as BUDS: `ft`, `er[N]`, `append[N]`

### Examples

```bash
# 20 clusters with fine-tuning
--algorithm ptgm --lifelong_algo s20-ft

# No t-SNE, 30 clusters, 10% experience replay
--algorithm ptgm --lifelong_algo notsnes30-er10

# 20 clusters, 40 groups, LoRA dimension 4
--algorithm ptgm --lifelong_algo s20g40-append4

# Default clusters with LoRA expansion
--algorithm ptgm --lifelong_algo append16
```

### Joint Training Mode

When using `--sync_type joint`, the interface cluster number is automatically set to 100.

---

## 3. IsCiL (Incremental Skill Continual Learning)

IsCiL uses a fixed LoRA-based expansion strategy with configurable prototype bases.

### Options

| Option Format | Description | Examples |
|--------------|-------------|----------|
| `bases[N]` | Number of prototype bases (default: 50) | `bases50`, `bases100`, `bases200` |

### Details

- **Always uses LoRA expansion**: Fixed at lora_dim=4, pool_length=8
- **Prototype bases**: Controls the number of basis vectors for skill prototypes
- **Semantic embeddings**: Uses instruction embeddings (512-dim for kitchen/mmworld)

### Examples

```bash
# Default configuration (50 bases)
--algorithm iscil --lifelong_algo bases50

# More bases for complex tasks
--algorithm iscil --lifelong_algo bases100

# If not specified, defaults to bases50
--algorithm iscil
```

---

## 4. Imanip (Instruction Manipulation)

Imanip is designed for instruction-based manipulation with temporal replay or LoRA expansion.

### Options

| Option Format | Description | Examples |
|--------------|-------------|----------|
| `tr[N]` or `tr[N%]` | Temporal Replay (keep N% of buffer) | `tr`, `tr10`, `tr20%` |
| `append[N]` | LoRA-based expansion (N = LoRA dimension, default: 16) | `append16`, `append32` |

### Details

- **Temporal Replay (`tr`)**: Similar to experience replay but optimized for temporal sequences
  - Default: `tr` → 10% buffer
  - `tr0` → No buffer (becomes appendable)
- **Append**: Uses LoRA with larger default dimension (16 vs 4 in other algorithms)
  - Default LoRA dimension: 16
  - Pool length: 10
- **Semantic embeddings**: Uses instruction embeddings (512-dim for kitchen/mmworld)

### Examples

```bash
# Temporal replay with 15% buffer
--algorithm imanip --lifelong_algo tr15

# LoRA expansion with dimension 32
--algorithm imanip --lifelong_algo append32

# No buffer (LoRA only)
--algorithm imanip --lifelong_algo tr0
```

---

## 5. LazySI / SILC (Lazy Skill Interface / Skill Incremental Learning with Clustering)

LazySI and SILC use the most complex configuration format, allowing independent control of decoder and policy components.

### Format

```
[algo_mode]/decoder_part/dec_conf/policy_algo/pol_conf
```

- 4 parts: Standard mode (no algo_mode prefix)
- 5 parts: With algo_mode specification

### 5.1 Algo Mode (Optional, 5-part format)

| Option | Description | Examples |
|--------|-------------|----------|
| `fewN[fracM]` | Few-shot learning (N shots, M fractions) | `few1`, `few5`, `few1frac2` |
| `confN[_chi2\|_percentile]` | Confidence threshold (N% confidence interval) | `conf99_chi2`, `conf95_percentile` |
| `zero` | Zero-shot mode (uses ZeroAgent) | `zero` |

**Details:**
- **Few-shot**: Limits training data to N shots per task
  - `fewN`: N shots with fraction=1
  - `fewNfracM`: N shots with 1/M of the data
- **Confidence**: Sets novelty detection threshold
  - `_chi2`: Chi-square threshold (default for Mahalanobis distance)
  - `_percentile`: Percentile threshold (default for Euclidean/Cosine)
  - Default confidence: 99%
- **Zero-shot**: Evaluates without any training on new tasks

### 5.2 Decoder Part

Format: `{algo}[_{ll}[_{cluster}]]`

| Component | Options | Description |
|-----------|---------|-------------|
| `algo` | `ptgm`, `buds`, `semantic` | Decoder algorithm type |
| `ll` (optional) | `ft`, `er[N]`, `append[N]` | Lifelong learning strategy |
| `cluster` (optional) | clustering options | Additional clustering config |

**Examples:**
- `ptgm` → PTGM decoder with fine-tuning
- `buds_ft` → BUDS decoder with explicit fine-tuning
- `semantic_append4` → Semantic decoder with LoRA dim 4
- `ptgm_er20` → PTGM decoder with 20% experience replay

### 5.3 Decoder Config

Configuration depends on the decoder algorithm:

#### PTGM Decoder Config

Format: `sXX[gYY][bZ]`

| Parameter | Format | Description | Default |
|-----------|--------|-------------|---------|
| Clusters | `sXX` | Number of skill clusters | Required |
| Goal offset | `gYY` | Goal state offset for prototypes | 0 |
| Bases | `bZ` | Number of prototype bases | 0 |

**Examples:**
- `s20` → 20 clusters
- `s20g10` → 20 clusters, goal offset 10
- `s20b4` → 20 clusters, 4 prototype bases
- `s20g10b4` → 20 clusters, goal offset 10, 4 bases

#### BUDS/Semantic Decoder Config

Format: `[gXX][bY]`

| Parameter | Format | Description | Default |
|-----------|--------|-------------|---------|
| Goal offset | `gXX` | Goal state offset for prototypes | 20 |
| Bases | `bY` | Number of prototype bases | 1 |

**Examples:**
- `g20b1` → Goal offset 20, 1 base
- `b4` → Default goal offset (20), 4 bases
- `g30b2` → Goal offset 30, 2 bases

### 5.4 Policy Algo

| Option | Description |
|--------|-------------|
| `ptgm` | PTGM-based task policy |
| `buds` | BUDS-based task policy |
| `instance` | Instance retrieval policy |
| `static` | Static policy (implemented as PTGM with 1 cluster) |

### 5.5 Policy Config

Uses the same format as decoder config (PTGM or BUDS format depending on policy algo).

### Distance Type (--dist_type)

Controls the distance metric for novelty detection:

| Option | Description | Default Threshold |
|--------|-------------|-------------------|
| `maha` | Mahalanobis distance | chi2 |
| `euclidean` | Euclidean distance | percentile |
| `cossim` | Cosine similarity | percentile |

### Complete Examples

```bash
# Standard: PTGM decoder (20 clusters, 4 bases) + PTGM policy
--algorithm lazysi --lifelong_algo "ptgm/s20b4/ptgm/s20b4"

# Few-shot (1 shot): PTGM decoder + Instance retrieval policy
--algorithm lazysi --lifelong_algo "few1/ptgm/s20b4/instance/g20b1"

# High confidence (99% chi2): BUDS decoder with LoRA + BUDS policy
--algorithm lazysi --lifelong_algo "conf99_chi2/buds_append4/g20b1/buds/g20b1"

# Semantic decoder with ER: PTGM policy
--algorithm lazysi --lifelong_algo "semantic_er10/g20b2/ptgm/s20g10b4"

# Zero-shot mode: PTGM decoder + static policy
--algorithm lazysi --lifelong_algo "zero/ptgm/s20b4/static/g1b1"

# With Euclidean distance
--algorithm lazysi --lifelong_algo "conf95_percentile/ptgm/s20b4/ptgm/s20b4" --dist_type euclidean

# BUDS decoder with fine-tuning, PTGM policy with LoRA
--algorithm lazysi --lifelong_algo "buds_ft/g20b1/ptgm_append4/s20b4"
```

### SILC vs LazySI

**SILC** is a refactored version of LazySI with improved modularity:
- Uses identical configuration format
- Uses `SILCAgent` and `SILCZeroAgent` classes
- Cleaner interface implementation in `src/SILGym/models/skill_interface/silc/`
- Fully compatible with LazySI configurations

```bash
# SILC with same configuration as LazySI
--algorithm silc --lifelong_algo "ptgm/s20b4/ptgm/s20b4"
```

---

## Summary Table

| Algorithm | Format Complexity | Configuration Pattern | Key Features |
|-----------|------------------|----------------------|--------------|
| **BUDS** | Simple | `{ft\|er[N]\|append[N]}` | Basic lifelong learning strategies |
| **PTGM** | Medium | `[prefix]-{base}` | Cluster configuration + lifelong strategies |
| **IsCiL** | Simple | `bases[N]` | Fixed LoRA expansion, configurable bases |
| **Imanip** | Simple | `{tr[N]\|append[N]}` | Temporal replay or LoRA expansion |
| **LazySI** | Complex | `[mode]/dec/conf/pol/conf` | Independent decoder and policy configuration |
| **SILC** | Complex | Same as LazySI | Refactored LazySI with improved modularity |

---

## Usage Examples

### Using trainer.sh Script

```bash
# BUDS with experience replay
bash exp/scripts/trainer.sh \
  --env kitchen --sc kitchenem --sy sync \
  --al buds --ll er20 \
  --gpu 0 --j 2 --start_seed 0 --num_exps 5 \
  --expid "buds_er20"

# PTGM with 30 clusters and LoRA expansion
bash exp/scripts/trainer.sh \
  --env kitchen --sc kitchenem --sy sync \
  --al ptgm --ll s30-append4 \
  --gpu 0 --j 2 --start_seed 0 --num_exps 5 \
  --expid "ptgm_s30_lora4"

# IsCiL with 100 prototype bases
bash exp/scripts/trainer.sh \
  --env kitchen --sc kitchenem --sy sync \
  --al iscil --ll bases100 \
  --gpu 0 --j 2 --start_seed 0 --num_exps 5 \
  --expid "iscil_bases100"

# Imanip with temporal replay
bash exp/scripts/trainer.sh \
  --env kitchen --sc kitchenem --sy sync \
  --al imanip --ll tr10 \
  --gpu 0 --j 2 --start_seed 0 --num_exps 5 \
  --expid "imanip_tr10"

# LazySI with full configuration
bash exp/scripts/trainer.sh \
  --env kitchen --sc kitchenem --sy sync \
  --al lazysi --ll ptgm/s20b4/ptgm/s20b4 \
  --gpu 0 --j 2 --start_seed 0 --num_exps 5 \
  --dec ddpm --dist_type maha \
  --expid "lazysi_ptgm"

# SILC with few-shot and instance retrieval
bash exp/scripts/trainer.sh \
  --env kitchen --sc kitchenem --sy sync \
  --al silc --ll few1/ptgm/s20b4/instance/g20b1 \
  --gpu 0 --j 2 --start_seed 0 --num_exps 5 \
  --dec ddpm --dist_type maha \
  --expid "silc_few1_instance"
```

### Using trainer.py Directly

```bash
# BUDS
python exp/trainer.py \
  --env kitchen --scenario_type objective --sync_type sync \
  --algorithm buds --lifelong ptgm/s20b4/ptgm/s20b4 \
  --seed 0 --do_eval --exp_id "buds_er"

# LazySI with confidence threshold
python exp/trainer.py \
  --env kitchen --scenario_type objective --sync_type sync \
  --algorithm lazysi --lifelong conf99_chi2/buds_append4/g20b1/buds/g20b1 \
  --seed 0 --do_eval --decoder_type ddpm --distance_type maha \
  --exp_id "lazysi_conf99"
```

---

## Implementation Details

### Configuration Parsing

All configurations are parsed in `src/SILGym/config/baseline_config.py`:

- **`parse_experience_replay_config(suffix)`**: Parses `er` configurations
- **`parse_append_config(suffix, default_lora_dim)`**: Parses `append` configurations
- **`parse_ptgm_config(cfg_str)`**: Parses PTGM cluster configurations
- **`parse_bases_config(cfg_str)`**: Parses BUDS/semantic configurations

### Model Appender

LoRA-based expansion is managed by `ModelAppender` classes:
- Located in `src/SILGym/models/skill_decoder/appender.py`
- Configurable via `AppendConfig(lora_dim, pool_length)`
- Three versions available (v1, v2, v3) in `APPENDER_CLASS_MAP`

### Agent Selection

Agent classes are automatically selected based on configuration:
- **LazySI**: `LazySIAgent` or `LazySIZeroAgent` (based on `algo_mode`)
- **SILC**: `SILCAgent` or `SILCZeroAgent` (based on `algo_mode`)

---

## Tips and Best Practices

1. **Start Simple**: Begin with `ft` or `er10` for initial experiments
2. **Scale Gradually**: Increase LoRA dimensions or buffer ratios as needed
3. **Match Complexity to Task**: Use simpler algorithms (BUDS, PTGM) for straightforward tasks
4. **Monitor Memory**: Experience replay increases memory usage linearly with buffer size
5. **Tune Clusters**: For PTGM/LazySI, adjust cluster numbers based on task diversity
6. **Experiment with Distance Metrics**: Try different `--dist_type` options for LazySI/SILC
7. **Use Few-shot for Data Efficiency**: `few1` or `few5` modes can significantly reduce training data requirements
8. **Check Semantic Embeddings**: Ensure correct embedding files exist for IsCiL/Imanip

---

## Related Files

- Configuration classes: `src/SILGym/config/baseline_config.py`
- Experiment configs: `src/SILGym/config/experiment_config.py`
- Model appenders: `src/SILGym/models/skill_decoder/appender.py`
- Agents: `src/SILGym/models/agent/base.py`
- Interfaces:
  - BUDS: `src/SILGym/models/skill_interface/buds/`
  - PTGM: `src/SILGym/models/skill_interface/ptgm/`
  - LazySI: `src/SILGym/models/skill_interface/lazySI/`
  - SILC: `src/SILGym/models/skill_interface/silc/`
