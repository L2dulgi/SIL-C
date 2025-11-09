# SILGym Trainer Documentation

Complete reference guide for `exp/trainer.py` - the main training script for skill incremental learning experiments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Environments](#supported-environments)
3. [Command-Line Arguments](#command-line-arguments)
4. [Algorithms](#algorithms)
5. [Lifelong Learning Strategies](#lifelong-learning-strategies)
6. [Decoder Types](#decoder-types)
7. [Distance Metrics](#distance-metrics)
8. [Usage Examples](#usage-examples)
9. [Output Structure](#output-structure)
10. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Quick Start

### Prerequisites

**Training environment:**
```bash
conda activate silgym
```

**Evaluation servers** (run in separate terminals):
```bash
# For Kitchen environments
conda activate kitchen_eval
python remoteEnv/kitchen_eval/kitchen_server.py  # Port 9999

# For MMWorld environments
conda activate mmworld_eval
python remoteEnv/mmworld_eval/mmworld_server.py  # Port 8888
```

### Basic Usage

```bash
python exp/trainer.py \
  --env kitchen \
  --scenario_type kitchenem \
  --sync_type sync \
  --algorithm lazysi \
  --lifelong ptgm/s20b4/ptgm/s20b4 \
  --dec ddpm \
  --seed 0
```

### Using the Training Script Wrapper

```bash
bash exp/scripts/trainer.sh \
  --env kitchen \
  --sc kitchenem \
  --sy sync \
  --al lazysi \
  --ll ptgm/s20b4/ptgm/s20b4 \
  --gpu 0 \
  --j 2 \
  --start_seed 0 \
  --num_exps 5 \
  --dec ddpm \
  --dist_type maha \
  --expid my_experiment
```

---

## Supported Environments

### 1. Kitchen (D4RL Kitchen Tasks)
**Environment:** `kitchen`
**Python Version:** 3.8.18
**State Dimension:** 60 (proprioceptive)
**Action Dimension:** 9

**Scenario Types:**
- `kitchenem` - Kitchen environment mode (standard continual learning)
- `kitchenex` - Kitchen exploration mode
- `objective` - Objective-based scenarios
- `objective_p1`, `objective_p2`, `objective_p3` - Phase-specific objectives
- `debug`, `debugmini` - Debug scenarios with reduced complexity

**Sync Type:** `sync` (only mode supported)

### 2. MMWorld
**Environment:** `mmworld`
**Python Version:** 3.10.16
**State Dimension:** 140
**Action Dimension:** 4

**Scenario Types:**
- `mmworldem` - MMWorld easy mode (standard multi-stage tasks)
- `mmworldex` - MMWorld explicit mode (tasks with explicit skill chunks)

**Sync Type:** `sync`

**Note:** The scenario types `easy` and `easy_explicit` are internal labels. When using the trainer, you must use `mmworldem` or `mmworldex`.

---

## Command-Line Arguments

### Core Arguments

#### `-d, --debug`
- **Type:** Flag
- **Default:** False
- **Description:** Enable debug mode. Sets `phase_epochs=1` for quick testing.
- **Example:** `--debug`

#### `-id, --exp_id`
- **Type:** String
- **Default:** `''` (empty)
- **Description:** Experiment ID suffix for organizing runs.
- **Example:** `--exp_id "ablation_study"`

### Environment & Scenario Configuration

#### `-e, --env`
- **Type:** String
- **Default:** `kitchen`
- **Choices:** `kitchen`, `mmworld`
- **Description:** Environment selection.
- **Example:** `--env kitchen`

#### `-sc, --scenario_type`
- **Type:** String
- **Default:** `objective`
- **Description:** Scenario type (environment-dependent, see [Supported Environments](#supported-environments)).
- **Example:** `--scenario_type kitchenem`

#### `-st, --sync_type`
- **Type:** String
- **Default:** `sync`
- **Choices:** `sync`, `async` (varies by environment)
- **Description:** Synchronization type for task presentation.
- **Example:** `--sync_type sync`

### Algorithm Configuration

#### `-al, --algorithm`
- **Type:** String
- **Default:** `ptgm`
- **Choices:** `ptgm`, `buds`, `iscil`, `imanip`, `lazysi`, `silc`
- **Description:** Skill incremental learning algorithm.
- **Example:** `--algorithm lazysi`

#### `-ll, --lifelong`
- **Type:** String
- **Default:** `append`
- **Description:** Lifelong learning strategy (format varies by algorithm).
- **Example:** `--lifelong ptgm/s20b4/ptgm/s20b4`

#### `-dec, --decoder`
- **Type:** String
- **Default:** `ddpm`
- **Choices:** `ddpm`, `diffusion` (alias), `fql`, `flow` (alias)
- **Description:** Skill decoder architecture type.
- **Example:** `--dec fql`

#### `-dt, --dist_type`
- **Type:** String
- **Default:** `maha`
- **Choices:** `maha`, `euclidean`, `cossim`
- **Description:** Distance metric type (for LazySI/SILC only).
- **Example:** `--dist_type maha`

### Training Parameters

#### `-epoch, --epoch`
- **Type:** Integer
- **Default:** `None` (uses scenario default)
- **Description:** Number of training epochs per phase.
- **Example:** `--epoch 5000`

#### `-seed, --seed`
- **Type:** Integer
- **Default:** `0`
- **Description:** Random seed for reproducibility.
- **Example:** `--seed 42`

### Evaluation Settings

#### `--do_eval` / `--no_eval`
- **Type:** Flag (mutually exclusive)
- **Default:** `--do_eval` (True)
- **Description:** Enable or disable remote evaluation after each phase.
- **Example:** `--no_eval`

#### `--eval_noise`
- **Type:** Flag
- **Default:** False
- **Description:** Enable Gaussian noise injection during evaluation.
- **Example:** `--eval_noise --eval_noise_scale 0.05`

#### `--eval_noise_scale`
- **Type:** Float
- **Default:** `0.01`
- **Description:** Scale/magnitude of Gaussian evaluation noise.
- **Example:** `--eval_noise_scale 0.05`

#### `--eval_noise_clip`
- **Type:** Float
- **Default:** `None`
- **Description:** Optional clipping range for noisy observations.
- **Example:** `--eval_noise_clip 0.1`

#### `--eval_noise_seed`
- **Type:** Integer
- **Default:** `None`
- **Description:** Random seed for evaluation noise reproducibility.
- **Example:** `--eval_noise_seed 123`

### Action Chunking

#### `--action_chunk`
- **Type:** Integer
- **Default:** `1`
- **Description:** Number of actions to predict per forward pass (1=disabled).
- **Example:** `--action_chunk 4`

#### `--action_chunk_padding`
- **Type:** String
- **Default:** `repeat_last`
- **Choices:** `repeat_last`, `zero`
- **Description:** Padding mode for action chunks at trajectory end.
- **Example:** `--action_chunk_padding zero`

---

## Algorithms

### 1. PTGM

**Description:** Uses prototype-based clustering to organize skills and guide task generation.

**Configuration Format:**
```
[ptgmplus[_birch]/][umap|notsne][s<clusters>][g<goal_offset>][b<bases>][/<lifelong>]
```

**Components:**
- `ptgmplus` - Use MiniBatchKMeans instead of standard KMeans
- `ptgmplus_birch` - Use BIRCH clustering algorithm
- `umap` - Use UMAP for dimensionality reduction (default: t-SNE)
- `notsne` - Skip dimensionality reduction
- `s<N>` - Number of clusters (e.g., `s20` = 20 clusters)
- `g<N>` - Goal offset parameter
- `b<N>` - Number of bases (e.g., `b4` = 4 bases)
- `/<lifelong>` - Lifelong learning strategy (see [Lifelong Learning Strategies](#lifelong-learning-strategies))

**Examples:**
```bash
# Standard PTGM with t-SNE, 20 clusters, append4 mode (LoRA dim 4)
--algorithm ptgm --lifelong s20b4/append4

# PTGM with UMAP and HDBSCAN, 20 clusters
--algorithm ptgm --lifelong umaps20b4/append4

# PTGM+ with MiniBatchKMeans
--algorithm ptgm --lifelong ptgmplus/s20b4/append4

# PTGM+ with BIRCH clustering
--algorithm ptgm --lifelong ptgmplus_birch/s20b4/append16
```

### 2. BUDS

**Description:** Discovers skills in an unsupervised manner through behavior clustering.

**Configuration Format:**
```
[lifelong_algo]
```

**Examples:**
```bash
# BUDS with fine-tuning
--algorithm buds --lifelong ft

# BUDS with experience replay (10%)
--algorithm buds --lifelong er10

# BUDS with append mode
--algorithm buds --lifelong append16
```

### 3. IsCiL

**Description:** Uses semantic embeddings from instruction text for skill organization.

**Configuration Format:**
```
bases[N]
```

**Components:**
- `bases<N>` - Number of basis functions (default: 50)

**Requirements:**
- Semantic embeddings must be available at: `exp/instruction_embedding/{env}/512.pkl`

**Examples:**
```bash
# IsCiL with 50 bases (default)
--algorithm iscil --lifelong bases50

# IsCiL with 100 bases
--algorithm iscil --lifelong bases100
```

### 4. Imanip

**Description:** Leverages instruction embeddings for skill learning and manipulation.

**Configuration Format:**
```
tr[N%]
```

**Components:**
- `tr<N%>` - Temporal replay with N% of previous data

**Requirements:**
- Semantic embeddings at: `exp/instruction_embedding/{env}/512.pkl`

**Examples:**
```bash
# Imanip with 10% temporal replay
--algorithm imanip --lifelong tr10
```

### 5. LazySI

**Description:** Main algorithm using lazy evaluation for skill discovery and routing.

**Configuration Format:**
```
[algo_mode]/decoder_part/dec_conf/policy_algo/pol_conf
```

**Components:**

**Algo Mode:**
- `few<N>[frac<F>]` - N-shot learning with optional fraction F
  - Examples: `few1`, `few5frac0.5`
- `conf<N>[_chi2|_percentile]` - Confidence threshold N with metric
  - Examples: `conf99_chi2`, `conf95_percentile`
- `zero` - Zero-shot mode

**Decoder Part:**
- `ptgm` - PTGM-based decoder
- `buds` - BUDS-based decoder
- Add `_ft`, `_er`, `_append` for lifelong variants

**Decoder Config:**
- `s<N>b<M>` - N clusters, M bases
- Example: `s20b4` = 20 clusters, 4 bases

**Policy Algorithm:**
- `ptgm` - PTGM policy
- `buds` - BUDS policy
- `instance` - Instance-based retrieval
- `static` - Static policy

**Policy Config:**
- `g<N>b<M>` - N goal offset, M bases
- Example: `g20b1` = 20 goal offset, 1 base

**Examples:**
```bash
# Standard LazySI with PTGM decoder and policy
--algorithm lazysi --lifelong ptgm/s20b4/ptgm/s20b4

# Few-shot (1-shot) with instance retrieval policy
--algorithm lazysi --lifelong few1/ptgm/s20b4/instance/g20b1

# Confidence-based with chi2 threshold
--algorithm lazysi --lifelong conf99_chi2/buds/g20b1/buds/g20b1

# With experience replay in decoder
--algorithm lazysi --lifelong ptgm/ptgm_er10/s20b4/ptgm/s20b4
```

### 6. SILC

**Description:** Refactored version of LazySI with improved modularity and architecture.

**Configuration Format:** Same as LazySI (see above)

**Examples:**
```bash
# Standard SILC
--algorithm silc --lifelong ptgm/s20b4/ptgm/s20b4

# SILC with different distance metric
--algorithm silc --lifelong ptgm/s20b4/ptgm/s20b4 --dist_type euclidean
```

---

## Lifelong Learning Strategies

### Fine-tuning

#### `ft` - Standard Fine-tuning
- Continues training without maintaining memory of previous tasks
- No catastrophic forgetting prevention
- Fastest training, minimal memory overhead

#### `ftscratch` - Fine-tuning from Scratch
- Resets decoder parameters at each new phase
- Starts each task with fresh initialization
- Useful for evaluating task interference

**Examples:**
```bash
--lifelong ft
--lifelong ftscratch
```

### Experience Replay

#### `er[N%]` - Experience Replay with N% Buffer
- Maintains buffer of N% of previous task data
- Replays old data during new task training
- Helps prevent catastrophic forgetting

**Format:**
- `er` - Default 10% replay
- `er10` - 10% replay
- `er20` - 20% replay
- `er50` - 50% replay

**Examples:**
```bash
--lifelong er        # 10% default
--lifelong er20      # 20% replay
```

### Parameter Expansion (LoRA-based)

#### `append[N]` - LoRA Expansion with Dimension N
- Uses Low-Rank Adaptation (LoRA) to expand model capacity
- Adds task-specific adapters without full retraining
- Dimension N controls adapter size (typically 4-32)

**Format:**
- `append` - Default dimension (usually 4)
- `append4` - Dimension 4
- `append8` - Dimension 8
- `append16` - Dimension 16
- `append32` - Dimension 32

**Note:** Higher LoRA dimension = more capacity but slower training

**Examples:**
```bash
--lifelong append4         # LoRA dimension 4
--lifelong append16        # LoRA dimension 16
--lifelong append32        # LoRA dimension 32
```

### Pool Length Configuration

For append mode, you can specify pool length:
- Format: `append<N>_pool<M>` or `append<N>_<M>`
- `N` = LoRA dimension
- `M` = Pool length

**Examples:**
```bash
--lifelong append4_pool10   # Dimension 4, pool length 10
--lifelong append16_20      # Dimension 16, pool length 20
```

---

## Decoder Types

### DDPM (Denoising Diffusion Probabilistic Model)

**Default decoder type for all algorithms.**

**Key Characteristics:**
- Fully parameterized diffusion sampling
- Iterative denoising process
- High-quality action generation
- Slower inference than FQL

**Configuration:**
- Uses `DEFAULT_DECODER_CONFIG` from `experiment_config.py`
- Default diffusion steps: 100 (training), 20 (inference)

**Usage:**
```bash
--dec ddpm  # or --dec diffusion
```

### FQL (Flow Q-Learning)

**Flow matching decoder with Euler integration.**

**Key Characteristics:**
- Flow-based action generation
- Faster convergence than diffusion
- Euler integration for sampling
- Optional one-step distillation

**Configuration:**
- Uses `DEFAULT_FQL_DECODER_CONFIG` from `experiment_config.py`
- Default flow steps: 10
- Supports distillation to 1-step inference

**Usage:**
```bash
--dec fql  # or --dec flow
```

**Performance Notes:**
- Generally faster training than DDPM
- Can achieve comparable or better performance
- Better for real-time applications with distillation

---

## Distance Metrics

**Available for:** LazySI and SILC algorithms only

### Mahalanobis Distance (`maha`)

**Default metric.**

**Characteristics:**
- Accounts for covariance structure in data
- Scale-invariant
- Uses chi-squared (χ²) distribution for confidence thresholds
- Better for high-dimensional spaces with correlated features

**Threshold Type:** Chi-squared based
- `conf99_chi2` - 99% confidence interval
- `conf95_chi2` - 95% confidence interval

**Usage:**
```bash
--dist_type maha
--lifelong conf99_chi2/ptgm/s20b4/ptgm/s20b4
```

### Euclidean Distance (`euclidean`)

**Standard L2 distance.**

**Characteristics:**
- Simple geometric distance
- Scale-dependent
- Uses percentile-based thresholds
- Good for normalized or similar-scale features

**Threshold Type:** Percentile based
- `conf99_percentile` - 99th percentile
- `conf95_percentile` - 95th percentile

**Usage:**
```bash
--dist_type euclidean
--lifelong conf95_percentile/ptgm/s20b4/ptgm/s20b4
```

### Cosine Similarity (`cossim`)

**Angular distance between vectors.**

**Characteristics:**
- Direction-based similarity
- Magnitude-invariant
- Uses percentile-based thresholds
- Excellent for high-dimensional embeddings

**Threshold Type:** Percentile based

**Usage:**
```bash
--dist_type cossim
--lifelong conf99_percentile/ptgm/s20b4/ptgm/s20b4
```

---

## Usage Examples

### Example 1: Basic Kitchen Training with PTGM

```bash
python exp/trainer.py \
  --env kitchen \
  --scenario_type kitchenem \
  --sync_type sync \
  --algorithm ptgm \
  --lifelong s20b4/append4 \
  --dec ddpm \
  --seed 0
```

### Example 2: LazySI with DDPM Decoder

```bash
python exp/trainer.py \
  --env kitchen \
  --scenario_type kitchenem \
  --sync_type sync \
  --algorithm lazysi \
  --lifelong ptgm/s20b4/ptgm/s20b4 \
  --dec ddpm \
  --dist_type maha \
  --seed 0
```

### Example 3: MMWorld with Experience Replay

```bash
python exp/trainer.py \
  --env mmworld \
  --scenario_type mmworldem \
  --sync_type sync \
  --algorithm ptgm \
  --lifelong s20b4/er20 \
  --dec ddpm \
  --seed 0
```

### Example 4: Few-Shot Learning with Instance Retrieval

```bash
python exp/trainer.py \
  --env kitchen \
  --scenario_type kitchenem \
  --sync_type sync \
  --algorithm lazysi \
  --lifelong few1/ptgm/s20b4/instance/g20b1 \
  --dec fql \
  --dist_type maha \
  --seed 0
```

### Example 5: Confidence-Based Skill Discovery

```bash
python exp/trainer.py \
  --env kitchen \
  --scenario_type kitchenem \
  --sync_type sync \
  --algorithm lazysi \
  --lifelong conf99_chi2/buds/g20b1/buds/g20b1 \
  --dec ddpm \
  --dist_type maha \
  --seed 0
```

### Example 6: Debug Mode (Quick Testing)

```bash
python exp/trainer.py \
  --env kitchen \
  --scenario_type debug \
  --sync_type sync \
  --algorithm ptgm \
  --lifelong ft \
  --dec ddpm \
  --debug \
  --no_eval \
  --seed 0
```

### Example 7: Custom Epochs and LoRA Dimension

```bash
python exp/trainer.py \
  --env kitchen \
  --scenario_type kitchenem \
  --sync_type sync \
  --algorithm ptgm \
  --lifelong append32 \
  --dec ddpm \
  --epoch 3000 \
  --seed 0
```

### Example 8: Evaluation with Noise Injection

```bash
python exp/trainer.py \
  --env kitchen \
  --scenario_type kitchenem \
  --sync_type sync \
  --algorithm lazysi \
  --lifelong ptgm/s20b4/ptgm/s20b4 \
  --dec ddpm \
  --eval_noise \
  --eval_noise_scale 0.05 \
  --eval_noise_seed 42 \
  --seed 0
```

### Example 9: Multiple Experiments with Training Script

```bash
bash exp/scripts/trainer.sh \
  --env kitchen \
  --sc kitchenem \
  --sy sync \
  --al lazysi \
  --ll ptgm/s20b4/ptgm/s20b4 \
  --gpu 0 \
  --j 2 \
  --start_seed 0 \
  --num_exps 5 \
  --dec ddpm \
  --dist_type maha \
  --expid ablation_study
```

---

## Output Structure

### Directory Organization

Experiments are saved to:
```
logs/{env_name}/{scenario_name}/sync/{algorithm}/{lifelong_details}/{date}seed{seed}{exp_id}/
```

**Example paths:**
```
logs/kitchen/kitchenem/sync/lazysi/ptgm_s20b4_ptgm_s20b4/20250110seed0_myexp/
logs/mmworld/mmworldem/sync/ptgm/s20b4_append4/20250110seed1/
```

### File Structure

```
{experiment_directory}/
├── policy/
│   ├── policy_0/
│   │   └── pre_0.pkl              # Policy checkpoint before phase 0
│   ├── policy_1/
│   │   └── pre_1.pkl              # Policy checkpoint before phase 1
│   └── ...
├── skills/
│   ├── decoder_pre_0.pkl          # Decoder checkpoint before phase 0
│   ├── decoder_pre_1.pkl          # Decoder checkpoint before phase 1
│   ├── interface_pre_0.pkl        # Interface checkpoint before phase 0
│   ├── interface_pre_1.pkl        # Interface checkpoint before phase 1
│   └── ...
├── skill_trainer.log              # Main training log
├── config.yaml                    # Experiment configuration
└── results/
    ├── phase_0_eval.json          # Phase 0 evaluation results
    ├── phase_1_eval.json          # Phase 1 evaluation results
    └── ...
```

### Checkpoint Files

**Policy Checkpoints** (`policy/policy_{phase}/pre_{prev_phase}.pkl`):
- Task policy model state
- Trained for high-level skill selection
- Saved before each phase

**Decoder Checkpoints** (`skills/decoder_pre_{phase}.pkl`):
- Skill decoder model state (DDPM/FQL)
- Low-level action generation
- Saved before each phase

**Interface Checkpoints** (`skills/interface_pre_{phase}.pkl`):
- Skill interface model state
- Prototype/cluster information
- Skill routing logic
- Saved before each phase

---

## Tips & Troubleshooting

### GPU Acceleration

**cuML GPU Acceleration** (Optional but Recommended):
- Provides 5-50x speedup for clustering (KMeans)
- Provides 10-100x speedup for manifold learning (UMAP, t-SNE)

**Check Status:**
```bash
python exp/trainer.py --env kitchen --sc debug --debug | grep "Clustering Algorithms Backend"
```

**Enable cuML:**
```bash
bash setup/python12/cuml.sh
```

### Memory Management

**For large environments or long sequences:**
- Reduce batch size in scenario config
- Consider using `fql` decoder (faster than DDPM)

**For memory buffer overflow:**
- Reduce experience replay percentage (e.g., `er10` instead of `er20`)
- Limit append pool length

### Performance Optimization

**Faster training:**
1. Use FQL decoder: `--dec fql`
2. Reduce phase epochs: `--epoch 2500`
3. Enable cuML GPU acceleration
4. Use smaller LoRA dimension: `--lifelong append4`

**Better performance:**
1. Increase phase epochs: `--epoch 7500`
2. Use larger LoRA dimension: `--lifelong append32`
3. Increase experience replay: `--lifelong er20`
4. Use appropriate distance metric for your data

### Common Issues

#### Issue: Server Connection Timeout
**Solution:** Ensure evaluation server is running before starting training:
```bash
# In separate terminal
conda activate kitchen_eval  # or mmworld_eval
python remoteEnv/{env}_eval/{env}_server.py
```

#### Issue: CUDA Out of Memory
**Solutions:**
- Reduce batch size in scenario config
- Use smaller LoRA dimension: `--lifelong append4`
- Switch to FQL decoder (lower memory than DDPM): `--dec fql`

#### Issue: Slow Training
**Solutions:**
- Enable cuML GPU acceleration
- Use `--debug` mode for testing
- Reduce phase epochs temporarily
- Use FQL decoder instead of DDPM

#### Issue: Poor Performance
**Check:**
- Verify appropriate lifelong strategy for your scenario
- Ensure evaluation server matches environment
- Verify scenario type matches your task

### Debugging Tips

**Quick sanity check:**
```bash
python exp/trainer.py \
  --env kitchen \
  --sc debug \
  --debug \
  --no_eval \
  --algorithm ptgm \
  --lifelong ft
```

**Check logs:**
```bash
# View training progress
tail -f logs/{env}/{scenario}/sync/{algo}/.../skill_trainer.log

# Check for errors
grep -i error logs/{env}/{scenario}/sync/{algo}/.../skill_trainer.log
```

**Verify environment:**
```bash
# Check conda environment
conda list | grep -E "jax|flax|mujoco"

# Test evaluation server
curl http://localhost:9999/health  # Kitchen
curl http://localhost:8888/health  # MMWorld
```

### Best Practices

1. **Always start evaluation server first** before running training
2. **Use meaningful experiment IDs** for organization: `--exp_id "experiment_name"`
3. **Run multiple seeds** for statistical significance (use training script wrapper)
4. **Check GPU utilization** during training: `nvidia-smi -l 1`
5. **Monitor training logs** for convergence and errors
6. **Use debug mode** for quick iteration during development
7. **Save checkpoints regularly** (automatic in the trainer)
8. **Document hyperparameters** in experiment notes

### Evaluation

After training completes, evaluate results:
```bash
python src/SILGym/utils/llmetrics.py -e kitchen -g keyword1 keyword2
```

This will aggregate results across multiple runs and compute:
- Success rates per phase
- Average performance metrics
- Forgetting analysis
- Transfer learning metrics

---

## Additional Resources

- **Main Documentation:** See `CLAUDE.md` in repository root
- **Configuration Details:** See `src/SILGym/config/baseline_config.py`
- **Model Implementations:** See `src/SILGym/models/`
- **Trainer Implementation:** See `src/SILGym/trainer/skill_trainer.py`

For more information or issues, please refer to the project documentation or contact the development team.
