# SILGym Trainer Execution Guide

This guide provides comprehensive documentation for all possible trainer execution options and configurations for the SILGym framework.

## Table of Contents
1. [Basic Command Structure](#basic-command-structure)
2. [Command-Line Arguments](#command-line-arguments)
3. [Algorithm Configurations](#algorithm-configurations)
4. [Environment and Scenario Options](#environment-and-scenario-options)
5. [Lifelong Learning Algorithms](#lifelong-learning-algorithms)
6. [Complete Examples](#complete-examples)

## Basic Command Structure

The basic command to run the trainer is:

```bash
python exp/trainer.py [OPTIONS]
```

## Command-Line Arguments

### Debug and Experiment Settings

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--debug` | `-d` | flag | False | Enable debug mode (sets phase_epochs=1) |
| `--exp_id` | `-id` | string | '' | Experiment ID for tracking and organization |
| `--seed` | `-seed` | int | 0 | Random seed for reproducibility |

### Environment and Scenario Configuration

| Argument | Short | Type | Default | Options | Description |
|----------|-------|------|---------|---------|-------------|
| `--env` | `-e` | string | 'kitchen' | kitchen, mmworld | Environment to use |
| `--scenario_type` | `-sc` | string | 'objective' | See [Scenario Options](#scenario-options) | Type of scenario |
| `--sync_type` | `-st` | string | 'sync' | See [Sync Options](#sync-options) | Synchronization type |

### Algorithm Configuration

| Argument | Short | Type | Default | Options | Description |
|----------|-------|------|---------|---------|-------------|
| `--algorithm` | `-al` | string | 'ptgm' | ptgm, buds, iscil, imanip, assil, lazysi, silc | Skill incremental learning algorithm |
| `--lifelong` | `-ll` | string | 'append' | See [Lifelong Options](#lifelong-learning-algorithms) | Decoder lifelong learning strategy |

### Training Configuration

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--epoch` | `-epoch` | int | None | Number of training epochs per phase |
| `--dist_type` | `-dt` | string | 'maha' | Distance metric type: maha, euclidean, cossim |

### Evaluation Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--do_eval` | flag | True | Perform remote evaluation after each phase |
| `--no_eval` | flag | False | Skip remote evaluation |

## Algorithm Configurations

### 1. PTGM (Prototype-based Task Generation and Matching)
- **Class**: `PTGMConfig`
- **Special lifelong options**: 
  - `notsne` prefix: Disable t-SNE
  - `s<num>`: Set cluster number
  - `g<num>`: Set groups number
- **Example**: `--algorithm ptgm --lifelong s20g20append4`

### 2. BUDS (Behavior Unsupervised Discovery of Skills)
- **Class**: `BUDSConfig`
- **Lifelong options**: Standard (ft, er, append)
- **Example**: `--algorithm buds --lifelong append4`

### 3. IsCiL (Incremental Skill Continual Learning)
- **Class**: `IsCiLConfig`
- **Special lifelong options**: 
  - `bases<num>`: Set number of prototype bases (default: 50)
- **Example**: `--algorithm iscil --lifelong bases100`

### 4. Imanip (Instruction Manipulation)
- **Class**: `ImanipConfig`
- **Special lifelong options**:
  - `tr<num>`: Temporal replay with buffer ratio
- **Example**: `--algorithm imanip --lifelong tr20`

### 5. AsSIL (Adaptive Skill Sequential Incremental Learning)
- **Class**: `AsSILConfig`
- **Special lifelong format**: `[zero]s<decoder>p<policy>b<bases>[r<ranks>][g<groups>]`
  - `zero` prefix: Use zero-shot agent
  - `s<num>`: Decoder skill count
  - `p<num>`: Policy skill count
  - `b<num>`: Number of prototype bases
  - `r<num>`: Optional rank dimension
  - `g<num>`: Optional groups/bin count
- **Example**: `--algorithm assil --lifelong s20p10b3r4g20`

### 6. LazySI (Lazy Skill Interface)
- **Class**: `LazySIConfig`
- **Complex lifelong format**: `[algo_mode]/decoder_part/dec_conf/policy_algo/pol_conf`
  - **algo_mode**: `few<shot>[frac<num>]` or `conf<num>[_chi2|_percentile]`
  - **decoder_part**: `ptgm`, `buds`, or `semantic`
  - **policy_algo**: `ptgm`, `buds`, `instance`, or `static`
- **Example**: `--algorithm lazysi --lifelong conf99/ptgm/s20g20b4/ptgm/s20g20b4`

### 7. SILC (Skill Incremental Learning with Clustering)
- **Class**: `SILCConfig`
- **Same options as LazySI** (refactored version)
- **Example**: `--algorithm silc --lifelong conf99_chi2/ptgm/s20b4/instance/g20b1`

## Environment and Scenario Options

### Kitchen Environment

#### Scenario Types
- `kitchenem`: Object-based clustering and synchronization
- `kitchenex`: Explicit skill incremental learning
- `objective_p1`, `objective_p2`, `objective_p3`: Object-based with permutations
- `debug`: Debug scenario (4 tasks only)

#### Sync Types
- `sync`: Standard synchronization (only option)

### MMWorld Environment

#### Scenario Types
- `mmworldem`: Easy mode explicit skills

#### Sync Types
- `sync`: Standard synchronization (only option)

## Lifelong Learning Algorithms

### Basic Options (Available for all algorithms)

1. **Fine-tuning (ft)**
   - No replay or model expansion
   - Example: `--lifelong ft`

2. **Experience Replay (er)**
   - Format: `er<percentage>`
   - Default: 10% if no percentage specified
   - Example: `--lifelong er20` (20% buffer)

3. **Model Expansion (append)**
   - Format: `append<lora_dim>`
   - Default LoRA dimension: 4
   - Example: `--lifelong append16`

### Algorithm-Specific Options

See individual algorithm descriptions above for special lifelong learning options.

## Complete Examples

### Basic Training Examples

```bash
# 1. Kitchen environment with PTGM
python exp/trainer.py --env kitchen --scenario_type kitchenem --algorithm ptgm --lifelong append4

# 2. MMWorld with BUDS and experience replay
python exp/trainer.py --env mmworld --scenario_type mmworldem --algorithm buds --lifelong er20

# 3. Debug mode with custom epochs
python exp/trainer.py --debug --env kitchen --scenario_type debug \
    --algorithm assil --lifelong s20p10b3 --epoch 100

# 4. AsSIL with zero-shot agent
python exp/trainer.py --env kitchen --scenario_type kitchenex \
    --algorithm assil --lifelong zeros20p10b3r16g20
```

### Advanced Configuration Examples

```bash
# 1. PTGM without t-SNE and custom clustering with larger LoRA dimension
python exp/trainer.py --env kitchen --scenario_type objective_p1 \
    --algorithm ptgm --lifelong notsnes30g40append32

# 2. IsCiL with high number of bases
python exp/trainer.py --env mmworld --scenario_type mmworldem \
    --algorithm iscil --lifelong bases200 --seed 42

# 3. Imanip with temporal replay
python exp/trainer.py --env kitchen --scenario_type kitchenex \
    --algorithm imanip --lifelong tr50 --epoch 10000
```

### Environment-Specific Examples

#### Kitchen
```bash
# Object-based scenario with permutation 2
python exp/trainer.py --env kitchen --scenario_type objective_p2 \
    --algorithm ptgm --lifelong s20g20append4
```

#### MMWorld
```bash
# Easy mode with reduced epochs (n1 scenario auto-adjusts epochs)
python exp/trainer.py --env mmworld --scenario_type mmworldem \
    --algorithm buds --lifelong append16
```

## Notes and Best Practices

1. **Experiment ID**: Always set a meaningful `--exp_id` to track your experiments
2. **Seeds**: Use different seeds for multiple runs to ensure reproducibility
3. **Debug Mode**: Use `--debug` for quick testing (sets epochs to 1)
4. **Evaluation**: Disable with `--no_eval` during development to save time
5. **Distance Metrics**: 
   - `maha` (Mahalanobis) is default and recommended for most cases
   - `euclidean` and `cossim` force percentile thresholds

## Environment Setup Requirements

Before running training, ensure environment servers are running:

```bash
# Terminal 1: Start environment server
conda activate {env_name}_eval  # e.g., kitchen_eval
python remoteEnv/{env_name}/{env_name}_server.py

# Terminal 2: Run training
conda activate silgym
python exp/trainer.py [OPTIONS]
```

## Output Structure

Training results are saved to:
```
logs/{env_name}/{scenario_type}/{sync_type}/{algorithm}_{lifelong}/{date}{exp_id}seed{seed}/
├── policy/
│   └── policy_{phase}_{version}.pkl
├── skills/
│   ├── decoder_{phase}.pkl
│   └── interface_{phase}.pkl
└── experiment_config.txt
```