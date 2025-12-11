# SILGym Setup Guide

Complete end-to-end installation guide for SILGym environment.

## Quick Start

### Default Installation (Python 3.12.12 + cuML)

```bash
bash setup.sh
```

This automated script will:
1. Create `silgym12` conda environment with Python 3.12.12
2. Install Python dependencies from `setup/python12/requirements.txt`
3. Install SILGym package in editable mode
4. Install RAPIDS cuML with auto-detected CUDA version
5. Verify installation
6. Run GPU check to ensure everything works

### Legacy Installation (Python 3.10.16)

```bash
bash setup.sh --legacy
```

This will:
1. Create `silgym` conda environment with Python 3.10.16
2. Install Python dependencies from `requirements.txt`
3. Install SILGym package in editable mode
4. Verify installation
5. Run GPU check

### Silent Installation

Skip all confirmation prompts:

```bash
bash setup.sh -y
```

## Installation Steps Explained

### Step 1: Environment Creation
- **Default**: Creates `silgym12` with Python 3.12.12
- **Legacy**: Creates `silgym` with Python 3.10.16

### Step 2: Python Dependencies
Installs core packages:
- JAX (0.8.0 for default, 0.4.34 for legacy)
- Flax (0.10.7 for default, 0.10.2 for legacy)
- PyTorch 2.8.0
- Transformers 4.57.0
- scikit-learn, umap-learn, matplotlib
- wandb, tqdm, einops
- rich (for beautiful terminal output)

### Step 3: Package Installation
Installs SILGym in editable mode using `pip install -e .`

### Step 4: cuML Installation (Default Mode Only)
- Auto-detects CUDA version from `nvidia-smi` or `nvcc`
- Auto-detects Python version from current environment
- Installs RAPIDS cuML 25.10 with matching versions
- Supports manual override: `--cuda 12.6 --python 3.10`

See `setup/python12/README.md` for cuML installation details.

### Step 5: Verification
Quick checks to ensure JAX, Flax, and cuML can be imported.

### Step 6: GPU Check
Runs comprehensive GPU availability check:
- **PyTorch**: CUDA availability and GPU operations
- **JAX**: GPU devices and computations
- **cuML**: CuPy and GPU-accelerated UMAP

If all GPUs are detected, also runs performance benchmarks comparing CPU vs GPU.

## Command Line Options

| Option | Description |
|--------|-------------|
| `--legacy` | Install legacy environment (Python 3.10.16) |
| `-y`, `--yes` | Skip all confirmation prompts |
| `-h`, `--help` | Show help message |

## Environment Comparison

| Feature | Default (`silgym12`) | Legacy (`silgym`) |
|---------|---------------------|-------------------|
| Python Version | 3.12.12 | 3.10.16 |
| JAX Version | 0.8.0 | 0.4.34 |
| Flax Version | 0.10.7 | 0.10.2 |
| qax/lorax | Not included | Included |
| cuML | Auto-installed | Not auto-installed |
| Requirements | `setup/python12/requirements.txt` | `requirements.txt` |

## Post-Installation

### Activate Environment

```bash
conda activate silgym12  # or 'silgym' for legacy
```

### MuJoCo Configuration

Add to your `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Verify Installation

Run the smoke tests:

```bash
pytest devtest -k smoke
```

### Run GPU Check Manually

```bash
python setup/gpu_check.py
```

## Troubleshooting

### Environment Already Exists

The script will ask if you want to remove and recreate it:
```bash
Warning: Conda environment 'silgym12' already exists
Do you want to remove and recreate it? (y/n)
```

Use `-y` flag to automatically remove and recreate.

### CUDA Not Detected

If cuML installation fails to detect CUDA:
```bash
# Manually specify CUDA version during cuML installation
# (the setup.sh will pass through to cuml.sh)
```

Or run cuML installation separately:
```bash
conda activate silgym12
bash setup/python12/cuml.sh --cuda 12.8
```

### GPU Check Fails

If some libraries show "CPU Only":
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Check LD_LIBRARY_PATH settings
4. Try reinstalling the specific library

### Import Errors

If you get import errors after installation:
```bash
# Ensure you're in the correct environment
conda activate silgym12

# Try reinstalling the package
pip install -e .
```

### cuML Import Error (scikit-learn compatibility)

If you see an error like:
```
AttributeError: type object 'BaseEstimator' has no attribute '_get_default_requests'
```

This is a compatibility issue between cuML 25.10 and scikit-learn 1.8.0+. Fix it by downgrading scikit-learn:
```bash
conda install -c conda-forge scikit-learn=1.7.2 -y
```

The cuml.sh script automatically handles this, but if you installed cuML manually, you may need to run this command.

## Directory Structure

```
SILGym/
├── setup.sh                      # Main installation script
├── setup/
│   ├── gpu_check.py              # GPU verification script
│   ├── python12/
│   │   ├── requirements.txt      # Python 3.12 dependencies
│   │   ├── cuml.sh               # cuML installation script
│   │   └── README.md             # cuML documentation
│   └── legacy/
│       └── ...                   # Legacy setup files
├── requirements.txt              # Legacy Python 3.10 dependencies
└── README.md                     # Main project documentation
```

## Advanced Usage

### Custom cuML Installation

The cuML installation script supports manual configuration:

```bash
# Auto-detect everything
bash setup/python12/cuml.sh

# Specify CUDA version
bash setup/python12/cuml.sh --cuda 12.0

# Specify both CUDA and Python versions
bash setup/python12/cuml.sh --cuda 12.6 --python 3.11

# Skip confirmation
bash setup/python12/cuml.sh -y
```

### Partial Installation

If you want to skip certain steps, you can run them manually:

```bash
# 1. Create environment
conda create -n silgym12 python=3.12.12
conda activate silgym12

# 2. Install dependencies
pip install -r setup/python12/requirements.txt

# 3. Install package
pip install -e .

# 4. Install cuML (optional)
bash setup/python12/cuml.sh

# 5. Run GPU check
python setup/gpu_check.py
```

## Getting Help

- Main documentation: `README.md`
- Development guide: `CLAUDE.md`
- Training guide: `docs/trainer_execution_guide.md`
- cuML installation: `setup/python12/README.md`

## Exit Codes

The setup script returns:
- `0`: Installation successful
- `1`: Installation failed or requirements not met
- GPU check returns `1` if not all GPUs are working (warning, not error)

---

**Note**: The default installation is recommended for new users and provides the most up-to-date packages with automatic GPU setup.
