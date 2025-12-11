# RAPIDS cuML Installation Script

Dynamic installation script for RAPIDS cuML that automatically detects your system configuration.

## Features

- **Auto-detection**: Automatically detects CUDA and Python versions
- **Manual override**: Allows manual specification of versions
- **Confirmation prompt**: Shows configuration before installation
- **Environment aware**: Detects current conda environment

## Usage

### 1. Auto-detection (Recommended)

```bash
bash setup/cuda12/cuml.sh
```

The script will:
- Detect your CUDA version from `nvidia-smi` or `nvcc`
- Detect your Python version from current environment
- Show configuration and ask for confirmation

### 2. Manual Version Specification

```bash
# Specify CUDA version only
bash setup/cuda12/cuml.sh --cuda 12.6

# Specify both CUDA and Python versions
bash setup/cuda12/cuml.sh --cuda 12.6 --python 3.11

# Skip confirmation prompt
bash setup/cuda12/cuml.sh --cuda 12.8 --python 3.10 -y
```

### 3. View Help

```bash
bash setup/cuda12/cuml.sh --help
```

## Options

| Option | Description | Example |
|--------|-------------|---------|
| `--cuda VERSION` | Manually specify CUDA version | `--cuda 12.8` |
| `--python VERSION` | Manually specify Python version | `--python 3.10` |
| `-y`, `--yes` | Skip confirmation prompt | `-y` |
| `-h`, `--help` | Show help message | `--help` |

## Examples

### Example 1: Auto-detect everything
```bash
bash setup/cuda12/cuml.sh
```

Output:
```
========================================
RAPIDS cuML Installation Script
========================================

Current conda environment: silgym
Python version (detected): 3.10
CUDA version (detected): 12.8

Installation Configuration:
  - conda environment: silgym
  - Python version: 3.10
  - CUDA version: 12.8

Proceed with installation? (y/n)
```

### Example 2: Specific CUDA version for older GPU
```bash
bash setup/cuda12/cuml.sh --cuda 12.0
```

### Example 3: Full manual control with auto-confirm
```bash
bash setup/cuda12/cuml.sh --cuda 12.6 --python 3.11 -y
```

## CUDA Version Detection

The script attempts to detect CUDA version in the following order:

1. **nvidia-smi**: Checks CUDA driver version
2. **nvcc**: Falls back to CUDA toolkit version if available
3. **Default**: Uses 12.8 if detection fails

## Troubleshooting

### CUDA version not detected
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Check CUDA toolkit: `nvcc --version`
- Manually specify version: `--cuda 12.8`

### Python version mismatch
- Activate the correct conda environment first
- Or manually specify: `--python 3.10`

### Permission errors
- Make script executable: `chmod +x setup/cuda12/cuml.sh`

## Notes

- The script installs RAPIDS version 25.10
- Compatible with CUDA 11.x and 12.x
- Python versions 3.9, 3.10, 3.11, 3.12 supported
- Installation requires conda or mamba
- **scikit-learn compatibility**: The script automatically downgrades scikit-learn to 1.7.2 because cuML 25.10 uses `BaseEstimator._get_default_requests` which was removed in scikit-learn 1.8.0

## Related Commands

```bash
# Check current configuration
conda info --envs
python --version
nvidia-smi

# After installation, verify
python -c "import cuml; print(cuml.__version__)"
```
