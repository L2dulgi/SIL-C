# SIL-C (Skill Incremental Learning with Clustering) - Refactored Components

This package contains the refactored components from `lazySI.py`, providing a cleaner, more modular implementation while preserving the original functionality.

## Overview

The refactoring splits the monolithic `lazySI.py` into several focused modules organized in the `silc/` package:

### 1. `core.py` - Core Components
- **Distance Metrics**: Pluggable distance metric implementations (Mahalanobis, Euclidean, Cosine)
- **Threshold Validators**: Chi-square and percentile-based validation strategies
- **Prototype Classes**: Enhanced prototype implementation with configurable metrics and validators
- **Data Structures**: `SkillPrototype` and `PolicyPrototype` with factory methods
- **PrototypeFactory**: Centralized prototype creation with optimal K selection

### 2. `clustering.py` - Clustering Strategies
- **Base Classes**: Abstract clustering strategy and configuration classes
- **Implementations**:
  - `PTGMClustering`: t-SNE + KMeans clustering
  - `BUDSClustering`: Hierarchical segmentation with spectral clustering
  - `InstanceRetrievalClustering`: Instance-based retrieval (no clustering)
  - `SemanticClustering`: Pre-defined skill labels with semantic embeddings
- **ClusteringStrategyFactory**: Factory for creating clustering strategies

### 3. `interface.py` - Main Interface
- **SILCInterface**: Refactored main interface with cleaner separation of concerns
- **DataLoaderManager**: Handles dataloader operations and transformations
- **SkillManager**: Manages skill entries and prototypes
- **SILCInterfaceConfig**: Enhanced configuration with post-initialization

### 4. `test_silc.py` - Comprehensive Tests
- Unit tests for all refactored components
- Integration tests ensuring compatibility with original code
- Mock data structures for testing

### 5. `example.py` - Usage Examples
- Basic usage example
- Comparison with original interface
- Configuration examples

## Key Improvements

1. **Modularity**: Clear separation between distance metrics, clustering, and interface logic
2. **Type Hints**: Full type annotations for better IDE support and documentation
3. **Pluggable Architecture**: Easy to add new distance metrics or clustering strategies
4. **Factory Pattern**: Centralized creation of strategies and prototypes
5. **Configuration**: Dataclass-based configuration with validation
6. **Testing**: Comprehensive test suite ensuring correctness

## Usage

The refactored components maintain API compatibility with the original `lazySI.py`:

```python
from SILGym.models.skill_interface.silc import SILCInterface, SILCInterfaceConfig
from SILGym.models.skill_interface.ptgm import PTGMInterfaceConfig

# Configure clustering
ptgm_config = PTGMInterfaceConfig(
    cluster_num=20,
    goal_offset=40,
    tsne_dim=3
)

# Create interface configuration
config = SILCInterfaceConfig(
    decoder_algo="ptgm",
    decoder_algo_config=ptgm_config,
    skill_prototype_bases=5,
    policy_algo="ptgm",
    policy_algo_config=ptgm_config,
    subtask_prototype_bases=5,
    confidence_interval=0.99,
    threshold_type="chi2",
    distance_type="maha"
)

# Create interface
interface = SILCInterface(config)

# Use as before
dataloader = interface.update_interface(dataloader)
(skill_ids, decoder_ids), skill_aux = interface.forward(entry, current_state)
```

## Migration Guide

To migrate from `lazySI.py` to the refactored components:

1. Replace `LazySIInterface` with `SILCInterface`
2. Replace `LazySIInterfaceConfig` with `SILCInterfaceConfig`
3. Update imports to use the new module structure
4. All other APIs remain the same

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SILCInterface                          │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ DataLoaderMgr   │  │ SkillManager │  │ Configuration │  │
│  └─────────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Clustering Strategies                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │   PTGM   │  │   BUDS   │  │ Instance │  │  Semantic  │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Components                          │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │  Prototypes │  │  Distances  │  │    Validators     │   │
│  └─────────────┘  └─────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Future Enhancements

1. Add more distance metrics (e.g., Wasserstein, KL divergence)
2. Implement online clustering strategies
3. Add visualization tools for prototypes and clusters
4. Support for hierarchical skill structures
5. GPU acceleration for distance computations