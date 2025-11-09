"""
Unit tests for refactored SIL-C components.

This module provides tests to ensure the refactored components work correctly
and maintain compatibility with the original implementation.
"""

import numpy as np
import pytest
from typing import Dict, Any

from SILGym.models.skill_interface.silc import (
    PrototypeConfig, Prototype, PrototypeFactory,
    SkillPrototype, PolicyPrototype,
    MahalanobisDistance, EuclideanDistance, CosineDistance,
    ClusteringStrategyFactory, PTGMClustering, BaseClusteringConfig,
    SILCInterface, SILCInterfaceConfig, DataLoaderManager, SkillManager
)


# ==============================
# Mock Data Structures
# ==============================

class MockDataLoader:
    """Mock dataloader for testing."""
    def __init__(self, n_samples: int = 100, obs_dim: int = 10):
        self.stacked_data = {
            'observations': np.random.randn(n_samples, obs_dim),
            'actions': np.random.randn(n_samples, 4),
            'terminals': np.zeros(n_samples, dtype=int),
            'skills': ['skill_' + str(i % 5) for i in range(n_samples)]
        }
        # Set some terminals
        self.stacked_data['terminals'][24] = 1
        self.stacked_data['terminals'][49] = 1
        self.stacked_data['terminals'][74] = 1
        self.stacked_data['terminals'][-1] = 1


# ==============================
# Test Distance Metrics
# ==============================

def test_distance_metrics():
    """Test different distance metric implementations."""
    # Create test data
    x = np.random.randn(5, 10)
    mean = np.random.randn(3, 10)
    variance = np.ones((3, 10)) * 0.5
    
    # Test Mahalanobis distance
    maha = MahalanobisDistance()
    maha_dist = maha.compute(x, mean, variance)
    assert maha_dist.shape == (5, 3)
    assert np.all(maha_dist >= 0)
    
    # Test Euclidean distance
    eucl = EuclideanDistance()
    eucl_dist = eucl.compute(x, mean)
    assert eucl_dist.shape == (5, 3)
    assert np.all(eucl_dist >= 0)
    
    # Test Cosine distance
    cos = CosineDistance()
    cos_dist = cos.compute(x, mean)
    assert cos_dist.shape == (5, 3)
    assert np.all(cos_dist >= 0)
    assert np.all(cos_dist <= 2)  # Cosine distance is bounded [0, 2]
    
    print("✓ Distance metrics test passed")


# ==============================
# Test Prototype Creation
# ==============================

def test_prototype_creation():
    """Test prototype creation and validation."""
    # Create test configuration
    config = PrototypeConfig(
        distance_type="maha",
        threshold_type="chi2",
        confidence_interval=0.95,
        num_bases=3
    )
    
    # Create test data
    mean = np.random.randn(3, 10)
    variance = np.ones((3, 10)) * 0.5
    
    # Create prototype
    proto = Prototype(mean, variance, config=config)
    
    # Test distance computation
    x = np.random.randn(5, 10)
    distances = proto.compute_distances(x)
    assert distances.shape == (5, 3)
    
    # Test validation
    valid, min_dist = proto.validate(x)
    assert valid.shape == (5,)
    assert min_dist.shape == (5,)
    assert len(valid) == len(min_dist)
    
    print("✓ Prototype creation test passed")


# ==============================
# Test Prototype Factory
# ==============================

def test_prototype_factory():
    """Test prototype factory functionality."""
    config = PrototypeConfig(num_bases=3)
    factory = PrototypeFactory(config)
    
    # Create test data
    data = np.random.randn(100, 10)
    
    # Test prototype creation
    centroids, variances, thresholds = factory.create_prototypes_from_data(data)
    
    assert centroids.shape == (3, 10)
    assert variances.shape == (3, 10)
    assert thresholds.shape == (3, 1)
    assert np.all(variances >= 0)
    assert np.all(thresholds >= 0)
    
    # Test instance-based creation
    inst_cent, inst_var, inst_thr = factory.create_prototypes_from_data(
        data[:10], instance_based=True
    )
    assert inst_cent.shape == (10, 10)
    assert np.array_equal(inst_cent, data[:10])
    
    print("✓ Prototype factory test passed")


# ==============================
# Test Skill and Policy Prototypes
# ==============================

def test_skill_subtask_prototypes():
    """Test SkillPrototype and PolicyPrototype creation."""
    config = PrototypeConfig()
    
    # Test data
    skill_aux = np.random.randn(10)
    state_data = (np.random.randn(3, 10), np.ones((3, 10)), np.ones((3, 1)))
    action_data = (np.random.randn(3, 4), np.ones((3, 4)), np.ones((3, 1)))
    subgoal_data = (np.random.randn(3, 10), np.ones((3, 10)), np.ones((3, 1)))
    
    # Create SkillPrototype
    skill_proto = SkillPrototype.from_data(
        skill_id=0,
        decoder_id=0,
        skill_aux=skill_aux,
        state_data=state_data,
        action_data=action_data,
        subgoal_data=subgoal_data,
        data_count=50,
        config=config
    )
    
    assert skill_proto.skill_id == 0
    assert skill_proto.decoder_id == 0
    assert np.array_equal(skill_proto.skill_aux, skill_aux)
    assert skill_proto.data_count == 50
    
    # Create PolicyPrototype
    policy_proto = PolicyPrototype.from_data(
        prototype_id=0,
        subgoal=skill_aux,
        state_data=state_data,
        data_count=30,
        config=config
    )
    
    assert policy_proto.prototype_id == 0
    assert np.array_equal(policy_proto.subgoal, skill_aux)
    assert policy_proto.data_count == 30
    
    print("✓ Skill and policy prototypes test passed")


# ==============================
# Test Clustering Strategies
# ==============================

def test_clustering_strategies():
    """Test clustering strategy factory and basic clustering."""
    # Test factory creation
    config = BaseClusteringConfig(goal_offset=20)
    
    # Create PTGM strategy
    from SILGym.models.skill_interface.ptgm import PTGMInterfaceConfig
    ptgm_config = PTGMInterfaceConfig(
        cluster_num=5,
        goal_offset=20,
        tsne_dim=2,
        tsne_perplexity=15
    )
    
    strategy = ClusteringStrategyFactory.create('ptgm', ptgm_config)
    assert isinstance(strategy, PTGMClustering)
    
    # Test on mock data
    dataloader = MockDataLoader()
    labels, centroid_map, extra = strategy.cluster(dataloader)
    
    assert len(labels) == len(dataloader.stacked_data['observations'])
    assert len(centroid_map) > 0
    assert 'subgoals' in extra
    assert 'timesteps' in extra
    
    print("✓ Clustering strategies test passed")


# ==============================
# Test DataLoader Manager
# ==============================

def test_dataloader_manager():
    """Test dataloader manager operations."""
    manager = DataLoaderManager()
    dataloader = MockDataLoader()
    
    # Test initialization
    dataloader = manager.init_entry_fields(dataloader)
    assert 'entry' in dataloader.stacked_data
    assert 'skill_id' in dataloader.stacked_data
    assert 'skill_aux' in dataloader.stacked_data
    
    # Test augmentation
    original_shape = dataloader.stacked_data['observations'].shape
    dataloader = manager.augment_observations(dataloader)
    new_shape = dataloader.stacked_data['observations'].shape
    assert new_shape[1] == original_shape[1] * 2  # Doubled due to concatenation
    assert 'orig_obs' in dataloader.stacked_data
    
    # Test rollback
    dataloader = manager.rollback_observations(dataloader)
    assert dataloader.stacked_data['observations'].shape == original_shape
    assert 'orig_obs' not in dataloader.stacked_data
    
    print("✓ DataLoader manager test passed")


# ==============================
# Test SIL-C Interface
# ==============================

def test_silc_interface():
    """Test the main SIL-C interface."""
    from SILGym.models.skill_interface.ptgm import PTGMInterfaceConfig
    
    # Create configuration
    ptgm_config = PTGMInterfaceConfig(
        cluster_num=5,
        goal_offset=20,
        tsne_dim=2,
        tsne_perplexity=15
    )
    
    config = SILCInterfaceConfig(
        decoder_algo="ptgm",
        decoder_algo_config=ptgm_config,
        skill_prototype_bases=3,
        policy_algo="ptgm",
        policy_algo_config=ptgm_config,
        subtask_prototype_bases=3,
        confidence_interval=0.95,
        threshold_type="chi2",
        distance_type="maha"
    )
    
    # Create interface
    interface = SILCInterface(config)
    
    # Test with mock data
    dataloader = MockDataLoader()
    
    # Update interface
    dataloader = interface.update_interface(dataloader)
    assert interface.num_skills > 0
    
    # Test forward pass
    entry = np.array([0])
    current_state = dataloader.stacked_data['orig_obs'][0]
    
    (skill_ids, decoder_ids), skill_aux = interface.forward(
        entry, current_state, static=True
    )
    
    assert skill_ids.shape == (1,)
    assert decoder_ids.shape == (1,)
    assert skill_aux.shape[0] == 1
    
    # Create policy prototypes
    policy_protos = interface.create_subtask_prototype(dataloader)
    assert len(policy_protos) > 0
    
    # Test rollback
    dataloader = interface.rollback_dataloader(dataloader)
    
    print("✓ SIL-C interface test passed")


# ==============================
# Integration Test
# ==============================

def test_integration():
    """Test integration between original and refactored code."""
    # This test ensures the refactored code can work alongside the original
    from SILGym.models.skill_interface.lazySI import LazySIInterface, LazySIInterfaceConfig
    from SILGym.models.skill_interface.ptgm import PTGMInterfaceConfig
    
    ptgm_config = PTGMInterfaceConfig(
        cluster_num=5,
        goal_offset=20,
        tsne_dim=2,
        tsne_perplexity=15
    )
    
    # Create original interface
    orig_config = LazySIInterfaceConfig(
        decoder_algo="ptgm",
        decoder_algo_config=ptgm_config,
        skill_prototype_bases=3,
        policy_algo="ptgm",
        policy_algo_config=ptgm_config,
        subtask_prototype_bases=3,
        confidence_interval=0.95
    )
    
    # Create refactored interface
    new_config = SILCInterfaceConfig(
        decoder_algo="ptgm",
        decoder_algo_config=ptgm_config,
        skill_prototype_bases=3,
        policy_algo="ptgm",
        policy_algo_config=ptgm_config,
        subtask_prototype_bases=3,
        confidence_interval=0.95
    )
    
    orig_interface = LazySIInterface(orig_config)
    new_interface = SILCInterface(new_config)
    
    # Test with same data
    dataloader1 = MockDataLoader(n_samples=50)
    dataloader2 = MockDataLoader(n_samples=50)
    dataloader2.stacked_data = dataloader1.stacked_data.copy()
    
    # Update both interfaces
    dataloader1 = orig_interface.update_interface(dataloader1)
    dataloader2 = new_interface.update_interface(dataloader2)
    
    # Both should have same number of skills
    assert orig_interface.num_skills == new_interface.num_skills
    
    print("✓ Integration test passed")


# ==============================
# Run All Tests
# ==============================

if __name__ == "__main__":
    print("Running SIL-C refactoring tests...\n")
    
    test_distance_metrics()
    test_prototype_creation()
    test_prototype_factory()
    test_skill_subtask_prototypes()
    test_clustering_strategies()
    test_dataloader_manager()
    test_silc_interface()
    test_integration()
    
    print("\n✅ All tests passed successfully!")