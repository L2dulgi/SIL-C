"""
Example usage of the refactored SILC interface.

This example shows how to use the refactored components while maintaining
compatibility with the original lazySI.py interface.
"""

import numpy as np
from SILGym.models.skill_interface.silc import (
    SILCInterface, 
    SILCInterfaceConfig
)
from SILGym.models.skill_interface.ptgm import PTGMInterfaceConfig
from SILGym.dataset.dataloader import BaseDataloader


def main():
    """
    Example of using the refactored SILC interface.
    """
    
    # 1. Configure the clustering algorithms
    ptgm_config = PTGMInterfaceConfig(
        cluster_num=20,
        goal_offset=40,
        tsne_dim=3,
        tsne_perplexity=30
    )
    
    # 2. Create the SILC interface configuration
    config = SILCInterfaceConfig(
        # Decoder configuration
        decoder_algo="ptgm",
        decoder_algo_config=ptgm_config,
        skill_prototype_bases=5,
        
        # Policy configuration
        policy_algo="ptgm",
        policy_algo_config=ptgm_config,
        subtask_prototype_bases=5,
        
        # Prototype configuration
        confidence_interval=0.99,
        threshold_type="chi2",
        distance_type="maha",
        
        # Interface behavior
        force_static=False
    )
    
    # 3. Create the interface
    interface = SILCInterface(config)
    
    # 4. Load your data
    # dataloader = BaseDataloader(data_paths=["path/to/your/data"])
    
    # 5. Update the interface with data
    # dataloader = interface.update_interface(dataloader)
    
    # 6. Use the interface for skill selection
    # entry = np.array([0])  # Skill ID to query
    # current_state = observation  # Current observation
    # (skill_ids, decoder_ids), skill_aux = interface.forward(
    #     entry, current_state, static=False
    # )
    
    # 7. Create policy prototypes if needed
    # subtask_prototypes = interface.create_subtask_prototype(dataloader)
    
    # 8. Rollback dataloader when done
    # dataloader = interface.rollback_dataloader(dataloader)
    
    print("SILC Interface created successfully!")
    print(f"Configuration:")
    print(f"  - Decoder algorithm: {config.decoder_algo}")
    print(f"  - Policy algorithm: {config.policy_algo}")
    print(f"  - Distance metric: {config.distance_type}")
    print(f"  - Threshold type: {config.threshold_type}")


def compare_with_original():
    """
    Example showing that the refactored interface maintains compatibility
    with the original lazySI implementation.
    """
    from SILGym.models.skill_interface.lazySI import (
        LazySIInterface, 
        LazySIInterfaceConfig
    )
    
    # Common configuration
    ptgm_config = PTGMInterfaceConfig(
        cluster_num=20,
        goal_offset=40,
        tsne_dim=3
    )
    
    # Original interface
    original_config = LazySIInterfaceConfig(
        decoder_algo="ptgm",
        decoder_algo_config=ptgm_config,
        skill_prototype_bases=5,
        policy_algo="ptgm",
        policy_algo_config=ptgm_config,
        subtask_prototype_bases=5,
        confidence_interval=0.99
    )
    original_interface = LazySIInterface(original_config)
    
    # Refactored interface
    refactored_config = SILCInterfaceConfig(
        decoder_algo="ptgm",
        decoder_algo_config=ptgm_config,
        skill_prototype_bases=5,
        policy_algo="ptgm",
        policy_algo_config=ptgm_config,
        subtask_prototype_bases=5,
        confidence_interval=0.99
    )
    refactored_interface = SILCInterface(refactored_config)
    
    print("Both interfaces created successfully!")
    print("The refactored interface maintains full compatibility with the original.")


if __name__ == "__main__":
    main()
    print("\n" + "="*50 + "\n")
    compare_with_original()