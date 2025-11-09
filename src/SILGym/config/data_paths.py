"""
Centralized data path configuration for SILGym.

This module provides a single source of truth for all dataset paths,
supporting environment variable overrides for flexible deployment.
"""

import os
from typing import Optional, Dict


class DataPathConfig:
    """
    Centralized configuration for all dataset paths in SILGym.

    Supports environment variable overrides:
    - SILGYM_DATA_ROOT: Base data directory (default: './data')
    - KITCHEN_VIS_MODEL: Default kitchen visual model (default: 'base')
    - LIBERO_MODEL: Default libero model (default: 'base')
    """

    # Base data directory (can be overridden with SILGYM_DATA_ROOT)
    _DATA_ROOT = os.environ.get("SILGYM_DATA_ROOT", "./data")

    # ========================================================================
    # Kitchen Domain Paths
    # ========================================================================

    # Kitchen task dataset path (evolving kitchen raw data)
    KITCHEN_DATASET_PATH = os.path.join(_DATA_ROOT, "evolving_kitchen", "raw")

    # Kitchen skill segments for incremental learning
    KITCHEN_SKILL_SEGMENTS_PATH = os.path.join(_DATA_ROOT, "evolving_kitchen", "skill_segments")

    # Kitchen visual embedding dataset roots
    KITCHEN_VIS_ROOT = os.path.join(_DATA_ROOT, "kitchen_lerobot_embed")
    KITCHEN_STUDIO_ROOT = os.path.join(_DATA_ROOT, "kitchenstudio_embed")

    # Default kitchen visual model
    DEFAULT_KITCHEN_VIS_MODEL = os.environ.get("KITCHEN_VIS_MODEL", "base")

    # ========================================================================
    # Metaworld (MMWorld) Domain Paths
    # ========================================================================

    # Metaworld task dataset path
    MMWORLD_DATASET_PATH = os.path.join(_DATA_ROOT, "evolving_world", "raw", "easy")

    # Metaworld skill dataset path
    MMWORLD_SKILL_PATH = os.path.join(_DATA_ROOT, "evolving_world", "raw_skill", "easy")

    # ========================================================================
    # Libero Domain Paths
    # ========================================================================

    # Libero embedding dataset root
    LIBERO_EMBED_ROOT = os.path.join(_DATA_ROOT, "libero_embed")

    # Default libero model
    DEFAULT_LIBERO_MODEL = os.environ.get("LIBERO_MODEL", "base")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    @classmethod
    def get_kitchen_vis_dataset_root(cls, model_name: Optional[str] = None) -> str:
        """
        Get the kitchen visual embedding dataset root for a specific model.

        Args:
            model_name: Model variant ('base', 'large', 'smallplus').
                       If None, uses DEFAULT_KITCHEN_VIS_MODEL.

        Returns:
            Absolute path to the model's raw dataset directory.

        Raises:
            ValueError: If model_name is not a valid variant.
        """
        from SILGym.config.kitchen_scenario import KITCHEN_VIS_MODEL_EMBEDDINGS

        if model_name is None:
            model_name = cls.DEFAULT_KITCHEN_VIS_MODEL

        model_name = model_name.lower()
        if model_name not in KITCHEN_VIS_MODEL_EMBEDDINGS:
            raise ValueError(
                f"Unknown kitchen visual embedding model '{model_name}'. "
                f"Expected one of {list(KITCHEN_VIS_MODEL_EMBEDDINGS.keys())}."
            )

        return os.path.join(cls.KITCHEN_VIS_ROOT, model_name, "raw")

    @classmethod
    def get_kitchenstudio_vis_dataset_root(cls, model_name: Optional[str] = None) -> str:
        """
        Get the kitchen studio visual embedding dataset root for a specific model.

        Args:
            model_name: Model variant ('base', 'large', 'smallplus').
                       If None, uses DEFAULT_KITCHEN_VIS_MODEL.

        Returns:
            Absolute path to the model's raw dataset directory.

        Raises:
            ValueError: If model_name is not a valid variant.
        """
        from SILGym.config.kitchen_scenario import KITCHEN_VIS_MODEL_EMBEDDINGS

        if model_name is None:
            model_name = cls.DEFAULT_KITCHEN_VIS_MODEL

        model_name = model_name.lower()
        if model_name not in KITCHEN_VIS_MODEL_EMBEDDINGS:
            raise ValueError(
                f"Unknown kitchenstudio visual embedding model '{model_name}'. "
                f"Expected one of {list(KITCHEN_VIS_MODEL_EMBEDDINGS.keys())}."
            )

        return os.path.join(cls.KITCHEN_STUDIO_ROOT, model_name, "raw")

    @classmethod
    def get_libero_skill_dataset_path(cls, model_name: Optional[str] = None) -> str:
        """
        Get the libero skill dataset path for a specific model.

        Args:
            model_name: Model name ('base', 'large', 'smallplus').
                       If None, uses DEFAULT_LIBERO_MODEL.

        Returns:
            Path to the model's dataset directory.

        Raises:
            ValueError: If model_name is not a valid variant.
        """
        from SILGym.config.libero_scenario import LIBERO_MODEL_EMBEDDINGS

        if model_name is None:
            model_name = cls.DEFAULT_LIBERO_MODEL

        if model_name not in LIBERO_MODEL_EMBEDDINGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Must be one of: {list(LIBERO_MODEL_EMBEDDINGS.keys())}"
            )

        return os.path.join(cls.LIBERO_EMBED_ROOT, model_name)

    @classmethod
    def get_paths_for_env(cls, env: str) -> Dict[str, str]:
        """
        Get all relevant paths for a specific environment.

        Args:
            env: Environment name ('kitchen', 'mmworld', 'libero').

        Returns:
            Dictionary mapping path keys to their values.

        Raises:
            ValueError: If env is not recognized.
        """
        env = env.lower()

        if env == "kitchen":
            return {
                "dataset_path": cls.KITCHEN_DATASET_PATH,
                "skill_segments_path": cls.KITCHEN_SKILL_SEGMENTS_PATH,
                "vis_root": cls.KITCHEN_VIS_ROOT,
                "studio_root": cls.KITCHEN_STUDIO_ROOT,
            }
        elif env in ("mmworld", "metaworld"):
            return {
                "dataset_path": cls.MMWORLD_DATASET_PATH,
                "skill_path": cls.MMWORLD_SKILL_PATH,
            }
        elif env == "libero":
            return {
                "embed_root": cls.LIBERO_EMBED_ROOT,
            }
        else:
            raise ValueError(
                f"Unknown environment '{env}'. "
                f"Expected one of: kitchen, mmworld, libero"
            )

    @classmethod
    def convert_skill_path_to_embed(cls, path: str, dataset_root: str) -> str:
        """
        Convert a legacy kitchen .pkl skill path to its embedded HDF5 counterpart.

        Args:
            path: Original .pkl skill path.
            dataset_root: Root directory for embedded datasets.

        Returns:
            Path to the embedded HDF5 file.
        """
        filename = os.path.basename(path)
        stem, _ = os.path.splitext(filename)
        embed_name = stem.replace(" ", "_") + "_demo.hdf5"
        return os.path.join(dataset_root, embed_name)


# ============================================================================
# Backward Compatibility Exports
# ============================================================================
# These allow existing code to continue using simple module-level imports

# Kitchen paths
skill_dataset_path = DataPathConfig.KITCHEN_DATASET_PATH
RAW_SKILL_DATASET_PATH = DataPathConfig.KITCHEN_SKILL_SEGMENTS_PATH
KITCHEN_VIS_DATASET_ROOT = DataPathConfig.KITCHEN_VIS_ROOT
KITCHEN_STUDIO_DATASET_ROOT = DataPathConfig.KITCHEN_STUDIO_ROOT
DEFAULT_KITCHEN_VIS_MODEL = DataPathConfig.DEFAULT_KITCHEN_VIS_MODEL

# MMWorld paths
MMWORLD_DATASET_PATH = DataPathConfig.MMWORLD_DATASET_PATH
MMWORLD_SKILL_DATASET_PATH = DataPathConfig.MMWORLD_SKILL_PATH

# Libero paths
LIBERO_SKILL_DATASET_PATH = DataPathConfig.LIBERO_EMBED_ROOT
DEFAULT_LIBERO_MODEL = DataPathConfig.DEFAULT_LIBERO_MODEL

# Helper functions
get_kitchen_vis_dataset_root = DataPathConfig.get_kitchen_vis_dataset_root
get_kitchenstudio_vis_dataset_root = DataPathConfig.get_kitchenstudio_vis_dataset_root
get_libero_skill_dataset_path = DataPathConfig.get_libero_skill_dataset_path
convert_skill_path_to_embed = DataPathConfig.convert_skill_path_to_embed
