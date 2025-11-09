"""
Variant registry for resolving environment and model aliases.

This module provides a registry pattern for mapping user-friendly aliases
to canonical model variant names, making it easier to test and extend
variant resolution logic.
"""

from typing import Dict, List, Optional, Set
from abc import ABC, abstractmethod


class VariantRegistry(ABC):
    """
    Abstract base class for variant resolution registries.

    A registry maps user-friendly aliases (e.g., "kitchen-vis-l", "libero-base")
    to canonical model variant names (e.g., "large", "base").
    """

    def __init__(self, default_variant: str):
        """
        Initialize the registry with a default variant.

        Args:
            default_variant: The default variant to return when no alias is provided.
        """
        self._default_variant = default_variant
        self._registry: Dict[str, str] = {}

    def register(self, alias: str, variant: str) -> None:
        """
        Register an alias to a variant mapping.

        Args:
            alias: User-friendly alias (case-insensitive).
            variant: Canonical variant name.
        """
        self._registry[alias.lower()] = variant

    def register_many(self, aliases: List[str], variant: str) -> None:
        """
        Register multiple aliases to the same variant.

        Args:
            aliases: List of user-friendly aliases.
            variant: Canonical variant name to map all aliases to.
        """
        for alias in aliases:
            self.register(alias, variant)

    def resolve(self, alias: Optional[str]) -> str:
        """
        Resolve an alias to its canonical variant name.

        Args:
            alias: User-provided alias string (case-insensitive).
                  If None or empty, returns the default variant.

        Returns:
            Canonical variant name.

        Raises:
            ValueError: If alias is not recognized.
        """
        # Handle None or empty string
        normalized_alias = (alias or "").strip().lower()
        if not normalized_alias:
            return self._default_variant

        # Direct lookup
        if normalized_alias in self._registry:
            return self._registry[normalized_alias]

        # Try normalizing hyphens/underscores
        if "_" in normalized_alias:
            normalized_alias = normalized_alias.replace("_", "-")
            if normalized_alias in self._registry:
                return self._registry[normalized_alias]
        elif "-" in normalized_alias:
            normalized_alias = normalized_alias.replace("-", "_")
            if normalized_alias in self._registry:
                return self._registry[normalized_alias]

        # Alias not found
        raise ValueError(
            f"Could not resolve variant from alias '{alias}'. "
            f"Supported aliases: {self.list_aliases()}"
        )

    def list_aliases(self) -> List[str]:
        """Return a sorted list of all registered aliases."""
        return sorted(set(self._registry.keys()))

    def list_variants(self) -> List[str]:
        """Return a sorted list of all canonical variants."""
        return sorted(set(self._registry.values()))

    def get_default_variant(self) -> str:
        """Return the default variant."""
        return self._default_variant

    @abstractmethod
    def _populate_registry(self) -> None:
        """
        Populate the registry with alias mappings.

        Subclasses must implement this method to define their specific mappings.
        """
        pass


# ============================================================================
# Kitchen Visual Embedding Variant Registry
# ============================================================================

class KitchenVisVariantRegistry(VariantRegistry):
    """
    Registry for kitchen visual embedding model variants.

    Supported canonical variants:
    - "base": ViT-B/16 (768-dim embeddings)
    - "large": ViT-L/16 (1024-dim embeddings)
    - "smallplus": ViT-S+/16 (384-dim embeddings)

    Supported prefixes:
    - kitchen_vis, kitchen-vis, kitchenvis
    - kitchenstudio_vis, kitchenstudio-vis, kitchen_studio, kitchen-studio

    Supported suffixes:
    - (empty), -base, -b → "base"
    - -large, -l → "large"
    - -smallplus, -small, -s → "smallplus"
    """

    # Canonical variants and their embedding dimensions
    VARIANTS = {
        "smallplus": 384,
        "base": 768,
        "large": 1024,
    }

    # Mapping to DINOv3 model names
    VARIANT_TO_DINOV3 = {
        "smallplus": "ViT-S+/16",
        "base": "ViT-B/16",
        "large": "ViT-L/16",
    }

    def __init__(self, default_variant: str = "base"):
        """
        Initialize the kitchen visual variant registry.

        Args:
            default_variant: Default variant when no alias is provided (default: "base").
        """
        if default_variant not in self.VARIANTS:
            raise ValueError(
                f"Invalid default variant '{default_variant}'. "
                f"Must be one of: {list(self.VARIANTS.keys())}"
            )

        super().__init__(default_variant)
        self._populate_registry()

    def _populate_registry(self) -> None:
        """Populate the registry with kitchen visual embedding aliases."""
        # Define prefixes and suffix mappings
        _KITCHEN_VIS_PREFIXES = ("kitchen_vis", "kitchen-vis", "kitchenvis")
        _KITCHENSTUDIO_VIS_PREFIXES = (
            "kitchenstudio_vis",
            "kitchenstudio-vis",
            "kitchen_studio",
            "kitchen-studio",
            "kitchenstudio",
        )

        _SUFFIX_MAP = {
            "": "base",
            "-base": "base",
            "-b": "base",
            "-large": "large",
            "-l": "large",
            "-smallplus": "smallplus",
            "-small": "smallplus",
            "-s": "smallplus",
        }

        # Register all combinations of prefixes and suffixes
        for prefix in _KITCHEN_VIS_PREFIXES:
            for suffix, variant in _SUFFIX_MAP.items():
                alias = (prefix + suffix).lower()
                self.register(alias, variant)
                # Also register underscore variant
                if "-" in alias:
                    self.register(alias.replace("-", "_"), variant)

        # Register kitchenstudio aliases (same variants)
        for prefix in _KITCHENSTUDIO_VIS_PREFIXES:
            for suffix, variant in _SUFFIX_MAP.items():
                alias = (prefix + suffix).lower()
                self.register(alias, variant)
                # Also register underscore variant
                if "-" in alias:
                    self.register(alias.replace("-", "_"), variant)

    def get_embedding_dim(self, variant: str) -> int:
        """
        Get the embedding dimension for a variant.

        Args:
            variant: Canonical variant name.

        Returns:
            Embedding dimension (int).

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Must be one of: {list(self.VARIANTS.keys())}"
            )
        return self.VARIANTS[variant]

    def get_dinov3_model(self, variant: str) -> str:
        """
        Get the DINOv3 model name for a variant.

        Args:
            variant: Canonical variant name.

        Returns:
            DINOv3 model name (e.g., "ViT-B/16").

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in self.VARIANT_TO_DINOV3:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Must be one of: {list(self.VARIANT_TO_DINOV3.keys())}"
            )
        return self.VARIANT_TO_DINOV3[variant]


# ============================================================================
# Libero Model Variant Registry
# ============================================================================

class LiberoVariantRegistry(VariantRegistry):
    """
    Registry for Libero model variants.

    Supported canonical variants:
    - "base": ViT-B/16 (768-dim embeddings)
    - "large": ViT-L/16 (1024-dim embeddings)
    - "smallplus": ViT-S+/16 (384-dim embeddings)

    Supported aliases:
    - libero, libero-b, libero-base → "base"
    - libero-l, libero-large → "large"
    - libero-s, libero-small, libero-smallplus → "smallplus"
    """

    # Canonical variants and their embedding dimensions
    VARIANTS = {
        "smallplus": 384,
        "base": 768,
        "large": 1024,
    }

    # Mapping to DINOv3 model names
    VARIANT_TO_DINOV3 = {
        "smallplus": "ViT-S+/16",
        "base": "ViT-B/16",
        "large": "ViT-L/16",
    }

    def __init__(self, default_variant: str = "base"):
        """
        Initialize the libero variant registry.

        Args:
            default_variant: Default variant when no alias is provided (default: "base").
        """
        if default_variant not in self.VARIANTS:
            raise ValueError(
                f"Invalid default variant '{default_variant}'. "
                f"Must be one of: {list(self.VARIANTS.keys())}"
            )

        super().__init__(default_variant)
        self._populate_registry()

    def _populate_registry(self) -> None:
        """Populate the registry with libero model aliases."""
        # Base variant aliases
        self.register_many(
            ["libero", "libero-b", "libero-base", "libero_b", "libero_base"],
            "base"
        )

        # Large variant aliases
        self.register_many(
            ["libero-l", "libero-large", "libero_l", "libero_large"],
            "large"
        )

        # Smallplus variant aliases
        self.register_many(
            [
                "libero-s",
                "libero-small",
                "libero-smallplus",
                "libero_s",
                "libero_small",
                "libero_smallplus",
            ],
            "smallplus"
        )

    def get_embedding_dim(self, variant: str) -> int:
        """
        Get the embedding dimension for a variant.

        Args:
            variant: Canonical variant name.

        Returns:
            Embedding dimension (int).

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Must be one of: {list(self.VARIANTS.keys())}"
            )
        return self.VARIANTS[variant]

    def get_dinov3_model(self, variant: str) -> str:
        """
        Get the DINOv3 model name for a variant.

        Args:
            variant: Canonical variant name.

        Returns:
            DINOv3 model name (e.g., "ViT-B/16").

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in self.VARIANT_TO_DINOV3:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Must be one of: {list(self.VARIANT_TO_DINOV3.keys())}"
            )
        return self.VARIANT_TO_DINOV3[variant]


# ============================================================================
# Global Registry Instances
# ============================================================================

# Create singleton instances with environment variable support
import os

_kitchen_default = os.environ.get("KITCHEN_VIS_MODEL", "base")
_libero_default = os.environ.get("LIBERO_MODEL", "base")

kitchen_vis_registry = KitchenVisVariantRegistry(default_variant=_kitchen_default)
libero_registry = LiberoVariantRegistry(default_variant=_libero_default)


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def resolve_kitchen_vis_variant(env_alias: Optional[str]) -> str:
    """
    Resolve kitchen visual embedding variant from an environment alias.

    Args:
        env_alias: User-provided alias (e.g., "kitchen-vis-l", "kitchenstudio_base").
                  If None or empty, returns the default variant.

    Returns:
        Canonical variant name ("base", "large", or "smallplus").

    Raises:
        ValueError: If alias is not recognized.
    """
    return kitchen_vis_registry.resolve(env_alias)


def resolve_libero_variant(env_alias: Optional[str]) -> str:
    """
    Resolve libero model variant from an environment alias.

    Args:
        env_alias: User-provided alias (e.g., "libero-l", "libero_base").
                  If None or empty, returns the default variant.

    Returns:
        Canonical variant name ("base", "large", or "smallplus").

    Raises:
        ValueError: If alias is not recognized.
    """
    return libero_registry.resolve(env_alias)


# Export backward-compatible dictionaries
KITCHEN_VIS_ENV_MAP = dict(kitchen_vis_registry._registry)
LIBERO_ENV_MODEL_MAP = dict(libero_registry._registry)
