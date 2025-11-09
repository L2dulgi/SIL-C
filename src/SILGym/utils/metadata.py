"""
Utilities for constructing metadata payloads used during remote evaluation.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Tuple, Union, Dict

MetadataInput = Optional[Union[Mapping[str, Any], MutableMapping[str, Any]]]
PhaseDescriptor = Optional[Union[str, Tuple[Any, ...], list]]


def _extract_phase_label(descriptor: PhaseDescriptor) -> Optional[str]:
    """
    Normalize phase descriptor values produced by agent_config lookups.
    Accepts either raw strings (e.g., 'pre_0') or tuples/lists of the form
    (phase_index, 'policy_2/pre_1').
    """
    if descriptor is None:
        return None
    if isinstance(descriptor, (list, tuple)) and len(descriptor) >= 2:
        label = descriptor[1]
    else:
        label = descriptor
    if label is None:
        return None
    return str(label)


def build_phase_metadata(
    agent_config: MetadataInput,
    phase_name: Optional[str] = None,
    environment: Optional[str] = None,
    extra: MetadataInput = None,
) -> Dict[str, str]:
    """
    Construct a metadata dictionary for remote evaluators.

    Parameters
    ----------
    agent_config:
        Mapping with entries such as {'decoder': (idx, 'pre_0'), 'policy': (idx, 'policy_2/pre_1')}.
    phase_name:
        Name of the current training/evaluation phase.
    environment:
        Canonical environment identifier (e.g., 'libero').
    extra:
        Optional mapping of additional metadata fields.
    """
    metadata: Dict[str, str] = {}

    if environment:
        metadata["environment"] = str(environment)
    if phase_name:
        metadata["phase_name"] = str(phase_name)

    if agent_config:
        decoder_phase = _extract_phase_label(agent_config.get("decoder"))  # type: ignore[arg-type]
        if decoder_phase:
            metadata["decoder_phase"] = decoder_phase

        policy_phase = _extract_phase_label(agent_config.get("policy"))  # type: ignore[arg-type]
        if policy_phase:
            metadata["policy_phase"] = policy_phase

        interface_phase = _extract_phase_label(agent_config.get("interface"))  # type: ignore[arg-type]
        if interface_phase:
            metadata["interface_phase"] = interface_phase

    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            metadata[str(key)] = str(value)

    return metadata

