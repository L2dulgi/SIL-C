#!/usr/bin/env python3
"""
Verify that converted kitchen_lerobot_embed datasets match their raw evolving_kitchen sources.

Usage example:
    python verify_dataset_alignment.py \
        --variant smallplus \
        --raw-dir data/evolving_kitchen/raw \
        --embed-dir data/kitchen_lerobot_embed
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py
import numpy as np


def sanitize_name(stem: str) -> str:
    """Match the filename transformation used during HDF5 export."""
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", stem)
    cleaned = cleaned.strip("_")
    return cleaned or "kitchen_demo"


def episode_segments(terminals: Sequence[bool]) -> List[Tuple[int, int]]:
    """Return inclusive-exclusive index pairs for each episode."""
    segments: List[Tuple[int, int]] = []
    start = 0
    for idx, flag in enumerate(terminals):
        if flag:
            segments.append((start, idx + 1))
            start = idx + 1
    if start < len(terminals):
        segments.append((start, len(terminals)))
    return segments


def stack_range(seq: Sequence, start: int, end: int, *, dtype=np.float32) -> np.ndarray:
    """Stack a list-like sequence of numpy arrays between [start, end)."""
    if start >= end:
        return np.zeros((0,), dtype=dtype)
    return np.stack([np.asarray(seq[i], dtype=dtype) for i in range(start, end)], axis=0)


def to_string_list(array: Iterable) -> List[str]:
    values: List[str] = []
    for item in array:
        if isinstance(item, bytes):
            values.append(item.decode("utf-8"))
        else:
            values.append(str(item))
    return values


def verify_file(
    raw_path: Path,
    embed_path: Path,
    *,
    rtol: float,
    atol: float,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    with raw_path.open("rb") as raw_file:
        raw_data = pickle.load(raw_file)

    observations = raw_data["observations"]
    actions = raw_data["actions"]
    rewards = raw_data["rewards"]
    terminals = raw_data["terminals"]
    skills = raw_data.get("skills")
    skill_done = raw_data.get("skill_done")
    skill_fobs = raw_data.get("skill_fobs")

    segments = episode_segments(terminals)

    if not embed_path.exists():
        errors.append(f"Missing converted file: {embed_path.name}")
        return False, errors

    with h5py.File(embed_path, "r") as h5_file:
        data_group = h5_file.get("data")
        if data_group is None:
            return False, [f"'data' group missing in {embed_path.name}"]

        demo_keys = sorted(
            data_group.keys(),
            key=lambda name: int(name.split("_")[1]),
        )

        if len(demo_keys) != len(segments):
            errors.append(
                f"Episode count mismatch (raw={len(segments)}, embed={len(demo_keys)})"
            )
            return False, errors

        for (start, end), demo_key in zip(segments, demo_keys):
            demo = data_group[demo_key]
            length = end - start

            def check(name: str, raw_array: np.ndarray, converted: np.ndarray) -> None:
                if raw_array.shape != converted.shape:
                    errors.append(
                        f"{demo_key}:{name} shape mismatch "
                        f"(raw={raw_array.shape}, embed={converted.shape})"
                    )
                    return
                if not np.allclose(raw_array, converted, rtol=rtol, atol=atol):
                    errors.append(f"{demo_key}:{name} values differ beyond tolerance")

            raw_states = stack_range(observations, start, end, dtype=np.float32)
            raw_actions = stack_range(actions, start, end, dtype=np.float32)
            raw_rewards = np.asarray(rewards[start:end], dtype=np.float32)
            raw_dones = np.asarray(terminals[start:end], dtype=np.uint8)

            check("states", raw_states, np.asarray(demo["states"]))
            check("actions", raw_actions, np.asarray(demo["actions"]))
            check("rewards", raw_rewards, np.asarray(demo["rewards"]))

            embed_dones = np.asarray(demo["dones"], dtype=np.uint8)
            if raw_dones.shape != embed_dones.shape or not np.array_equal(raw_dones, embed_dones):
                errors.append(f"{demo_key}:dones mismatch")

            if skills is not None:
                raw_skills = [str(skills[i]) for i in range(start, end)]
                embed_skills = to_string_list(demo["skills"][:])
                if raw_skills != embed_skills:
                    errors.append(f"{demo_key}:skills mismatch")

            if skill_done is not None:
                raw_sd = np.asarray(skill_done[start:end], dtype=np.uint8)
                embed_sd = np.asarray(demo["skill_done"], dtype=np.uint8)
                if raw_sd.shape != embed_sd.shape or not np.array_equal(raw_sd, embed_sd):
                    errors.append(f"{demo_key}:skill_done mismatch")

            if skill_fobs is not None:
                raw_sf = stack_range(skill_fobs, start, end, dtype=np.float32)
                embed_sf = np.asarray(demo["skill_fobs"])
                check("skill_fobs", raw_sf, embed_sf)

            if demo["states"].shape[0] != length:
                errors.append(f"{demo_key}: episode length mismatch ({length} vs {demo['states'].shape[0]})")

    return (len(errors) == 0), errors


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/evolving_kitchen/raw"),
        help="Directory containing the original pickle datasets.",
    )
    parser.add_argument(
        "--embed-dir",
        type=Path,
        default=Path("data/kitchen_lerobot_embed"),
        help="Root directory that stores embedded HDF5 datasets.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="smallplus",
        help="Subdirectory under embed-dir to validate (e.g., base, large, smallplus).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of files to verify (0 = all).",
    )
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for numeric checks.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for numeric checks.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = parse_args()
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir.expanduser().resolve()
    embed_variant_dir = (args.embed_dir / args.variant / "raw").expanduser().resolve()

    if not raw_dir.exists():
        print(f"[ERROR] Raw directory not found: {raw_dir}", file=sys.stderr)
        return 1
    if not embed_variant_dir.exists():
        print(f"[ERROR] Embedded variant directory not found: {embed_variant_dir}", file=sys.stderr)
        return 1

    raw_files = sorted(
        [path for path in raw_dir.glob("*.pkl") if path.is_file()]
    )
    if args.limit > 0:
        raw_files = raw_files[: args.limit]

    total = 0
    failures = 0

    for raw_path in raw_files:
        total += 1
        sanitized = sanitize_name(raw_path.stem)
        embed_path = embed_variant_dir / f"{sanitized}_demo.hdf5"

        ok, errors = verify_file(
            raw_path,
            embed_path,
            rtol=args.rtol,
            atol=args.atol,
        )
        if ok:
            print(f"[OK] {raw_path.name}")
            continue

        failures += 1
        print(f"[FAIL] {raw_path.name}")
        for msg in errors:
            print(f"       - {msg}")

    if total == 0:
        print("[WARN] No raw datasets found.")
        return 1

    print(f"\nChecked {total} file(s); {failures} mismatch(es).")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
