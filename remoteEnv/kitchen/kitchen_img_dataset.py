#!/usr/bin/env python3
"""Convert D4RL kitchen demonstrations into LeRobot-compatible HDF5 files.

The original datasets are pickled dictionaries following the D4RL convention.
This script reconstructs the MuJoCo simulator state for every timestep, renders
RGB observations from both the front (agent) and wrist (eye-in-hand) cameras,
and stores everything – together with the original proprioceptive signals – in
the LeRobot structure used by LIBERO (see ``./data/libero`` for examples).

Run this tool from the ``kitchen_eval`` conda environment so that MuJoCo,
dm_control, and d4rl are importable. Copy the customised ``chain0.xml`` as
documented in ``remoteEnv/kitchen/readme.md`` before generating wrist views.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# Ensure the sibling module (kitchen.py) is importable even when launched elsewhere.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from kitchen import KitchenEnv, KitchenTask  # type: ignore
except Exception as exc:  # pragma: no cover - surface a clearer error message.
    raise RuntimeError(
        "Failed to import KitchenEnv. Ensure that d4rl, mujoco, and dm_control "
        "are installed within the 'kitchen_eval' conda environment."
    ) from exc


ROBOT_QPOS_DIM = 9
BASE_VIEW_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
STUDIO_VIEW_MAPPING = {
    "overview": "agentview_rgb",
    "wrist": "eye_in_hand_rgb",
    "ovens": "ovens_rgb",
    "pannels": "pannels_rgb",
    "cabinets": "cabinets_rgb",
}
STUDIO_FALLBACK_SUFFIX = "_rgb"
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("~/SILGym/data/evolving_kitchen/raw").expanduser(),
        help="Directory containing the kitchen dataset pickle files.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Specific dataset filename (within data-root) to convert. "
        "When omitted, every *.pkl file in the directory is processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/kitchen_lerobot"),
        help="Directory to store generated HDF5 files.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Render width in pixels (>=224).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Render height in pixels (>=224).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=1,
        help="Render device id forwarded to KitchenEnv (see kitchen.py).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Metadata only: frames per second stored in the HDF5 file attributes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 files with the same output name.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show per-episode progress bars (helpful for long trajectories).",
    )
    parser.add_argument(
        "--studio",
        action="store_true",
        help="Enable multi-camera studio rendering using KitchenEnv.render_studio.",
    )
    return parser.parse_args()


def discover_dataset_paths(data_root: Path, dataset_name: str | None) -> List[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root {data_root} does not exist.")

    if dataset_name:
        candidate = data_root / dataset_name
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset {candidate} not found.")
        return [candidate]

    paths = sorted(data_root.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No *.pkl files found under {data_root}.")
    return paths


def parse_task_sequence(dataset_path: Path) -> Sequence[str]:
    """Infer the semantic skill ordering encoded in the pickle filename."""
    subtasks = dataset_path.stem.split("-")
    return [token.strip() for token in subtasks if token.strip()]


def episode_slices(terminals: Sequence[bool]) -> Iterator[Tuple[int, int]]:
    """Yield contiguous slices [start, end) representing episodes."""
    start = 0
    for idx, done in enumerate(terminals):
        if done:
            yield start, idx + 1
            start = idx + 1
    if start < len(terminals):
        yield start, len(terminals)


def observation_to_state(obs: np.ndarray, nq: int, nv: int) -> Tuple[np.ndarray, np.ndarray]:
    if obs.ndim != 1:
        raise ValueError(f"Observation is expected to be 1D, got shape {obs.shape}.")
    required = nq + nv
    if obs.shape[0] < required:
        raise ValueError(
            f"Observation dimension {obs.shape[0]} is smaller than nq+nv ({required}). "
            "Cannot reconstruct simulator state."
        )
    qpos = obs[:nq]
    qvel = obs[nq : nq + nv]
    return qpos, qvel


def set_sim_state_from_observation(env: KitchenEnv, obs: np.ndarray) -> None:
    """Write simulator state directly, mirroring FrankaRobot.reset."""
    nq = env.sim.model.nq
    nv = env.sim.model.nv
    qpos, qvel = observation_to_state(obs, nq, nv)
    env.sim.data.qpos[:nq] = qpos
    env.sim.data.qvel[:nv] = qvel
    env.sim.forward()
    if hasattr(env, "_get_obs"):
        env._get_obs()  # refresh cached obs for downstream reads


def sanitize_name(stem: str) -> str:
    """Produce a filesystem-friendly identifier."""
    cleaned = re.sub(r"[^0-9a-zA-Z._-]+", "_", stem)
    cleaned = cleaned.strip("_")
    return cleaned or "kitchen_demo"


def ensure_rgb_frame(frame: np.ndarray, width: int, height: int, view: str) -> np.ndarray:
    """Validate frame layout and resize to the requested resolution."""
    rgb = np.asarray(frame)
    if rgb.ndim != 3:
        raise ValueError(f"{view} frame must be HxWxC, got shape {rgb.shape}.")
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.shape[-1] != 3:
        raise ValueError(f"{view} frame has {rgb.shape[-1]} channels; expected 3.")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    h, w = rgb.shape[:2]
    if h != height or w != width:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    return np.ascontiguousarray(rgb)


def render_views(env: KitchenEnv, width: int, height: int, studio: bool) -> Dict[str, np.ndarray]:
    """Render all camera streams required for LeRobot format."""
    if studio:
        try:
            raw_views = env.render_studio(mode="rgb_array")
        except Exception as exc:  # pragma: no cover - depends on mujoco runtime
            raise RuntimeError(f"Failed to render studio views: {exc}") from exc
        if not isinstance(raw_views, dict):
            raise RuntimeError(
                f"KitchenEnv.render_studio must return a dict, got {type(raw_views)}."
            )

        frames: Dict[str, np.ndarray] = {}
        for raw_name, dataset_name in STUDIO_VIEW_MAPPING.items():
            if raw_name in raw_views:
                frames[dataset_name] = ensure_rgb_frame(
                    raw_views[raw_name], width, height, dataset_name
                )

        for raw_name, raw_frame in raw_views.items():
            if raw_name in STUDIO_VIEW_MAPPING:
                continue
            dataset_name = f"{raw_name}{STUDIO_FALLBACK_SUFFIX}"
            frames[dataset_name] = ensure_rgb_frame(raw_frame, width, height, dataset_name)

        if "agentview_rgb" not in frames:
            try:
                fallback = env.render(mode="rgb_array")
            except Exception as exc:  # pragma: no cover - depends on mujoco runtime
                raise RuntimeError(f"Failed to render agent view fallback: {exc}") from exc
            frames["agentview_rgb"] = ensure_rgb_frame(
                fallback, width, height, "agentview_rgb"
            )
        if "eye_in_hand_rgb" not in frames:
            try:
                wrist_frame = env.render_wrist_view(mode="rgb_array")
            except Exception as exc:  # pragma: no cover - depends on mujoco runtime
                raise RuntimeError(f"Failed to render wrist view fallback: {exc}") from exc
            frames["eye_in_hand_rgb"] = ensure_rgb_frame(
                wrist_frame, width, height, "eye_in_hand_rgb"
            )
        return frames

    frames: Dict[str, np.ndarray] = {}
    try:
        frames["agentview_rgb"] = ensure_rgb_frame(
            env.render(mode="rgb_array"), width, height, "agentview_rgb"
        )
    except Exception as exc:  # pragma: no cover - depends on mujoco runtime
        raise RuntimeError(f"Failed to render agent view: {exc}") from exc
    try:
        wrist_frame = env.render_wrist_view(mode="rgb_array")
        frames["eye_in_hand_rgb"] = ensure_rgb_frame(
            wrist_frame, width, height, "eye_in_hand_rgb"
        )
    except Exception as exc:  # pragma: no cover - depends on mujoco runtime
        raise RuntimeError(f"Failed to render wrist view: {exc}") from exc
    return frames


def query_end_effector_pose(env: KitchenEnv) -> Tuple[np.ndarray, np.ndarray]:
    """Return end-effector position and orientation (axis-angle)."""
    if not hasattr(env, "get_ee_info"):
        raise RuntimeError("KitchenEnv does not expose get_ee_info; update kitchen.py.")
    try:
        ee_pos, ee_ori = env.get_ee_info()
    except Exception as exc:  # pragma: no cover - depends on mujoco runtime
        raise RuntimeError(f"Failed to query end effector pose: {exc}") from exc
    return np.asarray(ee_pos, dtype=np.float32), np.asarray(ee_ori, dtype=np.float32)


def query_gripper_states(env: KitchenEnv) -> np.ndarray:
    """Return the two finger joint positions."""
    try:
        # Franka finger joints follow the seven arm joints inside qpos.
        fingers = np.asarray(env.sim.data.qpos[7:9], dtype=np.float32)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to query gripper state: {exc}") from exc
    if fingers.shape[0] != 2:
        raise ValueError(f"Expected two gripper joints, got shape {fingers.shape}.")
    return fingers


def prepare_episode_payload(
    env: KitchenEnv,
    obs_batch: np.ndarray,
    actions_batch: np.ndarray,
    rewards_batch: np.ndarray,
    terminals_batch: np.ndarray,
    skills_batch: Sequence[str] | None,
    skill_done_batch: np.ndarray | None,
    skill_fobs_batch: np.ndarray | None,
    width: int,
    height: int,
    show_progress: bool,
    studio: bool,
) -> Tuple[Dict[str, np.ndarray], Tuple[str, ...]]:
    """Render and collate all artefacts for a single episode."""

    frame_buffers: Dict[str, List[np.ndarray]] = {}
    frame_keys: List[str] = []
    robot_states: List[np.ndarray] = []
    ee_positions: List[np.ndarray] = []
    ee_orientations: List[np.ndarray] = []
    gripper_states: List[np.ndarray] = []

    iterator = enumerate(obs_batch)
    if show_progress:
        iterator = tqdm(iterator, total=len(obs_batch), leave=False, desc="frames")

    for _, obs_vec in iterator:
        try:
            set_sim_state_from_observation(env, obs_vec)
            frames = render_views(env, width, height, studio=studio)
            ee_pos, ee_ori = query_end_effector_pose(env)
            gripper = query_gripper_states(env)
        except Exception as exc:
            raise RuntimeError(f"Failed to reconstruct frame: {exc}") from exc

        if not frame_buffers:
            frame_keys = list(frames.keys())
            if "agentview_rgb" not in frames or "eye_in_hand_rgb" not in frames:
                raise RuntimeError("Renderer did not produce required agent or wrist views.")
            frame_buffers = {key: [] for key in frame_keys}
        elif list(frames.keys()) != frame_keys:
            raise RuntimeError(
                f"Inconsistent camera keys between frames: expected {frame_keys}, got {list(frames.keys())}."
            )

        for key in frame_keys:
            frame_buffers[key].append(frames[key])

        robot_state = np.asarray(env.sim.data.qpos[:ROBOT_QPOS_DIM], dtype=np.float32)
        if robot_state.shape[0] != ROBOT_QPOS_DIM:
            raise ValueError(
                f"Expected at least {ROBOT_QPOS_DIM} robot qpos entries, got {robot_state.shape[0]}."
            )
        robot_states.append(robot_state)
        ee_positions.append(ee_pos)
        ee_orientations.append(ee_ori)
        gripper_states.append(gripper)

    payload: Dict[str, np.ndarray] = {}
    payload["actions"] = np.asarray(actions_batch, dtype=np.float32)
    payload["states"] = np.asarray(obs_batch, dtype=np.float32)
    payload["rewards"] = np.asarray(rewards_batch, dtype=np.float32)

    dones = np.asarray(terminals_batch, dtype=np.uint8)
    if dones.size:
        dones[-1] = 1  # guarantee terminal
    payload["dones"] = dones
    if not frame_keys:
        raise RuntimeError("No camera frames collected for the episode.")

    for key, buffers in frame_buffers.items():
        payload[f"obs_{key}"] = np.stack(buffers).astype(np.uint8)
    payload["obs_robot_states"] = np.stack(robot_states).astype(np.float32)
    payload["obs_ee_pos"] = np.stack(ee_positions).astype(np.float32)
    payload["obs_ee_ori"] = np.stack(ee_orientations).astype(np.float32)
    payload["obs_ee_states"] = np.concatenate(
        [payload["obs_ee_pos"], payload["obs_ee_ori"]], axis=1
    ).astype(np.float32)
    payload["obs_gripper_states"] = np.stack(gripper_states).astype(np.float32)

    if skills_batch is not None:
        payload["skills"] = np.asarray(skills_batch, dtype=object)
    if skill_done_batch is not None:
        payload["skill_done"] = np.asarray(skill_done_batch, dtype=np.uint8)
    if skill_fobs_batch is not None:
        payload["skill_fobs"] = np.asarray(skill_fobs_batch, dtype=np.float32)

    return payload, tuple(frame_keys)


def write_episode(
    demo_group: h5py.Group,
    payload: Dict[str, np.ndarray],
    view_keys: Sequence[str],
) -> None:
    """Persist a single episode to the provided HDF5 group."""

    obs_group = demo_group.create_group("obs")

    for key in view_keys:
        dataset_key = f"obs_{key}"
        if dataset_key not in payload:
            raise RuntimeError(f"Missing payload entry for {dataset_key}.")
        frames = payload[dataset_key]
        obs_group.create_dataset(
            key,
            data=frames,
            compression="gzip",
            compression_opts=4,
            chunks=(1,) + frames.shape[1:],
        )
    obs_group.create_dataset(
        "robot_states",
        data=payload["obs_robot_states"],
        compression="gzip",
        compression_opts=4,
        chunks=(min(len(payload["obs_robot_states"]), 256), ROBOT_QPOS_DIM),
    )
    obs_group.create_dataset(
        "ee_pos",
        data=payload["obs_ee_pos"],
        compression="gzip",
        compression_opts=4,
        chunks=(min(len(payload["obs_ee_pos"]), 256), 3),
    )
    obs_group.create_dataset(
        "ee_ori",
        data=payload["obs_ee_ori"],
        compression="gzip",
        compression_opts=4,
        chunks=(min(len(payload["obs_ee_ori"]), 256), 3),
    )
    obs_group.create_dataset(
        "ee_states",
        data=payload["obs_ee_states"],
        compression="gzip",
        compression_opts=4,
        chunks=(min(len(payload["obs_ee_states"]), 256), 6),
    )
    obs_group.create_dataset(
        "gripper_states",
        data=payload["obs_gripper_states"],
        compression="gzip",
        compression_opts=4,
        chunks=(min(len(payload["obs_gripper_states"]), 256), 2),
    )

    demo_group.create_dataset(
        "actions",
        data=payload["actions"],
        compression="gzip",
        compression_opts=4,
        chunks=(min(len(payload["actions"]), 256), payload["actions"].shape[-1]),
    )
    demo_group.create_dataset(
        "states",
        data=payload["states"],
        compression="gzip",
        compression_opts=4,
        chunks=(min(len(payload["states"]), 256), payload["states"].shape[-1]),
    )
    demo_group.create_dataset(
        "rewards",
        data=payload["rewards"],
        compression="gzip",
        compression_opts=4,
        chunks=(min(len(payload["rewards"]), 512),),
    )
    demo_group.create_dataset(
        "dones",
        data=payload["dones"],
        compression="gzip",
        compression_opts=4,
        chunks=(min(len(payload["dones"]), 512),),
    )

    if "skills" in payload:
        dataset = demo_group.create_dataset("skills", (len(payload["skills"]),), dtype=STRING_DTYPE)
        dataset[:] = np.asarray(payload["skills"], dtype=object)
    if "skill_done" in payload:
        demo_group.create_dataset(
            "skill_done",
            data=payload["skill_done"],
            compression="gzip",
            compression_opts=4,
            chunks=(min(len(payload["skill_done"]), 512),),
        )
    if "skill_fobs" in payload:
        demo_group.create_dataset(
            "skill_fobs",
            data=payload["skill_fobs"],
            compression="gzip",
            compression_opts=4,
            chunks=(min(len(payload["skill_fobs"]), 256), payload["skill_fobs"].shape[-1]),
        )


def process_dataset(
    dataset_path: Path,
    args: argparse.Namespace,
) -> None:
    with dataset_path.open("rb") as f:
        data = pickle.load(f)

    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32)
    terminals = np.asarray(data["terminals"], dtype=np.uint8)
    skills = data.get("skills")
    if skills is not None:
        skills = list(skills)
    skill_done = np.asarray(data.get("skill_done", []), dtype=np.uint8) if data.get("skill_done") else None
    skill_fobs = np.asarray(data.get("skill_fobs", []), dtype=np.float32) if data.get("skill_fobs") else None

    if observations.shape[0] != actions.shape[0]:
        raise ValueError("Observation and action counts do not match.")

    episodes = list(episode_slices(terminals))
    if not episodes:
        episodes = [(0, len(observations))]

    output_name = sanitize_name(dataset_path.stem) + "_demo.hdf5"
    output_path = args.output_dir.expanduser() / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"[SKIP] {output_path} exists. Use --overwrite to regenerate.")
        return

    env = KitchenEnv()
    env.set_render_options(
        width=max(args.width, 224),
        height=max(args.height, 224),
        device=args.device,
        fps=int(args.fps),
    )

    subtasks = parse_task_sequence(dataset_path)
    if subtasks:
        try:
            env.set_task_default(KitchenTask(subtasks=subtasks))
        except Exception as exc:
            print(f"[WARN] Failed to set task sequence {subtasks}: {exc}")

    try:
        with h5py.File(output_path, "w") as h5_file:
            h5_file.attrs["source_dataset"] = dataset_path.name
            h5_file.attrs["render_width"] = int(env.render_width)
            h5_file.attrs["render_height"] = int(env.render_height)
            h5_file.attrs["fps"] = float(args.fps)
            h5_file.attrs["camera_keys"] = np.asarray(BASE_VIEW_KEYS, dtype=STRING_DTYPE)
            if subtasks:
                h5_file.attrs["task_subtasks"] = np.asarray(subtasks, dtype=STRING_DTYPE)

            data_group = h5_file.create_group("data")

            expected_view_keys: Tuple[str, ...] | None = None
            iterator = enumerate(
                tqdm(episodes, desc=dataset_path.stem, disable=not args.progress)
            )
            for ep_idx, (start, end) in iterator:
                demo_group = data_group.create_group(f"demo_{ep_idx}")

                episode_payload, frame_keys = prepare_episode_payload(
                    env=env,
                    obs_batch=observations[start:end],
                    actions_batch=actions[start:end],
                    rewards_batch=rewards[start:end],
                    terminals_batch=terminals[start:end],
                    skills_batch=skills[start:end] if skills is not None else None,
                    skill_done_batch=skill_done[start:end] if skill_done is not None else None,
                    skill_fobs_batch=skill_fobs[start:end] if skill_fobs is not None else None,
                    width=env.render_width,
                    height=env.render_height,
                    show_progress=args.progress,
                    studio=args.studio,
                )

                if expected_view_keys is None:
                    expected_view_keys = frame_keys
                    h5_file.attrs["camera_keys"] = np.asarray(frame_keys, dtype=STRING_DTYPE)
                elif frame_keys != expected_view_keys:
                    raise RuntimeError(
                        f"Inconsistent camera keys across demos: expected {expected_view_keys}, got {frame_keys}."
                    )

                write_episode(demo_group, episode_payload, frame_keys)
                demo_group.attrs["num_frames"] = episode_payload[f"obs_{BASE_VIEW_KEYS[0]}"].shape[0]
    except Exception:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    print(f"[OK] Saved {len(episodes)} demos to {output_path}")


def main() -> None:
    args = parse_args()
    paths = discover_dataset_paths(args.data_root, args.dataset)
    for path in paths:
        try:
            process_dataset(path, args)
        except Exception as exc:
            print(f"[ERROR] Conversion failed for {path.name}: {exc}")


if __name__ == "__main__":
    main()
