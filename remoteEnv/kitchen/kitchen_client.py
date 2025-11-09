"""
Interactive client for Kitchen remote evaluation.

This script mirrors the server-side API implemented in ``kitchen_server.py``
and adds helpers that make it easy to inspect observations – including the
vision payload returned when ``vision_mode`` + ``embedding_mode`` are enabled.

Usage (from the ``kitchen_eval`` conda environment):

    python kitchen_client.py --vision --embedding --episodes 1 --steps 2

By default the client prints concise per-step summaries (dtype / shapes) for up
to three environments. Increase ``--max-envs`` or ``--print-every`` as needed.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def recvall(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
    """Receive exactly ``num_bytes`` from the socket."""
    data = bytearray()
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def send_pickle(sock: socket.socket, obj: Any) -> None:
    """Pickle ``obj`` and send it with a 4-byte length prefix."""
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(payload)))
    sock.sendall(payload)


def recv_pickle(sock: socket.socket) -> Any:
    """Receive a pickled payload that was written with ``send_pickle``."""
    length_bytes = recvall(sock, 4)
    if not length_bytes:
        return None
    (length,) = struct.unpack("!I", length_bytes)
    data = recvall(sock, length)
    if not data:
        return None
    return pickle.loads(data)


@dataclass
class DummyModel:
    """Simple random policy useful for probing the remote API."""

    action_dim: int = 9

    def __call__(self, observations: Sequence[Any]) -> List[List[float]]:
        batch = len(observations) if observations is not None else 1
        actions = np.random.uniform(-1.0, 1.0, size=(batch, self.action_dim))
        return actions.tolist()


class KitchenRemoteClient:
    """Client wrapper that understands the multi-env Kitchen server protocol."""

    DEFAULT_CONFIG: List[Dict[str, Any]] = [{"data_name": "mbls"}]

    def __init__(
        self,
        host: str,
        port: int,
        action_dim: int = 9,
        expected_schema: Optional[Mapping[str, Tuple[Tuple[int, ...], str]]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.host, self.port))
        print(f"[client] Connected to server at {self.host}:{self.port}")

        self.model = DummyModel(action_dim=action_dim)
        self.current_eval_config: List[Dict[str, Any]] = list(self.DEFAULT_CONFIG)
        self.current_response: Dict[str, Any] = {}
        self.num_envs = 0
        self.expected_schema = dict(expected_schema) if expected_schema else {}

        initial = recv_pickle(self.sock)
        if initial is None:
            raise RuntimeError("Failed to receive initial observation from server.")
        self._update_state(initial)

    # ------------------------------------------------------------------ helpers
    def _update_state(self, response: Mapping[str, Any]) -> Mapping[str, Any]:
        observations = response.get("observations") or []
        self.current_response = dict(response)
        self.num_envs = len(observations)
        return response

    def _ensure_eval_config(self, eval_config: Optional[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
        if eval_config is None:
            return list(self.current_eval_config)
        packed: List[Dict[str, Any]] = []
        for item in eval_config:
            packed.append({str(k): v for k, v in dict(item).items()})
        return packed or list(self.DEFAULT_CONFIG)

    @staticmethod
    def _format_image_size(image_size: Optional[Tuple[int, int] | int]) -> Optional[List[int] | int]:
        if image_size is None:
            return None
        if isinstance(image_size, (tuple, list)):
            if len(image_size) != 2:
                raise ValueError("image_size tuple must contain (width, height).")
            return [int(image_size[0]), int(image_size[1])]
        return int(image_size)

    # ---------------------------------------------------------------- interface
    def set_task(
        self,
        eval_config: Optional[Iterable[Mapping[str, Any]]] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        noise_scale: Optional[float] = None,
    ) -> Mapping[str, Any]:
        msg: Dict[str, Any] = {"set_task": self._ensure_eval_config(eval_config)}
        if metadata:
            msg["metadata"] = {str(k): v for k, v in metadata.items()}
        if noise_scale is not None:
            msg["noise_scale"] = float(noise_scale)
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        if response is None:
            raise RuntimeError("Server closed connection during set_task.")
        self.current_eval_config = msg["set_task"]
        return self._update_state(response)

    def enable_vision_mode(
        self,
        *,
        eval_config: Optional[Iterable[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        payload = dict(metadata or {})
        payload.setdefault("obs_mode", "vision")
        payload.setdefault("vision_mode", True)
        config = self._ensure_eval_config(eval_config)
        print("[client] Requesting vision-mode observations...")
        return self.set_task(config, metadata=payload)

    def set_embedding_mode(
        self,
        enabled: bool,
        *,
        image_size: Optional[Tuple[int, int] | int] = None,
    ) -> Mapping[str, Any]:
        msg: Dict[str, Any] = {"set_embedding_mode": bool(enabled)}
        formatted_size = self._format_image_size(image_size)
        if formatted_size is not None:
            msg["image_size"] = formatted_size
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        if response is None:
            raise RuntimeError("Server closed connection during set_embedding_mode.")
        return self._update_state(response)

    def reset(self) -> Mapping[str, Any]:
        send_pickle(self.sock, {"reset": True})
        response = recv_pickle(self.sock)
        if response is None:
            raise RuntimeError("Server closed connection during reset.")
        return self._update_state(response)

    def step(self, actions: Sequence[Sequence[float]]) -> Mapping[str, Any]:
        formatted = self._format_actions(actions)
        send_pickle(self.sock, {"action": formatted})
        response = recv_pickle(self.sock)
        if response is None:
            raise RuntimeError("Server closed connection during step.")
        return self._update_state(response)

    def close(self) -> None:
        self.sock.close()
        print("[client] Connection closed.")

    # --------------------------------------------------------------- inspection
    @staticmethod
    def _describe_value(value: Any) -> Tuple[Tuple[int, ...], str]:
        if isinstance(value, np.ndarray):
            return tuple(value.shape), str(value.dtype)
        if isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
                return tuple(arr.shape), str(arr.dtype)
            except Exception:
                return (len(value),), type(value).__name__
        if isinstance(value, (int, float, bool)):
            return (), type(value).__name__
        return (), type(value).__name__

    def summarize_observations(self, *, step: int, max_envs: int = 3) -> None:
        obs_batch = self.current_response.get("observations") or []
        rewards = self.current_response.get("rewards")
        done_flags = self.current_response.get("done")

        print(f"[client] Step {step}: received {len(obs_batch)} env observations")
        for env_idx, obs in enumerate(obs_batch[:max_envs]):
            print(f"  └─ env[{env_idx}] -> type={type(obs).__name__}")
            if isinstance(obs, Mapping):
                for key in sorted(obs.keys()):
                    shape, dtype = self._describe_value(obs[key])
                    print(f"       • {key:20s} shape={shape} dtype={dtype}")
            else:
                shape, dtype = self._describe_value(obs)
                print(f"       • state shape={shape} dtype={dtype}")
        if len(obs_batch) > max_envs:
            print(f"  └─ ... {len(obs_batch) - max_envs} additional environments omitted")

        if self.expected_schema and obs_batch:
            observed_keys = set(obs_batch[0].keys()) if isinstance(obs_batch[0], Mapping) else set()
            missing_keys = sorted(k for k in self.expected_schema if k not in observed_keys)
            extra_keys = sorted(k for k in observed_keys if k not in self.expected_schema)
            if missing_keys:
                print(f"  schema missing keys -> {missing_keys}")
            if extra_keys:
                print(f"  schema extra keys   -> {extra_keys}")
            mismatches: list[str] = []
            for key in sorted(observed_keys & self.expected_schema.keys()):
                expected_shape, expected_dtype = self.expected_schema[key]
                actual_shape, actual_dtype = self._describe_value(obs_batch[0][key])
                if expected_shape and actual_shape and tuple(actual_shape) != tuple(expected_shape):
                    mismatches.append(f"{key}: expected {expected_shape}, got {actual_shape}")
                elif expected_shape and not actual_shape:
                    mismatches.append(f"{key}: expected {expected_shape}, got scalar")
                if expected_dtype and actual_dtype and actual_dtype != expected_dtype:
                    mismatches.append(f"{key}: expected dtype {expected_dtype}, got {actual_dtype}")
            if mismatches:
                print("  schema mismatches:")
                for entry in mismatches:
                    print(f"    - {entry}")

        if rewards is not None:
            rewards_arr = np.asarray(rewards)
            reward_mean = rewards_arr.mean() if rewards_arr.size else 0.0
            print(f"  reward stats -> shape={rewards_arr.shape}, mean={reward_mean:.3f}")
        if done_flags is not None:
            print(f"  done flags   -> {done_flags}")

    # --------------------------------------------------------------- rollouts
    def rollout(
        self,
        *,
        episodes: int,
        max_steps: int,
        print_every: int = 1,
        max_envs: int = 3,
        sleep: float = 0.0,
    ) -> None:
        for episode in range(episodes):
            print(f"\n[client] === Episode {episode + 1}/{episodes} ===")
            for step_idx in range(max_steps):
                if print_every and (step_idx % print_every == 0):
                    self.summarize_observations(step=step_idx, max_envs=max_envs)

                observations = self.current_response.get("observations") or []
                actions = self.model(observations)
                response = self.step(actions)

                if print_every and (step_idx % print_every == 0):
                    rewards = np.asarray(response.get("rewards") or [])
                    done_flags = response.get("done") or []
                    print(
                        f"[client]   -> after step: reward_mean={rewards.mean() if rewards.size else 0:.3f}, "
                        f"done={done_flags}"
                    )

                if all(response.get("done") or []):
                    print("[client]   -> all environments done, breaking episode.")
                    break
                if sleep > 0.0:
                    time.sleep(sleep)

            self.reset()
            print("[client] Environment reset.")

    def _format_actions(self, actions: Sequence[Sequence[float]]) -> List[List[float]]:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.shape[0] != self.num_envs:
            if self.num_envs == 1:
                arr = arr.reshape(1, -1)
            else:
                raise ValueError(
                    f"Expected {self.num_envs} action rows, received shape {arr.shape}."
                )
        return arr.tolist()


# --------------------------------------------------------------------------- CLI
def parse_metadata(entries: Sequence[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Metadata entry '{item}' must be in key=value format.")
        key, value = item.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def parse_eval_config(task_spec: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not task_spec:
        return None
    configs: List[Dict[str, Any]] = []
    for token in task_spec.split(","):
        cleaned = token.strip()
        if cleaned:
            configs.append({"data_name": cleaned})
    return configs or None


def summarize_dataset_sample(
    path: str, demo_index: int = 0
) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Print observation schema for a dataset sample and return feature specs."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("h5py is required to inspect kitchen datasets.") from exc

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with h5py.File(dataset_path, "r") as h5file:
        data_group = h5file.get("data")
        if data_group is None or not isinstance(data_group, h5py.Group):
            raise RuntimeError(f"'data' group missing in {dataset_path}")
        demo_keys = sorted(data_group.keys())
        if not demo_keys:
            raise RuntimeError(f"No trajectories found in {dataset_path}")
        demo_key = demo_keys[min(max(demo_index, 0), len(demo_keys) - 1)]
        obs_group = data_group[demo_key].get("obs")
        if obs_group is None or not isinstance(obs_group, h5py.Group):
            raise RuntimeError(f"'obs' group missing for {demo_key} in {dataset_path}")

        print(f"[dataset] {dataset_path.name} -> {demo_key}")
        schema: Dict[str, Tuple[Tuple[int, ...], str]] = {}
        for obs_key in sorted(obs_group.keys()):
            dataset = obs_group[obs_key]
            shape = dataset.shape
            dtype = dataset.dtype
            feature_dims = shape[1:] if len(shape) > 1 else ()
            schema[obs_key] = (tuple(int(dim) for dim in feature_dims), str(dtype))
            print(f"  • {obs_key:24s} shape={shape} dtype={dtype}")
        return schema


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kitchen remote evaluation client.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=9999, help="Server port.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to run.")
    parser.add_argument("--steps", type=int, default=10, help="Max steps per episode.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print observation summary every N steps (0 disables logging).",
    )
    parser.add_argument(
        "--max-envs",
        type=int,
        default=3,
        help="Maximum number of environments to print in summaries.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep (seconds) between steps to slow down logging.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Comma separated evaluation task identifiers (e.g. 'mbls,mkls').",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        nargs="*",
        default=None,
        help="Additional metadata key=value pairs forwarded with set_task.",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Request full vision observations (requires server support).",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Enable embedding mode so agent/wrist images are transmitted.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size requested when embedding mode is active.",
    )
    parser.add_argument(
        "--dataset-sample",
        type=str,
        help="Optional path to a kitchen_lerobot_embed *.hdf5 file to inspect for schema comparison.",
    )
    parser.add_argument(
        "--dataset-demo-index",
        type=int,
        default=0,
        help="Trajectory index within the dataset file to summarise (default: 0).",
    )
    parser.add_argument(
        "--skip-remote",
        action="store_true",
        help="Only perform dataset inspection; do not connect to the remote server.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.dataset_sample:
        try:
            dataset_schema = summarize_dataset_sample(
                args.dataset_sample, demo_index=args.dataset_demo_index
            )
        except Exception as exc:
            print(f"[dataset] Failed to summarise dataset: {exc}")
            dataset_schema = {}
        print()
    else:
        dataset_schema = {}

    if args.skip_remote:
        return

    metadata = parse_metadata(args.metadata) if args.metadata else {}
    eval_config = parse_eval_config(args.task)

    client = KitchenRemoteClient(
        host=args.host,
        port=args.port,
        expected_schema=dataset_schema or None,
    )

    try:
        if args.vision or metadata:
            if args.vision:
                metadata.setdefault("obs_mode", "vision")
                metadata.setdefault("vision_mode", True)
            client.enable_vision_mode(eval_config=eval_config, metadata=metadata)
        elif eval_config:
            client.set_task(eval_config, metadata=metadata or None)

        if args.embedding:
            size_payload: Optional[int | Tuple[int, int]] = None
            if args.image_size > 0:
                size_payload = int(args.image_size)
            client.set_embedding_mode(True, image_size=size_payload)

        client.rollout(
            episodes=args.episodes,
            max_steps=args.steps,
            print_every=args.print_every,
            max_envs=args.max_envs,
            sleep=args.sleep,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
