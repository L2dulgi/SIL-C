import socket
import pickle
import struct
import time
import logging
import numpy as np
from tqdm import tqdm

import typing
from typing import TYPE_CHECKING
import os
from SILGym.utils.logger import get_logger
from SILGym.config.libero_scenario import (
    DEFAULT_LIBERO_MODEL,
    LIBERO_ENV_MODEL_MAP,
    LIBERO_MODEL_TO_DINOV3,
)

if TYPE_CHECKING:
    from SILGym.utils.image_embedding import DINOv3Embedder

remote_server_ip = os.environ.get("REMOTE_SERVER_IP", "127.0.0.1")

# Dummy model that outputs random actions for a batch of observations (assuming action dimension is 9)
class DummyModel:
    def eval_model(self, observations):
        # observations is expected to be a list (or array) of observations, one per environment.
        num_envs = len(observations)
        # Generate random actions for each environment.
        actions = np.random.uniform(-1, 1, size=(num_envs, 9)).tolist()
        return actions

def recvall(sock, n):
    """
    Receive exactly n bytes from the socket.
    """
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_pickle(sock, obj):
    """
    Serialize an object with pickle using HIGHEST_PROTOCOL,
    prepend its length as a 4-byte integer, and send it.
    """
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    sock.sendall(data)

def recv_pickle(sock):
    """
    Receive a pickled object from the socket.
    """
    length_bytes = recvall(sock, 4)
    if not length_bytes:
        return None
    length = struct.unpack('!I', length_bytes)[0]
    data = recvall(sock, length)
    if not data:
        return None
    return pickle.loads(data)

class HistoryEvalHelper:
    """
    Maintains a sliding window of observations and concatenates historical views.
    Supports arbitrary stack depths (>=1) by evenly sampling from the retained history.
    """

    def __init__(self, history_len: int = 5, stack_depth: int = 2):
        self.history_len = history_len
        self.history: list[np.ndarray] = []
        self._stack_depth = 2
        self.stack_depth = stack_depth

    @property
    def stack_depth(self) -> int:
        return self._stack_depth

    @stack_depth.setter
    def stack_depth(self, value: int) -> None:
        if value < 1:
            raise ValueError("stack_depth must be >= 1")
        self._stack_depth = value

    def add(self, obs: np.ndarray) -> None:
        if len(self.history) >= self.history_len:
            self.history.pop(0)
        self.history.append(obs)

    def get_state(self, obs: np.ndarray) -> np.ndarray:
        self.add(obs)
        hist_len = len(self.history)
        if hist_len == 0:
            return obs

        indices = np.linspace(0, hist_len - 1, num=self.stack_depth)
        indices = np.round(indices).astype(int)
        indices = np.clip(indices, 0, hist_len - 1)
        stacked = [self.history[idx] for idx in indices]
        return np.concatenate(stacked, axis=-1)


class HistoryEvalHelperTriple(HistoryEvalHelper):
    """
    Backwards-compatible helper that always returns three-frame histories.
    """

    def __init__(self, history_len: int = 5):
        super().__init__(history_len=history_len, stack_depth=3)


class NoiseEvaluationHelper:
    """
    Adds noise to observations during evaluation.
    Can be used standalone or chained with other helpers.
    """
    def __init__(self, noise_type='gaussian', noise_scale=0.01, noise_clip=None,
                 base_helper=None, seed=None):
        """
        Args:
            noise_type: Type of noise ('gaussian', 'uniform')
            noise_scale: Scale/magnitude of noise
            noise_clip: Optional clipping range for noisy observations
            base_helper: Another helper to chain with (e.g., HistoryEvalHelper)
            seed: Random seed for reproducibility
        """
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noise_clip = noise_clip
        self.base_helper = base_helper
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def add_noise(self, obs):
        """Add noise to observation."""
        if self.noise_type == 'gaussian':
            noise = self.rng.normal(0, self.noise_scale, obs.shape)
        elif self.noise_type == 'uniform':
            noise = self.rng.uniform(-self.noise_scale, self.noise_scale, obs.shape)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        noisy_obs = obs + noise

        # Apply clipping if specified
        if self.noise_clip is not None:
            noisy_obs = np.clip(noisy_obs, -self.noise_clip, self.noise_clip)

        return noisy_obs

    def get_state(self, obs):
        """Process observation, optionally through base helper first."""
        # If we have a base helper, use it first
        if self.base_helper is not None:
            obs = self.base_helper.get_state(obs)

        # Add noise to the observation
        return self.add_noise(obs)


class ActionChunkHelper:
    """
    Helper for managing action chunks during evaluation.

    When chunk_size > 1, the model predicts multiple future actions in a single
    forward pass. This helper buffers those actions and returns them one at a time,
    only invoking the model again when the buffer is empty.

    This reduces inference frequency by a factor of chunk_size, improving evaluation
    efficiency and potentially temporal consistency.
    """
    def __init__(self, eval_fn, chunk_size: int = 1, action_dim: int = 7):
        """
        Args:
            eval_fn: Evaluation function that takes observations and returns actions.
                     Expected to return shape (batch, action_dim * chunk_size) when chunking.
            chunk_size: Number of actions predicted per inference (1 = no chunking)
            action_dim: Dimension of a single action
        """
        self.eval_fn = eval_fn
        self.chunk_size = max(1, int(chunk_size))
        self.action_dim = int(action_dim)
        self.action_buffer = []
        self.logger = get_logger(__name__)

        if self.chunk_size > 1:
            self.logger.info(
                f"ActionChunkHelper initialized: chunk_size={self.chunk_size}, "
                f"action_dim={self.action_dim}"
            )

    def get_action(self, obs):
        """
        Get the next action for the given observation.

        If the buffer is empty, invokes the model to predict chunk_size actions
        and buffers them. Otherwise, returns the next buffered action.

        Args:
            obs: Observation(s) to evaluate

        Returns:
            Single action (or batch of actions) of shape (..., action_dim)
        """
        # If buffer is empty, invoke model to get new chunk
        if len(self.action_buffer) == 0:
            chunked_action = self.eval_fn(obs)
            chunked_action = np.asarray(chunked_action)

            # Handle both batched and single predictions
            if chunked_action.ndim == 1:
                # Single prediction: (action_dim * chunk_size,)
                expected_size = self.action_dim * self.chunk_size
                if chunked_action.shape[0] != expected_size:
                    self.logger.warning(
                        f"Expected action size {expected_size}, got {chunked_action.shape[0]}. "
                        f"Using first {expected_size} elements."
                    )
                    chunked_action = chunked_action[:expected_size]

                # Split into individual actions
                for i in range(self.chunk_size):
                    start_idx = i * self.action_dim
                    end_idx = start_idx + self.action_dim
                    self.action_buffer.append(chunked_action[start_idx:end_idx])
            else:
                # Batched prediction: (batch, action_dim * chunk_size)
                batch_size = chunked_action.shape[0]
                expected_size = self.action_dim * self.chunk_size

                if chunked_action.shape[-1] != expected_size:
                    self.logger.warning(
                        f"Expected action size {expected_size}, got {chunked_action.shape[-1]}. "
                        f"Using first {expected_size} elements."
                    )
                    chunked_action = chunked_action[..., :expected_size]

                # Split into individual actions
                for i in range(self.chunk_size):
                    start_idx = i * self.action_dim
                    end_idx = start_idx + self.action_dim
                    self.action_buffer.append(chunked_action[..., start_idx:end_idx])

        # Return and remove first action from buffer
        action = self.action_buffer.pop(0)

        # Convert to list for consistency with non-chunked actions
        # This ensures all remote evaluators receive list format
        if isinstance(action, np.ndarray):
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"ActionChunkHelper returning action: shape={action.shape}, "
                    f"type={type(action).__name__} → converting to list"
                )
            return action.tolist()
        return action

    def reset(self):
        """Clear the action buffer. Should be called at episode boundaries."""
        if len(self.action_buffer) > 0:
            self.logger.debug(f"ActionChunkHelper reset: discarding {len(self.action_buffer)} buffered actions")
        self.action_buffer = []


class KitchenRemoteEvaluator:
    """
    KitchenRemoteEvaluator communicates with a remote server that
    performs multi-environment evaluations.
    """
    def __init__(self, host=remote_server_ip, port=9999, obs_helper=None, eval_fn=None, image_mode=False, action_chunk_size=1):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm to reduce latency.
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        self.logger = get_logger(__name__)
        self.logger.info(f"Connected to server at {host}:{port}")

        # ----------------------------
        # Set the evaluation function
        # ----------------------------
        self.obs_helper = obs_helper

        # Wrap eval_fn with ActionChunkHelper if chunking is enabled
        self.action_chunk_size = int(action_chunk_size)
        if self.action_chunk_size > 1 and eval_fn is not None:
            self.chunk_helper = ActionChunkHelper(
                eval_fn=eval_fn,
                chunk_size=self.action_chunk_size,
                action_dim=9  # Kitchen action dimension
            )
            self.eval_fn = self.chunk_helper.get_action
        else:
            self.chunk_helper = None
            self.eval_fn = eval_fn  # Function to evaluate observations and return actions.

        # ----------------------------
        # Image mode for visual feedback
        # ----------------------------
        self.image_mode = image_mode
        self.recorded_images = []  # Store rendered images when image_mode is enabled

    def set_task(self, eval_configure, noise_scale=None, metadata=None):
        """
        Send a task-setting command to the remote server to update the evaluation configuration.
        
        Args:
            eval_configure (list): A list of dictionaries specifying new evaluation configurations.
                                For example:
                                [
                                    {'data_name': 'mbls'},
                                    {'data_name': 'mktl'},
                                ]
            noise_scale (float, optional): Scale of Gaussian noise to add to observations.
                                         If None, no noise is added.
        Returns:
            The response from the server (new initial observations) if available.
        """
        msg = {"set_task": eval_configure}
        if noise_scale is not None:
            msg["noise_scale"] = noise_scale
        if metadata is not None:
            msg["metadata"] = metadata
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        if response is not None:
            self.logger.info("Task settings updated on server.")
            # Save the current evaluation tasks for later use in evaluate().
            self.current_eval_tasks = eval_configure
            return response
        else:
            self.logger.warning("No response received for task setting.")
            return None

    def set_image_mode(self, enabled):
        """
        Enable or disable image mode on the remote server.

        Args:
            enabled (bool): Whether to enable image mode

        Returns:
            The response from the server confirming the image mode setting.
        """
        self.image_mode = enabled
        msg = {"set_image_mode": enabled}
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        if response is not None:
            self.logger.info(f"Image mode set to: {enabled}")
            return response
        else:
            self.logger.warning("No response received for image mode setting.")
            return None

    def get_images(self):
        """
        Get the list of recorded images.

        Returns:
            List of numpy arrays containing rendered images.
        """
        return self.recorded_images

    def clear_images(self):
        """
        Clear the recorded images list.
        """
        self.recorded_images = []
        self.logger.info("Recorded images cleared.")

    def evaluate(self, num_episodes=3, max_steps=280):
        """
        Evaluate the remote multi-environment for the specified number of episodes.
        Each episode runs for at most max_steps. If evaluation tasks have been set,
        this function tracks and returns rewards per task.
        """
        import numpy as np
        from tqdm import tqdm
        from heapq import nsmallest

        init_response = recv_pickle(self.sock)
        if init_response is None:
            self.logger.error("No data received. Connection closed.")
            return

        # Store initial rendered image if image_mode is enabled
        if self.image_mode and "image" in init_response:
            image_data = init_response["image"]
            image = np.array(image_data, dtype=np.uint8)
            self.recorded_images.append(image)

        current_response = init_response
        num_envs = len(current_response.get("observations", []))

        tasks_active = hasattr(self, 'current_eval_tasks') and self.current_eval_tasks is not None
        if tasks_active:
            task_names = [task.get("data_name", f"task_{i}") for i, task in enumerate(self.current_eval_tasks)]
            eval_dict = {task_name: [] for task_name in task_names}
        else:
            eval_dict = {"default": []}

        eval_rewards = []
        eval_fn_times_ms = []

        for ep in range(num_episodes):
            if tasks_active:
                cumulative_rewards_dict = {task_name: 0.0 for task_name in task_names}
            else:
                cumulative_reward = 0.0

            pbar = tqdm(range(max_steps), desc=f"Episode {ep+1}", postfix={"reward": "0.00"})

            for step in pbar:
                observations = np.array(current_response.get("observations", []))
                if observations.ndim == 1:
                    observations = np.expand_dims(observations, axis=0)
                if self.obs_helper is not None:
                    observations = self.obs_helper.get_state(observations)

                # --- Timing eval_fn ---
                start_time = time.time()
                actions = self.eval_fn(observations)
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000  # Convert to ms
                eval_fn_times_ms.append(elapsed_ms)
                # ----------------------

                actions = np.array(actions).tolist()
                msg = {"action": actions}
                send_pickle(self.sock, msg)
                response = recv_pickle(self.sock)
                if response is None:
                    pbar.close()
                    self.logger.error("Connection closed by server.")
                    return

                # Store rendered image if image_mode is enabled and image is in response
                if self.image_mode and "image" in response:
                    image_data = response["image"]
                    # Convert back to numpy array
                    image = np.array(image_data, dtype=np.uint8)
                    self.recorded_images.append(image)

                raw_rewards = response.get("rewards", None)
                if tasks_active:
                    if isinstance(raw_rewards, dict):
                        for task_name in task_names:
                            task_reward_list = raw_rewards.get(task_name, [0.0] * num_envs)
                            cumulative_rewards_dict[task_name] += np.mean(task_reward_list)
                    else:
                        reward_val = np.mean(raw_rewards) if raw_rewards is not None else 0.0
                        for task_name in task_names:
                            cumulative_rewards_dict[task_name] += reward_val
                    postfix_str = {task_name: f"{cumulative_rewards_dict[task_name]:.2f}" for task_name in task_names}
                    pbar.set_postfix(postfix_str)
                else:
                    rewards = np.array(raw_rewards if raw_rewards is not None else [0.0] * num_envs)
                    cumulative_reward += np.mean(rewards)
                    pbar.set_postfix({"total_reward": f"{cumulative_reward:.2f}"})
                current_response = response
                done_flags = current_response.get("done", [False] * num_envs)
                if all(done_flags):
                    break
            pbar.close()

            if tasks_active:
                global_reward = np.mean(list(cumulative_rewards_dict.values()))
                eval_rewards.append(global_reward)
                for task_name in task_names:
                    self.logger.info(f"Episode {ep+1} for task '{task_name}' finished with mean reward: {cumulative_rewards_dict[task_name]:.2f}")
                    eval_dict[task_name].append(cumulative_rewards_dict[task_name])
                self.logger.info(f"Episode {ep+1} finished with global mean reward: {global_reward:.2f}")
            else:
                self.logger.info(f"Episode {ep+1} finished with total mean reward: {cumulative_reward:.2f}")
                eval_rewards.append(cumulative_reward)
                eval_dict["default"].append(cumulative_reward)

            send_pickle(self.sock, {"reset": True})
            current_response = recv_pickle(self.sock)

            # Reset action chunk buffer at episode boundary
            if self.chunk_helper is not None:
                self.chunk_helper.reset()

            # Store rendered image from reset response if image_mode is enabled
            if self.image_mode and current_response and "image" in current_response:
                image_data = current_response["image"]
                image = np.array(image_data, dtype=np.uint8)
                self.recorded_images.append(image)

            time.sleep(0.1)

        if tasks_active:
            for task_name in task_names:
                task_rewards = eval_dict[task_name]
                avg_reward = np.mean(task_rewards) if task_rewards else 0.0
                eval_dict[task_name] = {
                    "episode_rewards": [round(r, 3) for r in task_rewards],
                    "avg_reward": round(avg_reward, 3)
                }
        else:
            default_rewards = eval_dict["default"]
            avg_reward = np.mean(default_rewards) if default_rewards else 0.0
            eval_dict["default"] = {
                "episode_rewards": [round(r, 3) for r in default_rewards],
                "avg_reward": round(avg_reward, 3)
            }

        # Compute stats for top 99% fastest eval_fn calls
        top_99 = nsmallest(int(len(eval_fn_times_ms) * 0.999), eval_fn_times_ms)
        mean_99 = round(np.mean(top_99), 3)
        var_99 = round(np.var(top_99), 3)
        self.logger.info(f"[eval_fn timing] Fastest 99% - mean: {mean_99} ms, variance: {var_99} ms²")
        return eval_rewards, eval_dict


    def close(self):
        self.sock.close()
        self.logger.info("Connection closed.")

class KitchenEmbedRemoteEvaluator(KitchenRemoteEvaluator):
    DEFAULT_MODEL_NAME = "ViT-B/16"
    DEFAULT_CAMERA_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "agentview_rgb_dinov3",
            (
                "agentview_image",
                "image",
                "render",
            ),
        ),
        (
            "eye_in_hand_rgb_dinov3",
            (
                "eye_in_hand_image",
                "robot0_eye_in_hand_image",
            ),
        ),
    )
    # Match LeRobotDataLoader ordering to ensure policy input consistency:
    # [agentview_rgb_dinov3, eye_in_hand_rgb_dinov3, joint_states, ee_states, gripper_states, robot_states]
    # NOTE NOTE NOTE Sequence Really matters.
    DEFAULT_PROPRIO_KEYS: tuple[str, ...] = (
        "joint_states",
        "ee_states",
        "gripper_states",
        "robot_states",
    )
    DEFAULT_PROPRIO_DIMS: dict[str, int] = {
        "robot_states": 9,
        "joint_states": 7,
        "ee_states": 6,
        "gripper_states": 2,
        "object_states": 0,
    }

    def __init__(
        self,
        host=remote_server_ip,
        port=9999,
        obs_helper=None,
        eval_fn=None,
        image_mode=False,
        *,
        embedder: typing.Optional["DINOv3Embedder"] = None,
        embedder_kwargs: dict[str, typing.Any] | None = None,
        model_name: str | None = None,
        embed_batch_size: int = 2,
        camera_aliases: dict[str, typing.Iterable[str]] | None = None,
        proprio_keys: tuple[str, ...] | None = None,
        embed_image_size: int | tuple[int, int] | None = 224,
        verbose_obs: bool = False,
        use_oracle_proprio: bool = False,
        oracle_key_name: str = "state",
        action_chunk_size: int = 1,
    ):
        super().__init__(
            host=host,
            port=port,
            obs_helper=obs_helper,
            eval_fn=eval_fn,
            image_mode=image_mode,
            action_chunk_size=action_chunk_size,
        )
        self.verbose_obs = verbose_obs
        self.embed_image_size = embed_image_size
        self.embed_batch_size = max(1, int(embed_batch_size))

        if embedder is None:
            kwargs = dict(embedder_kwargs or {})
            kwargs.setdefault("model_name", model_name or self.DEFAULT_MODEL_NAME)
            if isinstance(embed_image_size, int):
                kwargs.setdefault("image_size", int(embed_image_size))
            from SILGym.utils.image_embedding import DINOv3Embedder as _DINOv3Embedder

            self.embedder = _DINOv3Embedder(**kwargs)
        else:
            self.embedder = embedder

        self._embed_dim = int(getattr(self.embedder, "embed_dim", 0))
        if self._embed_dim <= 0:
            raise ValueError("Invalid embedding dimension returned by DINOv3 embedder.")

        # Resolve canonical modality order from LeRobotDataLoader, falling back to defaults.
        try:
            from SILGym.dataset.dataloader import LeRobotDataLoader as _LRDL  # lazy import to avoid cycles
            canonical_camera_order = tuple(_LRDL.get_camera_embed_keys())
            canonical_proprio_order = tuple(_LRDL.get_proprio_keys())
        except Exception:
            canonical_camera_order = tuple(k for k, _ in self.DEFAULT_CAMERA_ALIASES)
            canonical_proprio_order = self.DEFAULT_PROPRIO_KEYS

        # Build camera alias mapping honoring dataloader order unless overridden
        if camera_aliases is None:
            default_alias_map = {k: tuple(v) for k, v in self.DEFAULT_CAMERA_ALIASES}
            ordered_items: list[tuple[str, tuple[str, ...]]] = []
            for key in canonical_camera_order:
                ordered_items.append((key, default_alias_map.get(key, (key,))))
            alias_items = ordered_items
        else:
            # Caller provided explicit alias mapping; preserve its order
            alias_items = [(key, tuple(values)) for key, values in camera_aliases.items()]

        self._camera_aliases: dict[str, tuple[str, ...]] = {key: vals for key, vals in alias_items}
        self._camera_keys: tuple[str, ...] = tuple(self._camera_aliases.keys())

        # Proprioceptive feature concatenation order
        self._use_oracle_proprio: bool = bool(use_oracle_proprio)
        self._oracle_key_name: str = str(oracle_key_name)
        if proprio_keys is not None:
            self.proprio_feature_order: tuple[str, ...] = tuple(proprio_keys)
        else:
            if self._use_oracle_proprio:
                self.proprio_feature_order = ("oracle_state",)
            else:
                self.proprio_feature_order = canonical_proprio_order
        self._proprio_dims: dict[str, int] = dict(self.DEFAULT_PROPRIO_DIMS)
        # Placeholder for oracle proprio dimension; set dynamically on first observation
        self._proprio_dims.setdefault("oracle_state", 0)

        self._camera_embed_dims: dict[str, int] = {
            key: self._embed_dim for key in self._camera_keys
        }
        self._expected_obs_dim: int | None = None
        self._refresh_expected_obs_dim()

        self._embedding_enabled = False

    def _refresh_expected_obs_dim(self) -> None:
        camera_dim = sum(self._camera_embed_dims.get(k, self._embed_dim) for k in self._camera_keys)
        proprio_dim = 0
        for key in self.proprio_feature_order:
            dim = self._proprio_dims.get(key, 0)
            if dim > 0:
                proprio_dim += dim
        self._expected_obs_dim = camera_dim + proprio_dim

    def _format_image_size(self):
        size = self.embed_image_size
        if size is None:
            return None
        if isinstance(size, (tuple, list)) and len(size) == 2:
            return [int(size[0]), int(size[1])]
        if isinstance(size, int):
            return int(size)
        return None

    def _ensure_embedding_enabled(self, fallback_response):
        if self._embedding_enabled:
            return fallback_response
        request = {"set_embedding_mode": True}
        size_payload = self._format_image_size()
        if size_payload is not None:
            request["image_size"] = size_payload
        send_pickle(self.sock, request)
        response = recv_pickle(self.sock)
        if response is None:
            raise RuntimeError("Failed to enable kitchen embedding mode on remote server.")
        self._embedding_enabled = True
        return response

    def _extract_camera_image(
        self,
        obs: typing.Mapping[str, typing.Any],
        canonical_key: str,
    ) -> np.ndarray | None:
        for candidate in self._camera_aliases.get(canonical_key, ()):
            if candidate not in obs:
                continue
            value = obs[candidate]
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 3:
                if arr.ndim == 4:
                    arr = arr[0]
                return arr
        return None

    def _extract_proprio_features(
        self,
        obs: typing.Mapping[str, typing.Any],
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
        # If using oracle proprio, prefer the configured key but tolerate common fallbacks
        if self._use_oracle_proprio:
            candidate_keys = [self._oracle_key_name]
            for fallback_key in ("states", "state", "oracle_state", "oracle_proprio", "proprio_state", "oracle"):
                if fallback_key not in candidate_keys:
                    candidate_keys.append(fallback_key)
            for key in candidate_keys:
                if key in obs and obs[key] is not None:
                    oracle = np.asarray(obs.get(key), dtype=np.float32).reshape(-1)
                    self._proprio_dims["oracle_state"] = int(oracle.size)
                    if verbose:
                        self.logger.info(f"Oracle proprio ({key}) shape={oracle.shape}")
                    return {"oracle_state": oracle}
            if verbose:
                self.logger.warning(
                    "Oracle proprio key not present in observation. "
                    f"Searched keys={candidate_keys}; available keys={sorted(obs.keys())}"
                )

        features: dict[str, np.ndarray] = {}

        joint_source = obs.get("robot0_joint_pos")
        if joint_source is not None:
            joint_states = np.asarray(joint_source, dtype=np.float32).reshape(-1)[:7]
            features["joint_states"] = joint_states
        else:
            proprio_all = obs.get("robot0_proprio-state")
            if proprio_all is not None:
                proprio_arr = np.asarray(proprio_all, dtype=np.float32).reshape(-1)
                if proprio_arr.size >= 7:
                    features["joint_states"] = proprio_arr[:7]

        robot_states = obs.get("robot_states")
        if robot_states is not None:
            features["robot_states"] = np.asarray(robot_states, dtype=np.float32).reshape(-1)

        eef_pos = obs.get("robot0_eef_pos")
        eef_quat = obs.get("robot0_eef_quat")
        if eef_pos is not None and eef_quat is not None:
            pos = np.asarray(eef_pos, dtype=np.float32).reshape(-1)[:3]
            quat = np.asarray(eef_quat, dtype=np.float32).reshape(-1)
            features["ee_states"] = np.concatenate([pos, quat[:3]])
        elif "robot0_proprio-state" in obs:
            proprio_arr = np.asarray(obs["robot0_proprio-state"], dtype=np.float32).reshape(-1)
            if proprio_arr.size >= 13 and "ee_states" not in features:
                features["ee_states"] = proprio_arr[7:13]

        gripper = obs.get("robot0_gripper_qpos")
        if gripper is not None:
            features["gripper_states"] = np.asarray(gripper, dtype=np.float32).reshape(-1)[:2]
        elif "robot0_proprio-state" in obs:
            proprio_arr = np.asarray(obs["robot0_proprio-state"], dtype=np.float32).reshape(-1)
            if proprio_arr.size >= 15 and "gripper_states" not in features:
                features["gripper_states"] = proprio_arr[13:15]

        object_state = obs.get("object-state")
        if object_state is not None:
            features["object_states"] = np.asarray(object_state, dtype=np.float32).reshape(-1)

        if verbose:
            for key, value in features.items():
                self.logger.info(f"Proprio feature {key}: shape={value.shape}")

        return features

    def _prepare_state_obs(
        self,
        obs: typing.Mapping[str, typing.Any] | None,
        verbose: bool = False,
    ) -> np.ndarray:
        target_dim = self._expected_obs_dim or 0
        if verbose:
            self.logger.info("=" * 60)
            self.logger.info("KitchenEmbedRemoteEvaluator._prepare_state_obs - Starting")
            self.logger.info(f"Expected obs dimension: {self._expected_obs_dim}")
            self.logger.info(f"Camera keys: {self._camera_keys}")
            self.logger.info(f"Proprio keys: {self.proprio_feature_order}")

        if obs is None or not isinstance(obs, dict):
            if verbose:
                self.logger.warning(f"Observation is None or not dict: {type(obs)}")
            return np.zeros(target_dim, dtype=np.float32)

        features: list[np.ndarray | None] = []
        pending_embeddings: list[np.ndarray] = []
        pending_slots: list[tuple[int, str]] = []
        dirty_expected = False

        for canonical_key in self._camera_keys:
            if canonical_key in obs and obs[canonical_key] is not None:
                vec = np.asarray(obs[canonical_key], dtype=np.float32).reshape(-1)
                features.append(vec)
                self._camera_embed_dims[canonical_key] = vec.size
                dirty_expected = True
                if verbose:
                    self.logger.info(f"Using precomputed embedding for {canonical_key} (dim={vec.size})")
                continue

            image = self._extract_camera_image(obs, canonical_key)
            if image is None:
                dim = self._camera_embed_dims.get(canonical_key, self._embed_dim)
                features.append(np.zeros(dim, dtype=np.float32))
                if verbose:
                    self.logger.warning(f"No image found for {canonical_key}, using zeros({dim})")
                continue

            slot_index = len(features)
            features.append(None)
            pending_embeddings.append(image)
            pending_slots.append((slot_index, canonical_key))
            if verbose:
                self.logger.info(f"Queued image for {canonical_key}: shape={image.shape}")

        if pending_embeddings:
            embeddings = self.embedder(
                pending_embeddings,
                mode="global",
                batch_size=min(self.embed_batch_size, len(pending_embeddings)),
                return_numpy=True,
            )
            for (slot_index, canonical_key), embedding in zip(pending_slots, embeddings):
                vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
                features[slot_index] = vec
                self._camera_embed_dims[canonical_key] = vec.size
                dirty_expected = True
                if verbose:
                    self.logger.info(
                        f"Embedded {canonical_key}: shape={vec.shape}, min={vec.min():.4f}, max={vec.max():.4f}, mean={vec.mean():.4f}"
                    )

        for idx, value in enumerate(features):
            if value is None:
                canonical_key = self._camera_keys[idx]
                dim = self._camera_embed_dims.get(canonical_key, self._embed_dim)
                features[idx] = np.zeros(dim, dtype=np.float32)
                if verbose:
                    self.logger.warning(f"Filled missing embedding for {canonical_key} with zeros({dim})")

        proprio_features = self._extract_proprio_features(obs, verbose=verbose)
        for key in self.proprio_feature_order:
            if key in proprio_features:
                arr = proprio_features[key]
                features.append(arr)
                self._proprio_dims[key] = arr.size
                dirty_expected = True
                if verbose:
                    values_str = f"{arr[:5]}..." if arr.size > 5 else arr
                    self.logger.info(f"Added proprio {key}: shape={arr.shape}, values={values_str}")
            else:
                dim = self._proprio_dims.get(key, 0)
                if dim > 0:
                    features.append(np.zeros(dim, dtype=np.float32))
                    if verbose:
                        self.logger.info(f"Missing proprio {key}, using zeros({dim})")

        if dirty_expected:
            self._refresh_expected_obs_dim()

        if features:
            flat = np.concatenate(features, axis=0).astype(np.float32, copy=False)
        else:
            flat = np.zeros(0, dtype=np.float32)

        target = self._expected_obs_dim or flat.size
        if target and flat.size < target:
            flat = np.pad(flat, (0, target - flat.size))
            raise ValueError(f"_expected_obs_dim")
        return flat

    def _prepare_batch(
        self,
        observation_batch: typing.Any,
        verbose: bool = False,
    ) -> np.ndarray:
        if isinstance(observation_batch, np.ndarray):
            observation_batch = observation_batch.tolist()
        if not isinstance(observation_batch, (list, tuple)):
            observation_batch = [observation_batch]
        processed = [
            self._prepare_state_obs(obs, verbose=verbose)
            for obs in observation_batch
        ]
        if not processed:
            return np.zeros((0, self._expected_obs_dim or 0), dtype=np.float32)
        return np.stack(processed, axis=0)

    def evaluate(self, num_episodes=3, max_steps=280):
        import numpy as np
        from tqdm import tqdm
        from heapq import nsmallest

        init_response = recv_pickle(self.sock)
        if init_response is None:
            self.logger.error("No data received. Connection closed.")
            return [], {}

        init_response = self._ensure_embedding_enabled(init_response)

        if self.image_mode and "image" in init_response:
            image = np.array(init_response["image"], dtype=np.uint8)
            self.recorded_images.append(image)

        current_response = init_response
        obs_field = current_response.get("observations", [])
        if isinstance(obs_field, (list, tuple)):
            num_envs = len(obs_field)
        elif isinstance(obs_field, np.ndarray):
            num_envs = obs_field.shape[0] if obs_field.ndim > 1 else 1
        else:
            num_envs = 1

        tasks_active = hasattr(self, 'current_eval_tasks') and self.current_eval_tasks is not None
        if tasks_active:
            task_names = [task.get("data_name", f"task_{i}") for i, task in enumerate(self.current_eval_tasks)]
            eval_dict = {task_name: [] for task_name in task_names}
        else:
            task_names = ["default"]
            eval_dict = {"default": []}

        eval_rewards = []
        eval_fn_times_ms = []

        for ep in range(num_episodes):
            if tasks_active:
                cumulative_rewards_dict = {task_name: 0.0 for task_name in task_names}
            else:
                cumulative_reward = 0.0

            pbar = tqdm(range(max_steps), desc=f"Episode {ep+1}", postfix={"reward": "0.00"})

            for step in pbar:
                raw_observations = current_response.get("observations", [])
                observations = self._prepare_batch(raw_observations, verbose=self.verbose_obs)

                if self.obs_helper is not None:
                    observations = self.obs_helper.get_state(observations)

                if self.eval_fn is None:
                    raise RuntimeError("No evaluation function provided for KitchenEmbedRemoteEvaluator.")

                start_time = time.time()
                actions = self.eval_fn(observations)
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000.0
                eval_fn_times_ms.append(elapsed_ms)

                actions = np.asarray(actions)
                if actions.ndim == 1:
                    actions = actions[np.newaxis, :]
                msg = {"action": actions.tolist()}
                send_pickle(self.sock, msg)
                response = recv_pickle(self.sock)
                if response is None:
                    pbar.close()
                    self.logger.error("Connection closed by server.")
                    return eval_rewards, eval_dict

                if self.image_mode and "image" in response:
                    image = np.array(response["image"], dtype=np.uint8)
                    self.recorded_images.append(image)

                raw_rewards = response.get("rewards", None)
                if tasks_active:
                    if isinstance(raw_rewards, dict):
                        for task_name in task_names:
                            task_reward_list = raw_rewards.get(task_name, [0.0] * num_envs)
                            cumulative_rewards_dict[task_name] += float(np.mean(task_reward_list))
                    else:
                        reward_val = float(np.mean(raw_rewards)) if raw_rewards is not None else 0.0
                        for task_name in task_names:
                            cumulative_rewards_dict[task_name] += reward_val
                    postfix_str = {task_name: f"{cumulative_rewards_dict[task_name]:.2f}" for task_name in task_names}
                    pbar.set_postfix(postfix_str)
                else:
                    rewards = np.array(raw_rewards if raw_rewards is not None else [0.0] * num_envs)
                    cumulative_reward += float(np.mean(rewards))
                    pbar.set_postfix({"total_reward": f"{cumulative_reward:.2f}"})

                current_response = response
                done_flags = current_response.get("done", [False] * num_envs)
                if all(done_flags):
                    break
            pbar.close()

            if tasks_active:
                global_reward = float(np.mean(list(cumulative_rewards_dict.values())))
                eval_rewards.append(global_reward)
                for task_name in task_names:
                    task_reward = cumulative_rewards_dict[task_name]
                    self.logger.info(
                        f"Episode {ep+1} for task '{task_name}' finished with mean reward: {task_reward:.2f}"
                    )
                    eval_dict[task_name].append(task_reward)
                self.logger.info(f"Episode {ep+1} finished with global mean reward: {global_reward:.2f}")
            else:
                self.logger.info(f"Episode {ep+1} finished with total mean reward: {cumulative_reward:.2f}")
                eval_rewards.append(cumulative_reward)
                eval_dict["default"].append(cumulative_reward)

            send_pickle(self.sock, {"reset": True})
            current_response = recv_pickle(self.sock)

            if self.image_mode and current_response and "image" in current_response:
                image = np.array(current_response["image"], dtype=np.uint8)
                self.recorded_images.append(image)

            time.sleep(0.1)

        if tasks_active:
            for task_name in task_names:
                task_rewards = eval_dict[task_name]
                avg_reward = np.mean(task_rewards) if task_rewards else 0.0
                eval_dict[task_name] = {
                    "episode_rewards": [round(r, 3) for r in task_rewards],
                    "avg_reward": round(float(avg_reward), 3),
                }
        else:
            default_rewards = eval_dict["default"]
            avg_reward = np.mean(default_rewards) if default_rewards else 0.0
            eval_dict["default"] = {
                "episode_rewards": [round(r, 3) for r in default_rewards],
                "avg_reward": round(float(avg_reward), 3),
            }

        if eval_fn_times_ms:
            top_99 = nsmallest(max(1, int(len(eval_fn_times_ms) * 0.999)), eval_fn_times_ms)
            mean_99 = round(float(np.mean(top_99)), 3)
            var_99 = round(float(np.var(top_99)), 3)
            self.logger.info(f"[eval_fn timing] Fastest 99% - mean: {mean_99} ms, variance: {var_99} ms²")

        return eval_rewards, eval_dict


class KitchenStudioEmbedRemoteEvaluator(KitchenEmbedRemoteEvaluator):
    """
    Kitchen evaluator with multi-camera studio rendering support.

    Extends KitchenEmbedRemoteEvaluator to handle 5 camera views:
    - agentview (overview)
    - wrist (eye-in-hand)
    - ovens (microwave/kettle area)
    - pannels (light switch area)
    - cabinets (cabinet area)

    Camera images are rendered server-side using render_studio() and embedded
    client-side using DINOv3.
    """

    DEFAULT_STUDIO_CAMERA_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "agentview_rgb_dinov3",
            (
                "agentview_image",
                "overview",
            ),
        ),
        (
            "eye_in_hand_rgb_dinov3",
            (
                "eye_in_hand_image",
                "wrist",
            ),
        ),
        (
            "ovens_rgb_dinov3",
            (
                "ovens_image",
                "ovens",
            ),
        ),
        (
            "pannels_rgb_dinov3",
            (
                "pannels_image",
                "pannels",
            ),
        ),
        (
            "cabinets_rgb_dinov3",
            (
                "cabinets_image",
                "cabinets",
            ),
        ),
    )

    def __init__(
        self,
        host=remote_server_ip,
        port=9999,
        obs_helper=None,
        eval_fn=None,
        image_mode=False,
        *,
        embedder: typing.Optional["DINOv3Embedder"] = None,
        embedder_kwargs: dict[str, typing.Any] | None = None,
        model_name: str | None = None,
        embed_batch_size: int = 5,  # Increased for 5 cameras
        camera_keys: tuple[str, ...] | None = None,
        camera_aliases: dict[str, typing.Iterable[str]] | None = None,
        proprio_keys: tuple[str, ...] | None = None,
        embed_image_size: int | tuple[int, int] | None = 224,
        verbose_obs: bool = False,
        use_oracle_proprio: bool = False,
        oracle_key_name: str = "state",
        action_chunk_size: int = 1,
    ):
        """
        Initialize KitchenStudioEmbedRemoteEvaluator with multi-camera support.

        Args:
            camera_keys: Tuple of camera keys to use. Defaults to all 5 studio cameras.
            Other args same as KitchenEmbedRemoteEvaluator.
        """
        # Use studio camera aliases by default
        if camera_aliases is None:
            camera_aliases = {key: tuple(values) for key, values in self.DEFAULT_STUDIO_CAMERA_ALIASES}

        # Use all 5 cameras by default if not specified
        if camera_keys is None:
            from SILGym.config.kitchen_scenario import DEFAULT_KITCHENSTUDIO_EMBED_CAMERA_KEYS
            camera_keys = DEFAULT_KITCHENSTUDIO_EMBED_CAMERA_KEYS

        # Initialize parent with studio configuration
        super().__init__(
            host=host,
            port=port,
            obs_helper=obs_helper,
            eval_fn=eval_fn,
            image_mode=image_mode,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            camera_aliases=camera_aliases,
            proprio_keys=proprio_keys,
            embed_image_size=embed_image_size,
            verbose_obs=verbose_obs,
            use_oracle_proprio=use_oracle_proprio,
            oracle_key_name=oracle_key_name,
            action_chunk_size=action_chunk_size,
        )

        # Enable studio mode on server
        self._studio_mode_enabled = False
        self.logger.info(
            f"KitchenStudioEmbedRemoteEvaluator initialized with {len(self._camera_keys)} cameras: "
            f"{', '.join(self._camera_keys)}"
        )

    def _ensure_studio_mode_enabled(self):
        """Enable studio mode on the remote server if not already enabled."""
        if self._studio_mode_enabled:
            return
        msg = {"set_studio_mode": True}
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        if response is None:
            raise RuntimeError("Failed to enable studio mode on remote server.")
        self._studio_mode_enabled = True
        self.logger.info("Studio mode enabled on remote server.")

    def _ensure_embedding_enabled(self, fallback_response):
        """Override to enable both embedding and studio modes."""
        # First enable embedding mode (parent implementation)
        response = super()._ensure_embedding_enabled(fallback_response)
        # Then enable studio mode
        self._ensure_studio_mode_enabled()
        return response


class MMWorldRemoteEvaluator:
    def __init__(self, host=remote_server_ip, port=8888, obs_helper=None, eval_fn=None, action_chunk_size=1):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        self.logger = get_logger(__name__)
        self.logger.info(f'Connected to mmworld server at {host}:{port}')
        self.obs_helper = obs_helper

        # Wrap eval_fn with ActionChunkHelper if chunking is enabled
        self.action_chunk_size = int(action_chunk_size)
        if self.action_chunk_size > 1 and eval_fn is not None:
            self.chunk_helper = ActionChunkHelper(
                eval_fn=eval_fn,
                chunk_size=self.action_chunk_size,
                action_dim=4  # MMWorld action dimension
            )
            self.eval_fn = self.chunk_helper.get_action
        else:
            self.chunk_helper = None
            self.eval_fn = eval_fn

    def set_task(self, eval_configure, metadata=None):
        """
        Send a task-setting command to the remote server to update the evaluation configuration.
        
        Args:
            eval_configure (list): A list of dictionaries specifying new evaluation configurations.
                                For example:
                                [
                                    {'data_name': 'door-button-drawer-puck'},
                                ]
        Returns:
            The response from the server (new initial observations) if available.
        """
        msg = {"set_task": eval_configure}
        if metadata is not None:
            msg["metadata"] = metadata
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        if response is not None:
            self.logger.info("Task settings updated on server.")
            # Save the current evaluation tasks for later use in evaluate().
            self.current_eval_tasks = eval_configure
            return response
        else:
            self.logger.warning("No response received for task setting.")
            return None

    def evaluate(self, num_episodes: int = 3, max_steps: int = 600):
        """Run *num_episodes* episodes and gather rewards.

        Returns
        -------
        tuple[list[float], dict[str, dict]]
            (eval_rewards, eval_dict) following the format used in
            *KitchenRemoteEvaluator.evaluate*.
        """

        # -----------------------------------------------------------------
        # Initial observation from server
        # -----------------------------------------------------------------
        response = recv_pickle(self.sock)
        if response is None:
            self.logger.error("No data received. Connection closed.")
            return [], {}
        current_obs = response.get("observation")

        # -----------------------------------------------------------------
        # Task bookkeeping
        # -----------------------------------------------------------------
        tasks_active = hasattr(self, "current_eval_tasks") and self.current_eval_tasks  # type: ignore[attr-defined]
        if tasks_active:
            task_names = [task.get("data_name", f"task_{i}") for i, task in enumerate(self.current_eval_tasks)]  # type: ignore[attr-defined]
            eval_dict: dict[str, list[float] | dict] = {name: [] for name in task_names}
        else:
            task_names = ["default"]
            eval_dict = {"default": []}

        eval_rewards: list[float] = []  # Global mean reward per episode

        # -----------------------------------------------------------------
        # Main episode loop
        # -----------------------------------------------------------------
        for ep in range(num_episodes):
            if tasks_active:
                cumulative_rewards: dict[str, float] = {name: 0.0 for name in task_names}
            else:
                cumulative_reward = 0.0

            pbar = tqdm(range(max_steps), desc=f"Episode {ep + 1}", leave=False)

            for _ in pbar:
                # Model inference ------------------------------------------------
                if self.obs_helper is not None:
                    current_obs = self.obs_helper.get_state(np.array(current_obs))
                action = self.eval_fn(current_obs)  # Assume model handles shape

                # Convert to list for pickle serialization consistency
                if isinstance(action, np.ndarray):
                    action = action.tolist()

                send_pickle(self.sock, {"action": action})
                response = recv_pickle(self.sock)
                if response is None:
                    self.logger.error("Connection closed by server.")
                    return eval_rewards, eval_dict  # Early exit

                # ----------------------------------------------------------------
                # Reward extraction ---------------------------------------------
                # ----------------------------------------------------------------
                reward_val = response.get("reward")
                if reward_val is None:
                    # Fallback if server uses plural key
                    reward_val = response.get("rewards")
                if isinstance(reward_val, (list, tuple, np.ndarray)):
                    reward_val = float(np.mean(reward_val))
                reward_val = float(reward_val if reward_val is not None else 0.0)

                if tasks_active:
                    for task_name in task_names:
                        cumulative_rewards[task_name] += reward_val
                    pbar.set_postfix({name: f"{cumulative_rewards[name]:.2f}" for name in task_names})
                else:
                    cumulative_reward += reward_val
                    pbar.set_postfix({"reward": f"{cumulative_reward:.2f}"})

                # ----------------------------------------------------------------
                done_flag = response.get("done", False)
                current_obs = response.get("observation")
                if done_flag:
                    break

            pbar.close()

            # --------------------------------------------------------------------
            # Episode summarization ---------------------------------------------
            # --------------------------------------------------------------------
            if tasks_active:
                global_reward = float(np.mean(list(cumulative_rewards.values())))
                eval_rewards.append(global_reward)
                for name in task_names:
                    eval_dict[name].append(cumulative_rewards[name])  # type: ignore[arg-type]
                    self.logger.info(f"Episode {ep + 1} – task '{name}': {cumulative_rewards[name]:.2f}")
                self.logger.info(f"Episode {ep + 1} – global reward: {global_reward:.2f}")
            else:
                eval_rewards.append(cumulative_reward)
                eval_dict["default"].append(cumulative_reward)  # type: ignore[arg-type]
                self.logger.info(f"Episode {ep + 1} reward: {cumulative_reward:.2f}")

            # --------------------------------------------------------------------
            # Reset env for next episode ----------------------------------------
            send_pickle(self.sock, {"reset": True})
            response = recv_pickle(self.sock)

            # Reset action chunk buffer at episode boundary
            if self.chunk_helper is not None:
                self.chunk_helper.reset()

            current_obs = response.get("observation") if response else None
            time.sleep(0.05)  # Small delay for stability

        # --------------------------------------------------------------------
        # Post‑processing: average rewards, rounding -------------------------
        # --------------------------------------------------------------------
        rounded_eval_dict: dict[str, dict[str, typing.Any]] = {}
        for name in task_names:
            rewards = eval_dict[name]  # type: ignore[index]
            avg_reward = float(np.mean(rewards)) if rewards else 0.0
            rounded_eval_dict[name] = {
                "episode_rewards": [round(r, 3) for r in rewards],
                "avg_reward": round(avg_reward, 3),
            }

        return eval_rewards, rounded_eval_dict

    def close(self):
        self.sock.close()
        self.logger.info('Connection closed.')

def default_libero_eval_fn(observation):
    """
    Default evaluation function for Libero that returns random actions.

    Args:
        observation: State observation (can be batched or single)

    Returns:
        Random action as a list in the range [-1, 1] with dimension 7 (Libero action space)

    Note:
        Actions are returned as lists (not numpy arrays) to avoid numpy version
        compatibility issues when serializing with pickle across different environments.
    """
    # Libero uses 7-DOF action space (6 for end-effector + 1 for gripper)
    action_dim = 7

    # Handle both batched and single observations
    if isinstance(observation, np.ndarray):
        if observation.ndim == 1:
            # Single observation - return single action as list
            return np.random.uniform(-1, 1, size=(action_dim,)).tolist()
        else:
            # Batched observations - return batched actions as list
            batch_size = observation.shape[0]
            return np.random.uniform(-1, 1, size=(batch_size, action_dim)).tolist()
    else:
        # Fallback for non-numpy inputs
        return np.random.uniform(-1, 1, size=(action_dim,)).tolist()

class LiberoRemoteEvaluator:
    """
    LiberoRemoteEvaluator communicates with a remote server and uses
    a user-provided eval_fn to compute actions.

    Args:
        host: Server IP address (default from REMOTE_SERVER_IP env var)
        port: Server port (default 7777)
        obs_helper: Optional observation helper for state processing
        eval_fn: Function to evaluate observations and return actions
                 (defaults to random actions if not provided)
        debug: Enable verbose printing of actions, rewards, and done flags
        verbose_obs: Enable detailed logging of observation processing
                    (defaults to debug value if not specified)
    """
    supports_task_metadata: bool = True

    def __init__(
        self,
        host=remote_server_ip,
        port=7777,
        obs_helper=None,
        eval_fn=None,
        debug: bool = False,
        verbose_obs=None,
        environment_name: typing.Optional[str] = None,
        action_chunk_size: int = 1,
    ):
        self.host = host
        self.port = port
        self.debug = debug
        # Allow separate control of observation verbose logging
        # If not specified, use debug flag
        self.verbose_obs = verbose_obs if verbose_obs is not None else debug

        self.obs_helper = obs_helper

        # Use default random eval_fn if none provided
        eval_fn = eval_fn if eval_fn is not None else default_libero_eval_fn

        # Wrap eval_fn with ActionChunkHelper if chunking is enabled
        self.action_chunk_size = int(action_chunk_size)
        if self.action_chunk_size > 1:
            self.chunk_helper = ActionChunkHelper(
                eval_fn=eval_fn,
                chunk_size=self.action_chunk_size,
                action_dim=7  # Libero action dimension
            )
            self.eval_fn = self.chunk_helper.get_action
        else:
            self.chunk_helper = None
            self.eval_fn = eval_fn

        self._expected_obs_dim: int | None = 130

        # Setup socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        self.logger = get_logger(__name__)
        self.logger.info(f"Connected to Libero server at {host}:{port}")
        self.environment_name = (environment_name or "libero").strip() or "libero"
        self.current_metadata: dict[str, str] | None = None

    def _normalize_metadata(self, metadata: typing.Optional[typing.Mapping[str, typing.Any]]):
        normalized: dict[str, str] = {}
        if metadata:
            for key, value in metadata.items():
                if value is None:
                    continue
                normalized[str(key)] = str(value)
        if "environment" not in normalized and self.environment_name:
            normalized["environment"] = self.environment_name
        return normalized

    def set_task(self, eval_configure, noise_scale=None, metadata=None) :
        # benchmark_name="libero_90", task_id=0):
        """
        Optionally reset the task on the server.
        Args:
            eval_configure (list): A list of dictionaries specifying new evaluation configurations.
                For example:
                [
                    {'data_name': 'libero_90-0'},
                ]
            noise_scale: Ignored, kept for interface parity with other remote evaluators.
            metadata: Optional mapping containing phase/policy identifiers.
        """
        if len(eval_configure) == 0 :
            self.logger.error("[Error] No evaluation configuration provided.")
            return
        elif len(eval_configure) > 1 :
            self.logger.warning("[Error] Multiple evaluation configurations provided. Only the first one will be used.")
        eval_task_str = eval_configure[0].get("data_name", "libero_90-0")
        benchmark_name, task_id = eval_task_str.split("-")
        task_id = int(task_id)
        msg = {"set_task": {"benchmark_name": benchmark_name, "task_id": task_id}}
        normalized_metadata = self._normalize_metadata(metadata)
        if normalized_metadata:
            msg["metadata"] = normalized_metadata
            self.current_metadata = normalized_metadata
            if self.debug:
                self.logger.debug(f"[set_task] Sending metadata: {normalized_metadata}")
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        if self.debug:
            self.logger.debug(f"[set_task] Server response: {response}")
        return response

    def _prepare_state_obs(
        self,
        obs: typing.Mapping[str, typing.Any] | None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Convert raw observation dictionary into a flat state vector.
        Only uses the observation components that match the dataset structure.

        Args:
            obs: Observation dictionary received from the remote server.
            verbose: If True, print detailed logging information for debugging.

        Returns:
            A 1D numpy array representing the policy input state.
        """
        target_dim = self._expected_obs_dim or 0

        if verbose:
            self.logger.info("=" * 60)
            self.logger.info("_prepare_state_obs - Starting observation processing")
            self.logger.info(f"Target dimension: {target_dim}")

        if obs is None or not isinstance(obs, dict):
            if verbose:
                self.logger.warning(f"Observation is None or not dict: {type(obs)}")
            if target_dim > 0:
                return np.zeros(target_dim, dtype=np.float32)
            return np.zeros(0, dtype=np.float32)

        if verbose:
            self.logger.info(f"Observation keys available: {list(obs.keys())}")
            for key, value in obs.items():
                if value is not None:
                    if hasattr(value, 'shape'):
                        self.logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype if hasattr(value, 'dtype') else type(value)}")
                    elif isinstance(value, (list, tuple)):
                        self.logger.info(f"  {key}: len={len(value)}, type={type(value)}")
                    else:
                        self.logger.info(f"  {key}: type={type(value)}, value={value}")
                else:
                    self.logger.info(f"  {key}: None")

        components: list[np.ndarray] = []
        component_info = []  # For logging

        # Match the dataset structure: only use gripper, joint_pos, and object state
        # Order matters for compatibility
        obs_key_mapping = [
            ("robot0_joint_pos", 7),     # Joint positions (7-dim)
            ("robot0_gripper_qpos", 2),  # Gripper state (2-dim)
            ("object-state", None),       # Object state (variable dim)
        ]

        for key, expected_dim in obs_key_mapping:
            value = obs.get(key)
            if value is None:
                if verbose:
                    self.logger.info(f"Key '{key}' not found or None in observation")
                # Add zeros if we know the expected dimension
                if expected_dim:
                    components.append(np.zeros(expected_dim, dtype=np.float32))
                    if verbose:
                        self.logger.info(f"Added zeros for missing '{key}': shape=({expected_dim},)")
                continue

            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            # Truncate or pad to expected dimension if specified
            if expected_dim and arr.shape[0] != expected_dim:
                if arr.shape[0] > expected_dim:
                    arr = arr[:expected_dim]
                    if verbose:
                        self.logger.info(f"Truncated '{key}' from {arr.shape[0]} to {expected_dim}")
                else:
                    pad_size = expected_dim - arr.shape[0]
                    arr = np.concatenate([arr, np.zeros(pad_size, dtype=np.float32)])
                    if verbose:
                        self.logger.info(f"Padded '{key}' from {arr.shape[0]} to {expected_dim}")

            components.append(arr)
            if verbose:
                component_info.append((key, arr.shape[0]))
                self.logger.info(f"Added component '{key}': shape={arr.shape}, values={arr[:5]}..." if len(arr) > 5 else f"Added component '{key}': shape={arr.shape}, values={arr}")

        if components:
            state_obs = np.concatenate(components, axis=0)
            if verbose:
                self.logger.info(f"Concatenated {len(components)} components")
                self.logger.info(f"Component breakdown: {component_info}")
        else:
            state_obs = np.zeros(0, dtype=np.float32)
            if verbose:
                self.logger.warning("No components found - creating zero array")

        original_shape = state_obs.shape[0]

        if target_dim and state_obs.shape[0] < target_dim:
            pad_width = target_dim - state_obs.shape[0]
            if pad_width > 0:
                state_obs = np.concatenate(
                    (state_obs, np.zeros(pad_width, dtype=np.float32)),
                    axis=0,
                )
                if verbose:
                    self.logger.info(f"Padded observation from {original_shape} to {target_dim} (added {pad_width} zeros)")

        if verbose:
            self.logger.info(f"Final state_obs shape: {state_obs.shape}")
            self.logger.info(f"Final state_obs min/max/mean: {state_obs.min():.4f}/{state_obs.max():.4f}/{state_obs.mean():.4f}")
            self.logger.info("=" * 60)

        return state_obs.astype(np.float32, copy=False)

    def evaluate(self, num_episodes=3, max_steps=50):
        """
        Run evaluation for a given number of episodes and steps.

        Returns:
            List of cumulative rewards per episode.
        """
        init_response = recv_pickle(self.sock)
        if init_response is None:
            self.logger.error("[Error] No initial data. Connection closed.")
            return []

        current = init_response
        episode_rewards = []

        for ep in range(num_episodes):
            cumulative_reward = 0.0
            pbar = tqdm(range(max_steps), desc=f"Episode {ep+1}", leave=False)

            for step in pbar:
                obs = current.get("observation")
                reward = current.get("reward", 0.0)
                done = current.get("done", False)

                # Wrap the observations for state (NOTE hard-coded for now)
                # Use verbose logging based on verbose_obs flag
                state_obs = self._prepare_state_obs(obs, verbose=self.verbose_obs)

                if self.obs_helper is not None:
                    state_obs = self.obs_helper.get_state(state_obs)

                action = self.eval_fn(state_obs[None,:])
                # print(f"action shape: {action}")

                # Convert to list for pickle serialization consistency
                if isinstance(action, np.ndarray):
                    action = action.tolist()

                send_pickle(self.sock, {"action": action})
                response = recv_pickle(self.sock)
                if response is None:
                    pbar.close()
                    self.logger.error("[Error] Connection closed mid-episode.")
                    return episode_rewards
                current = response

                cumulative_reward += reward
                pbar.set_postfix({"reward": f"{cumulative_reward:.2f}"})

                if self.debug:
                    self.logger.debug(f"\n[Step {step}] Observation: {type(obs)}")
                    if isinstance(obs, dict):
                        for k, v in obs.items():
                            shape = getattr(v, 'shape', np.array(v).shape)
                            self.logger.debug(f"  {k}: shape {shape}")
                    self.logger.debug(f"  Action: {action}")
                    self.logger.debug(f"  Reward: {reward}, Done: {done}\n")

                if done:
                    break

            pbar.close()
            episode_rewards.append(cumulative_reward)
            self.logger.info(f"Episode {ep+1} finished with reward: {cumulative_reward:.2f}")

            # Reset for next episode
            send_pickle(self.sock, {"reset": True})
            current = recv_pickle(self.sock)

            # Reset action chunk buffer at episode boundary
            if self.chunk_helper is not None:
                self.chunk_helper.reset()

            time.sleep(0.1)

        return episode_rewards, None # current Eval_dict is not used in this case

    def close(self):
        self.sock.close()
        self.logger.info("Connection closed.")


class LiberoEmbedRemoteEvaluator(LiberoRemoteEvaluator):
    """
    Libero evaluator that replaces raw RGB observations with DINOv3 embeddings.

    The resulting flat observation matches the ordering used by LeRobotDataLoader:
    [agentview_rgb_dinov3, eye_in_hand_rgb_dinov3, joint_states, ee_states, gripper_states, ...]

    Args:
        host: Server IP address (default from REMOTE_SERVER_IP env var)
        port: Server port (default 7777)
        obs_helper: Optional observation helper for state processing
        eval_fn: Function to evaluate observations and return actions
        debug: Enable verbose printing of actions, rewards, and done flags
        verbose_obs: Enable detailed logging of observation processing and embedding
                    (defaults to debug value if not specified)
        embedder: Optional pre-initialized DINOv3Embedder instance
        embedder_kwargs: Kwargs for creating DINOv3Embedder if embedder is None
        embed_batch_size: Batch size for embedding multiple images
        camera_aliases: Mapping of canonical camera names to possible observation keys
        proprio_keys: Ordered list of proprioceptive feature keys to extract
    """

    DEFAULT_CAMERA_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "agentview_rgb_dinov3",
            (
                "agentview_image",
            ),
        ),
        (
            "eye_in_hand_rgb_dinov3",
            (
                "robot0_eye_in_hand_image",
            ),
        ),
    )
    DEFAULT_PROPRIO_KEYS: tuple[str, ...] = (
        "joint_states",    # 7-dim joint positions
        "ee_states",       # 6-dim end-effector pos+ori
        "gripper_states",  # 2-dim gripper state
    )
    DEFAULT_PROPRIO_DIMS: dict[str, int] = {
        "joint_states": 7,
        "ee_states": 6,
        "gripper_states": 2,
    }

    def __init__(
        self,
        host=remote_server_ip,
        port=7777,
        obs_helper=None,
        eval_fn=None,
        debug: bool = False,
        verbose_obs: bool | None = None,
        embedder: typing.Optional["DINOv3Embedder"] = None,
        embedder_kwargs: dict[str, typing.Any] | None = None,
        model_variant: str | None = None,
        embed_batch_size: int = 2,
        camera_aliases: dict[str, typing.Iterable[str]] | None = None,
        proprio_keys: tuple[str, ...] | None = None,
        action_chunk_size: int = 1,
    ):
        super().__init__(
            host=host,
            port=port,
            obs_helper=obs_helper,
            eval_fn=eval_fn,
            debug=debug,
            verbose_obs=verbose_obs,
            action_chunk_size=action_chunk_size,
        )

        variant_key = (model_variant or DEFAULT_LIBERO_MODEL).lower()
        if variant_key in LIBERO_ENV_MODEL_MAP:
            variant_key = LIBERO_ENV_MODEL_MAP[variant_key]
        if variant_key not in LIBERO_MODEL_TO_DINOV3:
            variant_key = DEFAULT_LIBERO_MODEL
        self.model_variant = variant_key

        if embedder is None:
            embedder_kwargs = dict(embedder_kwargs or {})
            embedder_kwargs.setdefault(
                "model_name",
                LIBERO_MODEL_TO_DINOV3.get(self.model_variant, "ViT-B/16"),
            )
            from SILGym.utils.image_embedding import DINOv3Embedder as _DINOv3Embedder

            self.embedder = _DINOv3Embedder(**embedder_kwargs)
        else:
            self.embedder = embedder

        self.embed_batch_size = max(1, int(embed_batch_size))

        if camera_aliases is None:
            alias_items = list(self.DEFAULT_CAMERA_ALIASES)
        else:
            alias_items = [(key, tuple(values)) for key, values in camera_aliases.items()]

        self._camera_aliases: dict[str, tuple[str, ...]] = {
            key: tuple(values) for key, values in alias_items
        }
        self._camera_keys: tuple[str, ...] = tuple(key for key, _ in alias_items)

        self.proprio_feature_order: tuple[str, ...] = (
            proprio_keys if proprio_keys is not None else self.DEFAULT_PROPRIO_KEYS
        )
        self._proprio_dims: dict[str, int] = dict(self.DEFAULT_PROPRIO_DIMS)

        self._embed_dim = int(self.embedder.embed_dim)
        self._camera_embed_dims: dict[str, int] = {
            key: self._embed_dim for key in self._camera_keys
        }

        self._refresh_expected_obs_dim()
        self.logger.info(
            "LiberoEmbedRemoteEvaluator ready (variant=%s, obs_dim=%d, embed_dim=%d, cameras=%s)",
            self.model_variant,
            self._expected_obs_dim or -1,
            self._embed_dim,
            ", ".join(self._camera_keys),
        )

    def _refresh_expected_obs_dim(self) -> None:
        """Recompute the total observation dimension based on known components."""
        camera_dim = sum(self._camera_embed_dims.get(k, self._embed_dim) for k in self._camera_keys)
        proprio_dim = sum(self._proprio_dims.get(k, 0) for k in self.proprio_feature_order)
        self._expected_obs_dim = camera_dim + proprio_dim

    def _extract_proprio_features(self, obs: typing.Mapping[str, typing.Any], verbose: bool = False) -> dict[str, np.ndarray | None]:
        """
        Extract proprioceptive features from observation using fallback mappings.
        Match the exact structure used in the dataset (LeRobotDataLoader).

        Args:
            obs: Observation dictionary
            verbose: If True, log extraction details

        Returns dict mapping canonical proprio names to extracted arrays.
        """
        proprio_features = {}

        if verbose:
            self.logger.info("  Extracting proprioceptive features from observation...")

        # Only extract the features that are actually in the dataset
        # Priority: joint_states, ee_states, gripper_states

        # 1. Extract joint_states (7-dim)
        if "robot0_joint_pos" in obs and obs["robot0_joint_pos"] is not None:
            proprio_features["joint_states"] = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)[:7]
            if verbose:
                self.logger.info(f"    Found joint_states from 'robot0_joint_pos' (shape={proprio_features['joint_states'].shape})")
        elif "robot0_proprio-state" in obs and obs["robot0_proprio-state"] is not None:
            # Extract from proprio state if available (first 7 dims)
            proprio_all = np.asarray(obs["robot0_proprio-state"], dtype=np.float32)
            if len(proprio_all) >= 7:
                proprio_features["joint_states"] = proprio_all[:7]
                if verbose:
                    self.logger.info(f"    Extracted joint_states from robot0_proprio-state[:7]")

        # 2. Extract ee_states (6-dim: pos + ori)
        if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
            eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]
            eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
            # Take pos (3) + first 3 components of quat for 6D representation
            proprio_features["ee_states"] = np.concatenate([eef_pos, eef_quat[:3]])
            if verbose:
                self.logger.info(f"    Constructed ee_states from 'robot0_eef_pos' + 'robot0_eef_quat[:3]' (shape={proprio_features['ee_states'].shape})")
        elif "robot0_proprio-state" in obs and obs["robot0_proprio-state"] is not None:
            # Extract from proprio state if available (dims 7-13)
            proprio_all = np.asarray(obs["robot0_proprio-state"], dtype=np.float32)
            if len(proprio_all) >= 13 and "ee_states" not in proprio_features:
                proprio_features["ee_states"] = proprio_all[7:13]
                if verbose:
                    self.logger.info(f"    Extracted ee_states from robot0_proprio-state[7:13]")

        # 3. Extract gripper_states (2-dim)
        if "robot0_gripper_qpos" in obs and obs["robot0_gripper_qpos"] is not None:
            proprio_features["gripper_states"] = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:2]
            if verbose:
                self.logger.info(f"    Found gripper_states from 'robot0_gripper_qpos' (shape={proprio_features['gripper_states'].shape})")
        elif "robot0_proprio-state" in obs and obs["robot0_proprio-state"] is not None:
            # Extract from proprio state if available (dims 13-15)
            proprio_all = np.asarray(obs["robot0_proprio-state"], dtype=np.float32)
            if len(proprio_all) >= 15 and "gripper_states" not in proprio_features:
                proprio_features["gripper_states"] = proprio_all[13:15]
                if verbose:
                    self.logger.info(f"    Extracted gripper_states from robot0_proprio-state[13:15]")

        if verbose:
            self.logger.info(f"    Extracted features: {list(proprio_features.keys())}")

        return proprio_features

    def _extract_camera_image(
        self,
        obs: typing.Mapping[str, typing.Any],
        canonical_key: str,
    ) -> np.ndarray | None:
        """Extract an RGB image for embedding."""
        for candidate in self._camera_aliases.get(canonical_key, ()):
            if candidate not in obs:
                continue
            value = obs[candidate]
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 3:
                # Handle (H, W, C) or (1, H, W, C) layouts.
                if arr.ndim == 4:
                    arr = arr[0]
                return arr
        return None

    def _prepare_state_obs(
        self,
        obs: typing.Mapping[str, typing.Any] | None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Convert raw observation dictionary into a flat state vector with DINOv3 embeddings.

        Args:
            obs: Observation dictionary received from the remote server.
            verbose: If True, print detailed logging information for debugging.

        Returns:
            A 1D numpy array with embedded images and proprioceptive features.
        """
        if verbose:
            self.logger.info("=" * 60)
            self.logger.info("LiberoEmbedRemoteEvaluator._prepare_state_obs - Starting")
            self.logger.info(f"Expected obs dimension: {self._expected_obs_dim}")
            self.logger.info(f"Camera keys: {self._camera_keys}")
            self.logger.info(f"Proprio keys: {self.proprio_feature_order}")

        if obs is None or not isinstance(obs, dict):
            if verbose:
                self.logger.warning(f"Observation is None or not dict: {type(obs)}")
            target = self._expected_obs_dim or 0
            return np.zeros(target, dtype=np.float32)

        if verbose:
            self.logger.info(f"Observation keys available: {list(obs.keys())}")
            for key, value in obs.items():
                if value is not None:
                    if hasattr(value, 'shape'):
                        self.logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype if hasattr(value, 'dtype') else type(value)}")
                    elif isinstance(value, (list, tuple)):
                        self.logger.info(f"  {key}: len={len(value)}, type={type(value)}")
                    else:
                        self.logger.info(f"  {key}: type={type(value)}")
                else:
                    self.logger.info(f"  {key}: None")

        features: list[np.ndarray | None] = []
        pending_embeddings: list[np.ndarray] = []
        pending_slots: list[tuple[int, str]] = []

        # Process camera features
        for canonical_key in self._camera_keys:
            if verbose:
                self.logger.info(f"Processing camera key: {canonical_key}")

            # Extract raw image for embedding
            image = self._extract_camera_image(obs, canonical_key)
            if image is None:
                dim = self._camera_embed_dims.get(canonical_key, self._embed_dim)
                features.append(np.zeros(dim, dtype=np.float32))
                if verbose:
                    self.logger.warning(f"  No image found for {canonical_key}, using zeros({dim})")
                continue

            # Queue for embedding
            slot_index = len(features)
            features.append(None)
            pending_embeddings.append(image)
            pending_slots.append((slot_index, canonical_key))
            if verbose:
                self.logger.info(f"  Queued image for embedding: shape={image.shape}")

        # Perform batch embedding if needed
        if pending_embeddings:
            if verbose:
                self.logger.info(f"Embedding {len(pending_embeddings)} images...")

            embeddings = self.embedder(
                pending_embeddings,
                mode="global",
                batch_size=min(self.embed_batch_size, len(pending_embeddings)),
                return_numpy=True,
            )
            for (slot_index, canonical_key), embedding in zip(pending_slots, embeddings):
                vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
                features[slot_index] = vec
                self._camera_embed_dims[canonical_key] = vec.size
                if verbose:
                    self.logger.info(f"  Embedded {canonical_key}: shape={vec.shape}, min={vec.min():.4f}, max={vec.max():.4f}, mean={vec.mean():.4f}")

        # Fill any remaining None slots with zeros (should not happen, but keep safe).
        for idx, item in enumerate(features):
            if item is None:
                dim = self._embed_dim
                features[idx] = np.zeros(dim, dtype=np.float32)
                if verbose:
                    self.logger.warning(f"  Filling None slot {idx} with zeros({dim})")

        # Append proprioceptive features in the same ordering as the dataloader.
        if verbose:
            self.logger.info("Processing proprioceptive features...")

        # Use the extraction method to get proprio features matching dataset structure
        proprio_features = self._extract_proprio_features(obs, verbose=verbose)

        # Only use the features that are in our DEFAULT_PROPRIO_KEYS
        for key in self.proprio_feature_order:
            if key in proprio_features:
                arr = proprio_features[key]
                features.append(arr)
                self._proprio_dims[key] = arr.size
                if verbose:
                    values_str = f"{arr[:5]}..." if len(arr) > 5 else str(arr)
                    self.logger.info(f"  {key}: shape={arr.shape}, values={values_str}")
            else:
                # Use zeros with known dimensions for missing features
                dim = self._proprio_dims.get(key)
                if dim:
                    features.append(np.zeros(dim, dtype=np.float32))
                    if verbose:
                        self.logger.info(f"  {key}: not found, using zeros({dim})")
                else:
                    if verbose:
                        self.logger.info(f"  {key}: skipped (unknown dimension)")
                    continue

        self._refresh_expected_obs_dim()

        if not features:
            if verbose:
                self.logger.warning("No features found - returning zero vector")
            return np.zeros(self._expected_obs_dim or 0, dtype=np.float32)

        state_obs = np.concatenate(features, axis=0).astype(np.float32, copy=False)
        original_shape = state_obs.shape[0]

        if verbose:
            self.logger.info(f"Concatenated features: shape={state_obs.shape}")

        target_dim = self._expected_obs_dim or state_obs.shape[0]
        if state_obs.shape[0] < target_dim:
            pad_width = target_dim - state_obs.shape[0]
            if pad_width > 0:
                state_obs = np.concatenate(
                    (state_obs, np.zeros(pad_width, dtype=np.float32)),
                    axis=0,
                )
                if verbose:
                    self.logger.info(f"Padded observation from {original_shape} to {target_dim}")
        elif state_obs.shape[0] > target_dim:
            state_obs = state_obs[:target_dim]
            if verbose:
                self.logger.info(f"Truncated observation from {original_shape} to {target_dim}")

        if verbose:
            self.logger.info(f"Final state_obs: shape={state_obs.shape}, min={state_obs.min():.4f}, max={state_obs.max():.4f}, mean={state_obs.mean():.4f}")
            self.logger.info("=" * 60)

        return state_obs


# Example usage:
if __name__ == '__main__':

    # Example 2: Using LiberoEmbedRemoteEvaluator with custom eval_fn
    def custom_eval_fn(observation):
        """Custom evaluation function that can process embedded observations."""
        # For embedded observations, the dimension might be different
        # This is just an example - replace with your actual policy
        action_dim = 7
        # Return as list to avoid numpy serialization issues
        return np.random.uniform(-1, 1, size=(action_dim,)).tolist()

    evaluator_embed = LiberoEmbedRemoteEvaluator(
        host='127.0.0.1',
        port=7777,
        eval_fn=custom_eval_fn,
        debug=False,
        verbose_obs=True  # Enable verbose observation processing and embedding logs
    )

    # Change the task settings remotely before evaluation
    new_eval_config = [
        {'data_name': 'libero_goal-0'},
    ]
    evaluator_embed.set_task(new_eval_config)
    evaluator_embed.evaluate(num_episodes=1, max_steps=280)
    evaluator_embed.close()
