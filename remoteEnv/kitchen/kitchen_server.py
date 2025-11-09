# .remoteEnv/kitchen/kitchen_server.py
import socket
import pickle
import struct
import numpy as np
from kitchen import *  # Assume KitchenEnv, KitchenTask, etc. are imported from kitchen module
from multiprocessing import Process
import logging
import cv2
import os
import re
import math
from datetime import datetime
from typing import Optional, List, Set, Tuple

# Setup logging with a nice format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(processName)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def recvall(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_pickle(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    sock.sendall(data)

def recv_pickle(sock):
    length_bytes = recvall(sock, 4)
    if not length_bytes:
        return None
    length = struct.unpack('!I', length_bytes)[0]
    data = recvall(sock, length)
    if not data:
        return None
    return pickle.loads(data)

def handle_client_single(conn, addr, save_video=False):
    import logging
    import numpy as np
    import cv2

    logging.info(f"Client connected from {addr}")
    print(f"Client connected from {addr}")

    # Create a single instance of the environment using KitchenEnv
    env = KitchenEnv()

    # Setup video writer if video saving is enabled
    video = None
    if save_video:
        mp4v = cv2.VideoWriter_fourcc(*'mp4v')
        first_frame = env.render()  # Get the first frame to determine video size
        vid_size = (first_frame.shape[1], first_frame.shape[0])  # (width, height)
        video_path = f"data/videos/client_{addr[0]}_{addr[1]}_video.mp4"
        video = cv2.VideoWriter(video_path, mp4v, 30, vid_size)
        bgr_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        video.write(bgr_first_frame)

    # Reset the environment and send the initial observation to the client
    init_obs = env.reset_model()
    init_obs_serialized = init_obs.tolist() if isinstance(init_obs, np.ndarray) else init_obs
    response = {
        "observations": init_obs_serialized,
        "rewards": 0.0,
        "done": False,
        "info": {}
    }
    send_pickle(conn, response)
    logging.info(f"[{addr}] Sent initial observation.")

    # Main loop for client communication
    while True:
        msg = recv_pickle(conn)
        if msg is None:
            logging.info(f"[{addr}] Connection closed by client.")
            break

        # Process reset command: reset the environment and send new observation
        if msg.get("reset", False):
            init_obs = env.reset_model()
            init_obs_serialized = init_obs.tolist() if isinstance(init_obs, np.ndarray) else init_obs
            response = {
                "observations": init_obs_serialized,
                "rewards": 0.0,
                "done": False,
                "info": {}
            }
            send_pickle(conn, response)
            logging.info(f"[{addr}] Environment reset upon client request.")
            if save_video and video is not None:
                frame = env.render()
                if frame is not None:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video.write(bgr_frame)
            continue

        # Process action command: execute an environment step based on client action
        actions = msg.get("action")
        if actions is None:
            logging.warning(f"[{addr}] Received message without action.")
            continue

        # Convert actions to numpy array and perform one step in the environment
        actions = np.array(actions)
        if actions.ndim != 1 :
            actions = actions.squeeze()
        obs, rew, done, info = env.step(actions)
        obs_serialized = obs.tolist() if isinstance(obs, np.ndarray) else obs
        response = {
            "observations": obs_serialized,
            "rewards": rew if not isinstance(rew, np.ndarray) else rew.tolist(),
            "done": [done],
            "info": [info]
        }
        send_pickle(conn, response)
        logging.info(f"[{addr}] Processed action, reward: {rew}, done: {done}")

        # If video saving is enabled, record the current frame
        if save_video and video is not None:
            frame = env.render()
            if frame is not None:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(bgr_frame)

    conn.close()
    if save_video:
        if video is not None:
            video.release()
        if video_path:
            logging.info(f"[{addr}] Client process ended. Video saved to {video_path}")
        else:
            logging.info(f"[{addr}] Client process ended. Video recording not created.")
    else:
        logging.info(f"[{addr}] Client process ended.")

def handle_client(conn, addr, save_video=False, high_res=False):
    # Initial task name (default value)
    current_task_name = "default"

    # Image mode flag for returning rendered images to client
    image_mode = False
    embedding_mode = False
    embedding_image_size = (224, 224)
    vision_mode = False
    studio_mode = False  # Flag for multi-camera studio rendering
    high_res_mode = high_res  # Flag for high-resolution rendering (720x720)
    ROBOT_QPOS_DIM = 9

    output_dir = os.path.join("data", "videos")
    mp4v = cv2.VideoWriter_fourcc(*'mp4v')
    video = None
    video_path = ""
    video_width = None
    video_height = None
    timestamp = None
    file_basename = None
    metadata_context = {
        "environment": "kitchen",
        "label_components": [],
        "label": None,
    }
    buffered_frames: List[np.ndarray] = []

    def sanitize_for_filename(value: str) -> str:
        sanitized = re.sub(r'[^0-9A-Za-z_\-]+', '_', str(value))
        sanitized = sanitized.strip('_')
        return sanitized or "unknown"

    def build_label_components(meta: dict) -> List[str]:
        components: List[str] = []
        seen: Set[Tuple[str, str]] = set()

        def append_component(prefix: str, raw_value):
            value = sanitize_for_filename(raw_value)
            if not value:
                return
            key = (prefix, value)
            if key in seen:
                return
            seen.add(key)
            components.append(f"{prefix}_{value}")

        variant = meta.get("kitchen_vis_variant")
        if variant:
            append_component("variant", variant)

        obs_mode = meta.get("obs_mode") or meta.get("vision_mode") or meta.get("vision")
        if obs_mode:
            append_component("obs", obs_mode)

        task_name = meta.get("task_name")
        if task_name:
            append_component("task", task_name)

        decoder_val = meta.get("decoder_phase")
        if decoder_val:
            append_component("decoder", decoder_val)

        policy_val = meta.get("policy_phase")
        if policy_val:
            append_component("policy", policy_val)

        phase_val = meta.get("phase_name")
        if phase_val and phase_val not in {decoder_val, policy_val}:
            append_component("phase", phase_val)

        if not components:
            append_component("phase", "unknown")

        return components

    def process_frame(frame: np.ndarray, from_buffer: bool = False):
        nonlocal video, video_path, video_width, video_height, timestamp, file_basename
        if not save_video:
            return
        if frame is None:
            logging.warning(f"[{addr}] Frame is None, skipping video write")
            return

        # Validate and convert frame
        try:
            frame = np.asarray(frame, dtype=np.uint8)
        except Exception as e:
            logging.warning(f"[{addr}] Failed to convert frame to uint8: {e}")
            return

        # Check dimensions
        if frame.ndim != 3:
            logging.warning(f"[{addr}] Frame has {frame.ndim} dimensions, expected 3. Shape: {frame.shape}")
            return
        if frame.shape[2] != 3:
            logging.warning(f"[{addr}] Frame has {frame.shape[2]} channels, expected 3. Shape: {frame.shape}")
            return

        # Check if frame is valid for writing
        if frame.size == 0:
            logging.warning(f"[{addr}] Frame has zero size")
            return

        if metadata_context["label"] is None:
            if not from_buffer:
                buffered_frames.append(frame.copy())
            return

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_label = sanitize_for_filename(metadata_context["environment"])
        label = sanitize_for_filename(metadata_context["label"])
        if file_basename is None:
            file_basename = f"{timestamp}_{env_label}_{label}"
        os.makedirs(output_dir, exist_ok=True)

        # Check if we need to (re)initialize VideoWriter
        height, width = frame.shape[:2]
        need_reinit = False

        if video is None:
            need_reinit = True
        elif video_width and video_height and video_width > 0 and video_height > 0:
            # Check if frame size matches stored VideoWriter size
            if width != video_width or height != video_height:
                logging.warning(
                    f"[{addr}] Frame size changed from ({video_width}x{video_height}) "
                    f"to ({width}x{height}). Reinitializing VideoWriter."
                )
                video.release()
                need_reinit = True

        if need_reinit:
            video_path = os.path.join(output_dir, f"{file_basename}.mp4")
            video = cv2.VideoWriter(video_path, mp4v, 30, (width, height))

            if not video.isOpened():
                logging.error(f"[{addr}] Failed to open VideoWriter at {video_path}")
                video = None
                video_width = None
                video_height = None
                return

            # Store the video dimensions
            video_width = width
            video_height = height
            logging.info(f"[{addr}] VideoWriter initialized: size=({width}x{height}), path={video_path}")

        # Ensure proper color format (RGB to BGR)
        try:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logging.warning(f"[{addr}] Failed to convert frame color space: {e}, using original frame")
            bgr_frame = frame  # Fallback to original

        # Write frame with error handling
        try:
            success = video.write(bgr_frame)
            if not success:
                logging.error(f"[{addr}] Failed to write frame to video. Frame shape: {bgr_frame.shape}, dtype: {bgr_frame.dtype}")
        except Exception as e:
            logging.error(f"[{addr}] Exception during video write: {e}")

    def update_metadata_context(new_metadata):
        nonlocal video, video_path, video_width, video_height, file_basename, timestamp
        if not isinstance(new_metadata, dict):
            return
        env_value = new_metadata.get("env_alias") or new_metadata.get("environment")
        if env_value:
            metadata_context["environment"] = str(env_value)
        metadata_context["label_components"] = build_label_components(new_metadata)
        if metadata_context["label_components"]:
            metadata_context["label"] = "_".join(metadata_context["label_components"])
        else:
            metadata_context["label"] = None
        file_basename = None
        timestamp = None
        if video is not None:
            video.release()
            video = None
            video_path = ""
            video_width = None
            video_height = None
        if buffered_frames and metadata_context["label"]:
            frames_to_flush = buffered_frames[:]
            buffered_frames.clear()
            for buffered in frames_to_flush:
                process_frame(buffered, from_buffer=True)

    def _tile_frames(frames: List[Optional[np.ndarray]], tile_height: int, tile_width: int, num_cols: int) -> np.ndarray:
        if tile_height <= 0 or tile_width <= 0:
            tile_height = tile_width = 224
        num_envs = len(frames)
        num_cols = max(1, num_cols)
        num_rows = max(1, math.ceil(num_envs / num_cols))
        total_slots = num_rows * num_cols
        padded_frames = frames + [None] * (total_slots - num_envs)
        grid_rows = []
        for row_idx in range(num_rows):
            row_frames = []
            for col_idx in range(num_cols):
                idx = row_idx * num_cols + col_idx
                frame = padded_frames[idx]
                if frame is None:
                    frame = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                else:
                    frame = np.asarray(frame, dtype=np.uint8)
                    if frame.ndim != 3 or frame.shape[2] != 3:
                        frame = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                    elif frame.shape[0] != tile_height or frame.shape[1] != tile_width:
                        frame = cv2.resize(frame, (tile_width, tile_height))
                row_frames.append(frame)
            grid_rows.append(np.hstack(row_frames))
        return np.vstack(grid_rows)

    def _compose_video_frame() -> Optional[np.ndarray]:
        # HIGH_RES MODE: Override dimensions to 720x720 when high_res_mode is enabled
        if high_res_mode:
            tile_width = tile_height = 720
        else:
            if embedding_image_size is None:
                tile_width = tile_height = 224
            elif isinstance(embedding_image_size, (list, tuple)):
                tile_width = int(embedding_image_size[0])
                tile_height = int(embedding_image_size[1])
            else:
                tile_width = tile_height = int(embedding_image_size)

        num_envs = max(1, len(mke.env_list))
        num_cols = min(4, num_envs)

        if studio_mode:
            # Studio mode: horizontal layout with agent view + 2x2 grid
            # Sizes depend on high_res_mode (via tile_width/tile_height)
            if len(mke.env_list) == 0:
                return None

            # Use first environment for studio layout
            env = mke.env_list[0]

            # Calculate half size for 2x2 grid cameras
            half_size = tile_width // 2

            # Render agent/overview at tile_width x tile_height
            try:
                agent_view = env.render(width=tile_width, height=tile_height)
            except Exception as exc:
                logging.warning(f"[{addr}] Failed to render agent view for video: {exc}")
                agent_view = None

            if agent_view is None:
                agent_view = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            else:
                agent_view = np.asarray(agent_view, dtype=np.uint8)

            # Render other cameras at half_size x half_size using render_studio
            try:
                studio_frames = env.render_studio(width=half_size, height=half_size)
            except Exception as exc:
                logging.warning(f"[{addr}] Failed to render studio frames for video: {exc}")
                studio_frames = {}

            # Get other camera frames
            wrist_frame = studio_frames.get("wrist")
            ovens_frame = studio_frames.get("ovens")
            pannels_frame = studio_frames.get("pannels")
            cabinets_frame = studio_frames.get("cabinets")

            # Convert to arrays or create black frames
            wrist_view = np.asarray(wrist_frame, dtype=np.uint8) if wrist_frame is not None else np.zeros((half_size, half_size, 3), dtype=np.uint8)
            ovens_view = np.asarray(ovens_frame, dtype=np.uint8) if ovens_frame is not None else np.zeros((half_size, half_size, 3), dtype=np.uint8)
            pannels_view = np.asarray(pannels_frame, dtype=np.uint8) if pannels_frame is not None else np.zeros((half_size, half_size, 3), dtype=np.uint8)
            cabinets_view = np.asarray(cabinets_frame, dtype=np.uint8) if cabinets_frame is not None else np.zeros((half_size, half_size, 3), dtype=np.uint8)

            # Create 2x2 grid on the right: [wrist, ovens] / [pannels, cabinets]
            top_row = np.hstack([wrist_view, ovens_view])  # (half_size, tile_width, 3)
            bottom_row = np.hstack([pannels_view, cabinets_view])  # (half_size, tile_width, 3)
            right_grid = np.vstack([top_row, bottom_row])  # (tile_height, tile_width, 3)

            # Combine horizontally: agent + right_grid = final (tile_height, 2*tile_width, 3)
            return np.concatenate([agent_view, right_grid], axis=1)
        else:
            # Non-studio mode: original behavior (agent + wrist views only)
            agent_frames: List[Optional[np.ndarray]] = []
            wrist_frames: List[Optional[np.ndarray]] = []
            for env in mke.env_list:
                try:
                    agent_frame = env.render(width=tile_width, height=tile_height)
                except Exception as exc:
                    logging.warning(f"[{addr}] Failed to render agent view for video: {exc}")
                    agent_frame = None
                try:
                    wrist_frame = env.render_wrist_view(width=tile_width, height=tile_height)
                except Exception as exc:
                    logging.warning(f"[{addr}] Failed to render wrist view: {exc}")
                    wrist_frame = None
                agent_frames.append(agent_frame)
                wrist_frames.append(wrist_frame)

            if not any(frame is not None for frame in agent_frames + wrist_frames):
                return None

            agent_grid = _tile_frames(
                [frame if frame is not None else None for frame in agent_frames],
                tile_height=tile_height,
                tile_width=tile_width,
                num_cols=num_cols,
            )
            wrist_grid = _tile_frames(
                [frame if frame is not None else None for frame in wrist_frames],
                tile_height=tile_height,
                tile_width=tile_width,
                num_cols=num_cols,
            )

            if wrist_grid.shape[0] != agent_grid.shape[0]:
                wrist_grid = cv2.resize(wrist_grid, (wrist_grid.shape[1], agent_grid.shape[0]))

            return np.concatenate([agent_grid, wrist_grid], axis=1)

    def _ensure_array(value, dtype):
        if value is None:
            return None
        arr = np.asarray(value)
        return arr.astype(dtype, copy=True)

    def _resize_frame(frame, override_size=None):
        if frame is None:
            return None
        size = override_size or embedding_image_size
        if size is None:
            return np.asarray(frame, dtype=np.uint8)
        if isinstance(size, (list, tuple)):
            width, height = int(size[0]), int(size[1])
        else:
            width = height = int(size)
        frame_arr = np.asarray(frame, dtype=np.uint8)
        if frame_arr.ndim != 3 or frame_arr.shape[2] != 3:
            return frame_arr
        if frame_arr.shape[0] == height and frame_arr.shape[1] == width:
            return frame_arr
        try:
            return cv2.resize(frame_arr, (width, height))
        except cv2.error:
            return frame_arr

    def _build_vision_observation(env, obs_vec, agent_frame=None, wrist_frame=None, studio_frames=None, include_images=False):
        state = _ensure_array(obs_vec, np.float32)
        robot_states = np.asarray(env.sim.data.qpos[:ROBOT_QPOS_DIM], dtype=np.float32).copy()
        ee_pos, ee_ori = env.get_ee_info()
        ee_pos = np.asarray(ee_pos, dtype=np.float32).reshape(-1)
        ee_ori = np.asarray(ee_ori, dtype=np.float32).reshape(-1)
        ee_states = np.concatenate([ee_pos, ee_ori], axis=0).astype(np.float32, copy=False)
        gripper = np.asarray(env.sim.data.qpos[7:9], dtype=np.float32).copy()

        payload = {
            "state": state,
            "robot_states": robot_states,
            "ee_states": ee_states,
            "gripper_states": gripper,
        }

        if include_images:
            if studio_mode and studio_frames is not None:
                # Use studio camera images (dict with multiple cameras)
                for cam_name, frame in studio_frames.items():
                    if frame is not None:
                        frame = _ensure_array(frame, np.uint8)
                        if embedding_image_size is not None:
                            frame = _resize_frame(frame)
                        # Map studio camera names to standard keys
                        if cam_name == "overview":
                            payload["agentview_image"] = frame
                        elif cam_name == "wrist":
                            payload["eye_in_hand_image"] = frame
                        elif cam_name == "ovens":
                            payload["ovens_image"] = frame
                        elif cam_name == "pannels":
                            payload["pannels_image"] = frame
                        elif cam_name == "cabinets":
                            payload["cabinets_image"] = frame
            else:
                # Use standard 2-camera images
                if agent_frame is None:
                    agent_frame = env.render()
                if wrist_frame is None:
                    wrist_frame = env.render_wrist_view()
                agent_frame = _ensure_array(agent_frame, np.uint8)
                wrist_frame = _ensure_array(wrist_frame, np.uint8)
                if embedding_image_size is not None:
                    agent_frame = _resize_frame(agent_frame)
                    wrist_frame = _resize_frame(wrist_frame)
                payload["agentview_image"] = agent_frame
                payload["eye_in_hand_image"] = wrist_frame

        return payload

    def _serialize_single_observation(obs, env=None, agent_frame=None, wrist_frame=None, studio_frames=None, include_images=False):
        if vision_mode and env is not None:
            return _build_vision_observation(
                env,
                obs,
                agent_frame=agent_frame,
                wrist_frame=wrist_frame,
                studio_frames=studio_frames,
                include_images=include_images,
            )
        if isinstance(obs, dict):
            serialized = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    serialized[key] = np.array(value, copy=True)
                elif include_images and isinstance(value, (list, tuple)):
                    serialized[key] = np.array(value, dtype=np.float32)
                else:
                    serialized[key] = value
        elif isinstance(obs, np.ndarray):
            cloned = np.array(obs, copy=True)
            serialized = cloned if include_images else cloned.tolist()
        elif isinstance(obs, (list, tuple)):
            serialized = list(obs)
        else:
            serialized = obs

        if agent_frame is not None:
            frame_array = np.asarray(agent_frame, dtype=np.uint8)
            if isinstance(serialized, dict):
                serialized["agentview_image"] = frame_array
            else:
                serialized = {
                    "state": serialized,
                    "agentview_image": frame_array,
                }
        if wrist_frame is not None and isinstance(serialized, dict):
            serialized["eye_in_hand_image"] = np.asarray(wrist_frame, dtype=np.uint8)
        return serialized

    def _prepare_observations(include_images=False):
        if include_images and vision_mode and studio_mode:
            # Render studio frames (dict of multiple cameras per env)
            studio_frames_list = []
            for env in mke.env_list:
                try:
                    studio_frames = env.render_studio()
                except Exception as exc:
                    logging.warning(f"[{addr}] Failed to render studio frames: {exc}")
                    studio_frames = {}
                # Resize all frames
                resized_studio_frames = {}
                for cam_name, frame in studio_frames.items():
                    resized_studio_frames[cam_name] = _resize_frame(frame)
                studio_frames_list.append(resized_studio_frames)
            agent_frames = [None] * len(mke.env_list)
            wrist_frames = [None] * len(mke.env_list)
        elif include_images and vision_mode:
            agent_frames = []
            wrist_frames = []
            for env in mke.env_list:
                try:
                    agent_frame = env.render()
                except Exception as exc:
                    logging.warning(f"[{addr}] Failed to render frames for vision mode: {exc}")
                    agent_frame = None
                try:
                    wrist_frame = env.render_wrist_view()
                except Exception as exc:
                    logging.warning(f"[{addr}] Failed to render wrist frames for vision mode: {exc}")
                    wrist_frame = None
                agent_frames.append(_resize_frame(agent_frame))
                wrist_frames.append(_resize_frame(wrist_frame))
            studio_frames_list = [None] * len(mke.env_list)
        elif include_images:
            agent_frames = mke.render_per_env(resize_shape=embedding_image_size)
            wrist_frames = [None] * len(agent_frames)
            studio_frames_list = [None] * len(agent_frames)
        else:
            agent_frames = [None] * len(mke.env_list)
            wrist_frames = [None] * len(mke.env_list)
            studio_frames_list = [None] * len(mke.env_list)
        serialized_obs = []
        for idx, obs in enumerate(mke.obs_list):
            env = mke.env_list[idx]
            frame = agent_frames[idx] if idx < len(agent_frames) else None
            wrist = wrist_frames[idx] if idx < len(wrist_frames) else None
            studio_frames = studio_frames_list[idx] if idx < len(studio_frames_list) else None
            serialized_obs.append(
                _serialize_single_observation(
                    obs,
                    env=env,
                    agent_frame=frame,
                    wrist_frame=wrist,
                    studio_frames=studio_frames,
                    include_images=include_images,
                )
            )
        return serialized_obs

    # Log client connection
    logging.info(f"Client connected from {addr}")
    print(f"Client connected from {addr}")

    # Initialize the multi-environment instance.
    # (Assumes MultiKitchenEnv has already been imported.)
    mke = MultiKitchenEnv(semantic_flag=False)  # NOTE: semantic_flag for evaluation modify here

    # Send initial observations for all environments
    include_images = embedding_mode or vision_mode
    init_obs_serialized = _prepare_observations(include_images=include_images)
    init_response = {
        "observations": init_obs_serialized,
        "rewards": [0.0] * len(mke.env_list),
        "done": [False] * len(mke.env_list),
        "info": {},
        "image_mode": image_mode
    }
    send_pickle(conn, init_response)
    logging.info(f"[{addr}] Sent initial observations.")
    if save_video:
        process_frame(_compose_video_frame())

    # Main loop for processing client messages
    while True:
        msg = recv_pickle(conn)
        if msg is None:
            logging.info(f"[{addr}] Connection closed by client.")
            break

        metadata_payload = msg.get("metadata")
        if metadata_payload and "set_task" not in msg:
            update_metadata_context(metadata_payload)

        # Handle embedding mode toggle before other commands
        if msg.get("set_embedding_mode") is not None:
            embedding_mode = bool(msg["set_embedding_mode"])
            size_value = msg.get("image_size")
            if size_value is not None:
                if isinstance(size_value, (list, tuple)) and len(size_value) == 2:
                    embedding_image_size = (int(size_value[0]), int(size_value[1]))
                elif isinstance(size_value, (int, float)):
                    side = int(size_value)
                    embedding_image_size = (side, side)
                elif size_value in (None, "none"):
                    embedding_image_size = None
            logging.info(
                f"[{addr}] Embedding mode set to {embedding_mode} "
                f"(resize={embedding_image_size})"
            )
            include_images = embedding_mode or vision_mode
            response = {
                "observations": _prepare_observations(include_images=include_images),
                "rewards": [0.0] * len(mke.env_list),
                "done": mke.done_list,
                "info": {"embedding_mode": embedding_mode},
            }
            send_pickle(conn, response)
            continue

        # Handle studio mode toggle
        if msg.get("set_studio_mode") is not None:
            studio_mode = bool(msg["set_studio_mode"])
            logging.info(f"[{addr}] Studio mode set to {studio_mode}")
            include_images = embedding_mode or vision_mode
            response = {
                "observations": _prepare_observations(include_images=include_images),
                "rewards": [0.0] * len(mke.env_list),
                "done": mke.done_list,
                "info": {"studio_mode": studio_mode},
            }
            send_pickle(conn, response)
            continue

        # Handle high-res mode toggle
        if msg.get("set_high_res_mode") is not None:
            high_res_mode = bool(msg["set_high_res_mode"])
            logging.info(f"[{addr}] High-res mode set to {high_res_mode}")
            include_images = embedding_mode or vision_mode
            response = {
                "observations": _prepare_observations(include_images=include_images),
                "rewards": [0.0] * len(mke.env_list),
                "done": mke.done_list,
                "info": {"high_res_mode": high_res_mode},
            }
            send_pickle(conn, response)
            continue

        # Handle task-setting command: update task configuration using MultiKitchenEnv.set_task
        if msg.get("set_task") is not None:
            new_eval_config = msg["set_task"]
            # Extract noise_scale if provided
            noise_scale = msg.get("noise_scale", None)
            metadata = dict(metadata_payload) if metadata_payload else {}
            provided_task_name = msg.get("task_name")
            if provided_task_name:
                current_task_name = provided_task_name
            else:
                data_names = [cfg.get("data_name") for cfg in new_eval_config if isinstance(cfg, dict) and cfg.get("data_name")]
                if data_names:
                    current_task_name = "-".join(data_names)
            if "task_name" not in metadata and current_task_name:
                metadata["task_name"] = current_task_name
            if metadata:
                vision_value = metadata.get("obs_mode", metadata.get("vision_mode", metadata.get("vision")))
                if isinstance(vision_value, str):
                    lowered = vision_value.strip().lower()
                    if lowered in ("vision", "true", "1", "yes"):
                        vision_mode = True
                    elif lowered in ("state", "false", "0", "no"):
                        vision_mode = False
                elif vision_value is not None:
                    vision_mode = bool(vision_value)
                logging.info(f"[{addr}] Vision mode set to {vision_mode} via metadata.")
            # If client provides a task name, update current_task_name and reinitialize video writer.
            if current_task_name:
                logging.info(f"[{addr}] Using task label: {current_task_name}")
            # Update task settings in the environment.
            mke.set_task(new_eval_config, noise_scale=noise_scale)
            if metadata:
                update_metadata_context(metadata)
            include_images = embedding_mode or vision_mode
            init_obs_serialized = _prepare_observations(include_images=include_images)
            response = {
                "observations": init_obs_serialized,
                "rewards": [0.0] * len(mke.env_list),
                "done": [False] * len(mke.env_list),
                "info": {}
            }

            # Include rendered image if image_mode is enabled
            if image_mode:
                frame = mke.render()
                if frame is not None:
                    response["image"] = frame.tolist()

            send_pickle(conn, response)
            logging.info(f"[{addr}] Task settings updated upon client request.")
            if save_video:
                process_frame(_compose_video_frame())
            continue

        # Handle image mode setting: enable/disable image mode
        if msg.get("set_image_mode") is not None:
            image_mode = msg["set_image_mode"]
            logging.info(f"[{addr}] Image mode set to: {image_mode}")
            response = {"image_mode": image_mode}
            send_pickle(conn, response)
            continue

        # Handle reset command: reset all environments and send back new observations.
        if msg.get("reset", False):
            mke.reset_model()
            include_images = embedding_mode or vision_mode
            init_obs_serialized = _prepare_observations(include_images=include_images)
            response = {
                "observations": init_obs_serialized,
                "rewards": [0.0] * len(mke.env_list),
                "done": [False] * len(mke.env_list),
                "info": {}
            }

            # Include rendered image if image_mode is enabled
            if image_mode:
                frame = mke.render()
                if frame is not None:
                    response["image"] = frame.tolist()

            send_pickle(conn, response)
            logging.info(f"[{addr}] Environments reset upon client request.")

            if save_video:
                process_frame(_compose_video_frame())
            continue

        # Process an action command: expect a batch of actions for all environments.
        actions = msg.get("action")
        if actions is None:
            logging.warning(f"[{addr}] Received message without action.")
            continue

        # Convert actions to a numpy array (expected shape: (N, act_dim))
        actions = np.array(actions)
        new_obs_list, rewards, done_list = mke.step(actions)
        include_images = embedding_mode or vision_mode
        serialized_obs = _prepare_observations(include_images=include_images)
        response = {
            "observations": serialized_obs,
            "rewards": rewards.tolist() if isinstance(rewards, np.ndarray) else rewards,
            "done": mke.done_list,
            "info": {}
        }

        # Include rendered image if image_mode is enabled
        if image_mode:
            frame = mke.render()
            if frame is not None:
                response["image"] = frame.tolist()

        send_pickle(conn, response)
        logging.info(f"[{addr}] Processed action, rewards: {rewards}, done: {mke.done_list}")

        if save_video:
            process_frame(_compose_video_frame())

    conn.close()
    if save_video:
        if video is not None:
            video.release()
        if video_path:
            logging.info(f"[{addr}] Client process ended. Video saved to {video_path}")
        else:
            logging.info(f"[{addr}] Client process ended. Video recording not created.")
    else:
        logging.info(f"[{addr}] Client process ended.")

class BaseServer:
    def __init__(self, host='0.0.0.0', port=9999, save_video=False, high_res=False):
        self.host = host
        self.port = port
        self.save_video = save_video
        self.high_res = high_res
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm to reduce latency
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Allow reusing the same address/port immediately after closing
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.sock.bind((self.host, self.port))
        self.sock.listen(10)
        logging.info(f"Server listening on {self.host}:{self.port}")

    def serve_forever(self):
        while True:
            conn, addr = self.sock.accept()
            logging.info(f"Accepted connection from {addr}")
            print(f"Accepted connection from {addr}")

            # Pass the save_video and high_res flags to the client handler process
            p = Process(target=handle_client, args=(conn, addr, self.save_video, self.high_res))
            p.daemon = True
            p.start()
            # test for non multi env

            # logging.info(f"Spawned process {p.pid} for client {addr}")
    def close(self):
        self.sock.close()
        logging.info("Server socket closed.")

import argparse
import logging

# Import the BaseServer class from your module (uncomment and modify the line below as needed)
# from your_module import BaseServer

def main():
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Server execution script")
    parser.add_argument('--host', type=str, default='0.0.0.0', help="Server host address (default: 0.0.0.0)")
    parser.add_argument('--port', type=int, default=9999, help="Server port number (default: 9999)")
    parser.add_argument('--save_video', action='store_true',
                        help="Flag to record video. If present, save_video is True; otherwise, it is False.")
    parser.add_argument('--high_res', action='store_true',
                        help="Flag to enable high-resolution (720x720) video recording. If present, high_res is True; otherwise, it is False.")

    # important to use EGL for rendering
    import multiprocessing as mp
    mp.set_start_method('spawn')
    args = parser.parse_args()

    # Initialize the BaseServer with the provided command-line arguments
    server = BaseServer(host=args.host, port=args.port, save_video=args.save_video, high_res=args.high_res)

    try:
        # Start the server and keep it running indefinitely
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server shutting down due to KeyboardInterrupt.")
        server.close()

if __name__ == '__main__':
    main()
