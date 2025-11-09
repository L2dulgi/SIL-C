import socket
import pickle
import struct
import logging
import numpy as np
import cv2
from datetime import datetime
from multiprocessing import Process

# --------------------------
# Required Libero-related modules
# --------------------------
import re
import os
import imageio
import json
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

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

def extract_language_from_bddl(bddl_file_path):
    """
    Extracts a natural language command from a BDDL file using the pattern (:language ...).
    """
    try:
        with open(bddl_file_path, "r") as f:
            text = f.read()
        match = re.search(r'\(:language\s+(.*?)\)', text)
        if match:
            return match.group(1).strip()
        else:
            return ""
    except Exception as e:
        print("[extract_language_from_bddl] Error:", e)
        return ""

class LiberoEnv:
    """
    Wrapper class using the 0th task of the libero_object benchmark by default.
    Can be reconfigured with set_task(benchmark_name, task_id).
    """
    def __init__(self, benchmark_name="libero_object", task_id=0):
        self.benchmark_name = benchmark_name
        self.task_id = task_id

        self.env = None
        self.natural_language = ""
        self.observation = None
        self.reward = 0.0
        self.done = False

        self.reset_model()

    def set_task(self, benchmark_name, task_id):
        """
        Updates the benchmark and task with new inputs, then resets the environment.
        """
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.reset_model()

    def reset_model(self):
        """
        Closes the existing environment (if any), then creates and resets a new one.
        Returns the initial observation.
        """
        bench_dict = benchmark.get_benchmark_dict()
        if self.benchmark_name not in bench_dict:
            logging.warning(f"[LiberoEnv] Unknown benchmark '{self.benchmark_name}'.")
            return None

        bench_cls = bench_dict[self.benchmark_name]
        bench_instance = bench_cls()

        num_tasks = bench_instance.get_num_tasks()
        if self.task_id >= num_tasks:
            logging.warning(f"[LiberoEnv] Task ID {self.task_id} is invalid. (num_tasks={num_tasks})")
            return None

        bddl_file_path = bench_instance.get_task_bddl_file_path(self.task_id)
        self.natural_language = extract_language_from_bddl(bddl_file_path)

        if self.env is not None:
            self.env.close()

        # Request both agent and eye-in-hand cameras so we can compose
        # a side-by-side video like the kitchen implementation.
        env_args = {
            "bddl_file_name": bddl_file_path,
            # Use ints so the same size applies to all cameras
            "camera_heights": 128,
            "camera_widths": 128,
            # Explicitly ask for both views if available in LIBERO
            # (falls back gracefully if not supported by the env)
            "camera_names": ["agentview", "robot0_eye_in_hand"],
        }
        self.env = OffScreenRenderEnv(**env_args)
        self.env.reset()

        init_states = bench_instance.get_task_init_states(self.task_id)
        if init_states is None or len(init_states) == 0:
            logging.warning("[LiberoEnv] No init states found in benchmark.")
        else:
            self.env.set_init_state(init_states[0])

        obs, rew, done, info = self.env.step([0.0]*7)
        obs['state'] = self.env.get_sim_state()
        self.observation = obs
        self.reward = rew
        self.done = done
        return obs

    def step(self, action):
        """
        Performs a step with the given 7D action and returns the result.
        """
        obs, rew, done, info = self.env.step(action)
        obs['state'] = self.env.get_sim_state()
        self.observation = obs
        self.reward = rew
        self.done = done
        return obs, rew, done, info

    def render(self):
        """
        Returns a composite frame for video recording combining agent view and
        eye-in-hand view side-by-side when both are available. Falls back to a
        single available view otherwise.
        """
        if self.observation is None or not isinstance(self.observation, dict):
            return None

        # Prefer agentview/frontview for the left panel
        agent = None
        if "agentview_image" in self.observation:
            agent = self.observation["agentview_image"]
        elif "frontview_image" in self.observation:
            agent = self.observation["frontview_image"]

        # Prefer robot0_eye_in_hand_image for the right panel
        wrist = None
        if "robot0_eye_in_hand_image" in self.observation:
            wrist = self.observation["robot0_eye_in_hand_image"]
        elif "eye_in_hand_image" in self.observation:
            wrist = self.observation["eye_in_hand_image"]

        if agent is None and wrist is None:
            return None

        # Ensure numpy uint8 arrays
        def to_u8(arr):
            a = np.asarray(arr)
            if a.dtype != np.uint8:
                a = a.astype(np.uint8)
            return a

        if agent is not None and wrist is not None:
            agent = to_u8(agent)
            wrist = to_u8(wrist)

            # Resize right panel to match left panel height/width if needed
            h, w = agent.shape[:2]
            if wrist.shape[:2] != (h, w):
                try:
                    wrist = cv2.resize(wrist, (w, h))
                except cv2.error:
                    # If resize fails, return agent only to keep video stable
                    return agent

            try:
                return np.concatenate([agent, wrist], axis=1)
            except Exception:
                # Fallback gracefully
                return agent

        # Only one available
        return to_u8(agent if agent is not None else wrist)

    def close(self):
        """
        Closes the MuJoCo environment and clears the EGL context.
        """
        if self.env is not None:
            self.env.close()
            self.env = None

def handle_client_single(conn, addr, save_video=False, save_gif=False):
    logging.info(f"Client connected from {addr}")
    print(f"Client connected from {addr}")

    env = LiberoEnv()

    video = None
    video_path = ""
    gif_writer = None
    gif_path = ""
    timestamp = None
    output_dir = os.path.join("data", "videos")
    mp4v = cv2.VideoWriter_fourcc(*'mp4v')
    file_basename = None
    metadata_context = {
        "environment": "libero",
        "label_components": [],
        "label": None,
    }
    buffered_frames = []

    def sanitize_for_filename(value: str) -> str:
        sanitized = re.sub(r'[^0-9A-Za-z_\-]+', '_', str(value))
        sanitized = sanitized.strip('_')
        return sanitized or "unknown"

    def build_label_components(meta: dict) -> list:
        components = []
        seen = set()

        def append_component(prefix: str, raw_value):
            value = sanitize_for_filename(raw_value)
            if not value:
                return
            key = (prefix, value)
            if key in seen:
                return
            seen.add(key)
            components.append(f"{prefix}_{value}")

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

    def update_metadata_context(new_metadata):
        nonlocal file_basename, video, gif_writer, video_path, gif_path, timestamp
        if not isinstance(new_metadata, dict):
            return
        env_value = new_metadata.get("environment")
        if env_value:
            metadata_context["environment"] = str(env_value)
        metadata_context["label_components"] = build_label_components(new_metadata)
        metadata_context["label"] = "_".join(metadata_context["label_components"])
        file_basename = None
        timestamp = None
        if video is not None:
            video.release()
            video = None
            video_path = ""
        if gif_writer is not None:
            gif_writer.close()
            gif_writer = None
            gif_path = ""
        if buffered_frames:
            frames_to_flush = buffered_frames[:]
            buffered_frames.clear()
            for frame, flip_flag in frames_to_flush:
                process_frame(frame, flip=flip_flag, from_buffer=True)

    def process_frame(frame, flip=False, from_buffer=False):
        nonlocal video, gif_writer, video_path, gif_path, timestamp, file_basename
        if frame is None or not (save_video or save_gif):
            return
        if metadata_context["label"] is None:
            if not from_buffer:
                buffered_frames.append((frame.copy(), flip))
            return
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_label = sanitize_for_filename(metadata_context["environment"])
        meta_label = metadata_context["label"]
        if file_basename is None:
            file_basename = f"{timestamp}_{env_label}_{meta_label}"
        os.makedirs(output_dir, exist_ok=True)
        target_frame = np.flip(frame, axis=0) if flip else frame
        created_video = False
        created_gif = False
        if save_video:
            if video is None:
                height, width, _ = target_frame.shape
                video_size = (width, height)
                video_path = os.path.join(output_dir, f"{file_basename}.mp4")
                video = cv2.VideoWriter(video_path, mp4v, 30, video_size)
                created_video = True
            bgr_frame = cv2.cvtColor(target_frame, cv2.COLOR_RGB2BGR)
            video.write(bgr_frame)
        if save_gif:
            if gif_writer is None:
                gif_path = os.path.join(output_dir, f"{file_basename}.gif")
                gif_writer = imageio.get_writer(gif_path, fps=10)
                created_gif = True
            gif_writer.append_data(target_frame)
        if created_video or created_gif:
            logging.info(
                f"[{addr}] Recording enabled (video={save_video}, gif={save_gif}) "
                f"env={metadata_context['environment']} metadata={metadata_context['label']}"
            )

    # Skip logging the very first frame from env init to avoid capturing
    # an inconsistent frame during setup. Subsequent frames (after steps/reset)
    # will be recorded normally.
    # first_frame = env.render()
    # (intentionally not recorded)

    obs = env.observation
    response = {
        "observation": obs,
        "reward": env.reward,
        "done": env.done,
        "info": {},
        "natural_language": env.natural_language,
    }
    send_pickle(conn, response)
    logging.info(f"[{addr}] Sent initial observation for LiberoEnv.")

    try:
        while True:
            msg = recv_pickle(conn)
            if msg is None:
                logging.info(f"[{addr}] Connection closed by client.")
                break
            metadata_payload = msg.get("metadata")
            if metadata_payload:
                update_metadata_context(metadata_payload)
                logging.info(
                    f"[{addr}] Received metadata env={metadata_context['environment']} "
                    f"label={metadata_context['label']}"
                )
            if "set_task" in msg:
                new_cfg = msg["set_task"]
                benchmark_name = new_cfg.get("benchmark_name", "libero_90")
                task_id = new_cfg.get("task_id", 0)
                env.set_task(benchmark_name, task_id)

                obs = env.observation
                response = {
                    "observation": obs,
                    "reward": env.reward,
                    "done": env.done,
                    "info": {},
                    "natural_language": env.natural_language,
                }
                send_pickle(conn, response)
                logging.info(f"[{addr}] Changed task -> {benchmark_name}, task_id={task_id}")
                continue

            if msg.get("reset", False):
                obs = env.reset_model()
                response = {
                    "observation": obs,
                    "reward": env.reward,
                    "done": env.done,
                    "info": {},
                    "natural_language": env.natural_language,
                }
                send_pickle(conn, response)
                logging.info(f"[{addr}] Env reset done.")
                if save_video or save_gif:
                    frame = env.render()
                    if frame is not None:
                        process_frame(frame, flip=False)
                continue

            actions = msg.get("action")
            if actions is None:
                logging.warning(f"[{addr}] Received message without action.")
                continue

            actions = np.array(actions)
            if actions.ndim != 1:
                actions = actions.squeeze()
            obs, rew, done, info = env.step(actions)

            response = {
                "observation": obs,
                "reward": rew,
                "done": done,
                "info": info,
                "natural_language": env.natural_language,
            }
            send_pickle(conn, response)
            logging.info(f"[{addr}] Step done, reward={rew}, done={done}")

            if save_video or save_gif:
                frame = env.render()
                if frame is not None:
                    process_frame(frame, flip=True)
    finally:
        env.close()
        conn.close()

        artifacts = []
        if save_video and video is not None:
            video.release()
            artifacts.append(f"video -> {video_path}")
        if save_gif and gif_writer is not None:
            gif_writer.close()
            artifacts.append(f"gif -> {gif_path}")
        if artifacts:
            logging.info(f"[{addr}] Client ended. Saved {'; '.join(artifacts)}")
        else:
            logging.info(f"[{addr}] Client ended.")

class BaseServer:
    def __init__(self, host='0.0.0.0', port=7777, save_video=False, save_gif=False):
        self.host = host
        self.port = port
        self.save_video = save_video
        self.save_gif = save_gif
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(10)
        logging.info(f"Server listening on {self.host}:{self.port}")

    def serve_forever(self):
        while True:
            conn, addr = self.sock.accept()
            logging.info(f"Accepted connection from {addr}")
            print(f"Accepted connection from {addr}")

            p = Process(
                target=handle_client_single,
                args=(conn, addr, self.save_video, self.save_gif)
            )
            p.daemon = True
            p.start()

    def close(self):
        self.sock.close()
        logging.info("Server socket closed.")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Libero Server execution script")
    parser.add_argument('--host', type=str, default='0.0.0.0', help="Server host address")
    parser.add_argument('--port', type=int, default=7777, help="Server port number")
    parser.add_argument('--save_video', action='store_true', 
                        help="If present, save_video is True; otherwise, False.")
    parser.add_argument('--save_gif', action='store_true',
                        help="If present, save_gif is True; otherwise, False.")

    import multiprocessing as mp
    mp.set_start_method('spawn')
    args = parser.parse_args()

    server = BaseServer(
        host=args.host,
        port=args.port,
        save_video=args.save_video,
        save_gif=args.save_gif
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server shutting down (KeyboardInterrupt).")
        server.close()

from typing import Dict, Any

def save_all_bddl_filenames_to_file(
    benchmark_name: str,
    output_file: str
) -> Dict[int, Dict[str, Any]]:
    """
    For all task_ids in the given benchmark_name,
    generates a dict of the form:
    { task_id: {"benchmark_name":…, "task_id":…, "bddl_file":…}, … }
    and saves it to a single JSON output_file.
    Returns the generated dict.
    """
    bench_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in bench_dict:
        logging.error(f"[save_all_bddl_filenames_to_file] Unknown benchmark '{benchmark_name}'")
        return {}

    bench_cls = bench_dict[benchmark_name]
    bench_instance = bench_cls()
    num_tasks = bench_instance.get_num_tasks()

    mapping: Dict[int, Dict[str, Any]] = {}
    for task_id in range(num_tasks):
        bddl_path = bench_instance.get_task_bddl_file_path(task_id)
        bddl_name = os.path.basename(bddl_path)
        dataset_name = os.path.splitext(bddl_name)[0]
        mapping[task_id] = {
            "benchmark_name": benchmark_name,
            "task_id": task_id,
            "bddl_file": dataset_name
        }

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        logging.info(f"[save_all_bddl_filenames_to_file] Saved all to {output_file}")
    except Exception as e:
        logging.error(f"[save_all_bddl_filenames_to_file] Failed to write {output_file}: {e}")

    return mapping

if __name__ == '__main__':
    main()
