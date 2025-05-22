import socket
import pickle
import struct
import logging
import numpy as np
import cv2
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

        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 128,
            "camera_widths": 128,
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
        Returns a render frame for video recording (either frontview_image or agentview_image).
        """
        if self.observation is not None and isinstance(self.observation, dict):
            if "frontview_image" in self.observation:
                return self.observation["frontview_image"]
            elif "agentview_image" in self.observation:
                return self.observation["agentview_image"]
        return None

    def close(self):
        """
        Closes the MuJoCo environment and clears the EGL context.
        """
        if self.env is not None:
            self.env.close()
            self.env = None

def handle_client_single(conn, addr, save_video=False):
    logging.info(f"Client connected from {addr}")
    print(f"Client connected from {addr}")

    env = LiberoEnv()

    video = None
    video_path = ""
    if save_video:
        logging.info(f"[{addr}] Video recording enabled.")
        mp4v = cv2.VideoWriter_fourcc(*'mp4v')
        first_frame = env.render()
        if first_frame is not None:
            height, width, _ = first_frame.shape
            video_size = (width, height)
            video_path = f"data/videos/libero_client_{addr[0]}_{addr[1]}.mp4"
            video = cv2.VideoWriter(video_path, mp4v, 30, video_size)
            bgr_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
            video.write(bgr_first_frame)

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
                if save_video and video is not None:
                    frame = env.render()
                    if frame is not None:
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video.write(bgr_frame)
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

            if save_video and video is not None:
                frame = env.render()
                if frame is not None:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    bgr_frame = cv2.flip(bgr_frame, 0)
                    video.write(bgr_frame)
    finally:
        env.close()
        conn.close()

        if save_video and video is not None:
            video.release()
            logging.info(f"[{addr}] Client ended. Video saved to {video_path}")
        else:
            logging.info(f"[{addr}] Client ended.")

class BaseServer:
    def __init__(self, host='0.0.0.0', port=7777, save_video=False):
        self.host = host
        self.port = port
        self.save_video = save_video
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

            p = Process(target=handle_client_single, args=(conn, addr, self.save_video))
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

    import multiprocessing as mp
    mp.set_start_method('spawn')
    args = parser.parse_args()

    server = BaseServer(host=args.host, port=args.port, save_video=args.save_video)
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
