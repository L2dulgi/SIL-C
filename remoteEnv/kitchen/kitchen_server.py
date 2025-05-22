# .remoteEnv/kitchen/kitchen_server.py
import socket
import pickle
import struct
import numpy as np
from kitchen import *  # Assume KitchenEnv, KitchenTask, etc. are imported from kitchen module
from multiprocessing import Process
import logging

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
    if save_video and video is not None:
        video.release()
        logging.info(f"[{addr}] Client process ended. Video saved to {video_path}")
    else:
        logging.info(f"[{addr}] Client process ended.")

def handle_client(conn, addr, save_video=False):
    # Initial task name (default value)
    current_task_name = "default"

    # Log client connection
    logging.info(f"Client connected from {addr}")
    print(f"Client connected from {addr}")

    # Initialize the multi-environment instance.
    # (Assumes MultiKitchenEnv has already been imported.)
    mke = MultiKitchenEnv(semantic_flag=False)  # NOTE: semantic_flag for evaluation modify here

    # Setup video writer if video saving is enabled using current_task_name.
    video = None
    if save_video:
        import cv2
        mp4v = cv2.VideoWriter_fourcc(*'mp4v')
        first_frame = mke.render()  # render() returns the combined grid image
        vid_size = (first_frame.shape[1], first_frame.shape[0])  # width, height
        video_path = f"data/videos/{current_task_name}_client_{addr[0]}_{addr[1]}_video.mp4"
        video = cv2.VideoWriter(video_path, mp4v, 30, vid_size)
        # Write the first frame to the video
        bgr_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        video.write(bgr_first_frame)

    # Send initial observations for all environments
    init_obs_serialized = [
        obs.tolist() if isinstance(obs, np.ndarray) else obs
        for obs in mke.obs_list
    ]
    init_response = {
        "observations": init_obs_serialized,
        "rewards": [0.0] * len(mke.env_list),
        "done": [False] * len(mke.env_list),
        "info": {}
    }
    send_pickle(conn, init_response)
    logging.info(f"[{addr}] Sent initial observations.")

    # Main loop for processing client messages
    while True:
        msg = recv_pickle(conn)
        if msg is None:
            logging.info(f"[{addr}] Connection closed by client.")
            break

        # Handle task-setting command: update task configuration using MultiKitchenEnv.set_task
        if msg.get("set_task") is not None:
            new_eval_config = msg["set_task"]
            # If client provides a task name, update current_task_name and reinitialize video writer.
            if "task_name" in msg:
                current_task_name = msg["task_name"]
                logging.info(f"[{addr}] Received new task name: {current_task_name}")
                if save_video:
                    # Release current video writer and reinitialize with new video path.
                    video.release()
                    import cv2
                    mp4v = cv2.VideoWriter_fourcc(*'mp4v')
                    # Use current frame to determine video dimensions.
                    first_frame = mke.render()
                    vid_size = (first_frame.shape[1], first_frame.shape[0])
                    video_path = f"data/videos/{current_task_name}_client_video.mp4"
                    video = cv2.VideoWriter(video_path, mp4v, 30, vid_size)
                    # Write the first frame to the new video file.
                    bgr_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
                    video.write(bgr_first_frame)
            # Update task settings in the environment.
            mke.set_task(new_eval_config)
            init_obs_serialized = [
                obs.tolist() if isinstance(obs, np.ndarray) else obs
                for obs in mke.obs_list
            ]
            response = {
                "observations": init_obs_serialized,
                "rewards": [0.0] * len(mke.env_list),
                "done": [False] * len(mke.env_list),
                "info": {}
            }
            send_pickle(conn, response)
            logging.info(f"[{addr}] Task settings updated upon client request.")
            continue

        # Handle reset command: reset all environments and send back new observations.
        if msg.get("reset", False):
            mke.reset_model()
            init_obs_serialized = [
                obs.tolist() if isinstance(obs, np.ndarray) else obs
                for obs in mke.obs_list
            ]
            response = {
                "observations": init_obs_serialized,
                "rewards": [0.0] * len(mke.env_list),
                "done": [False] * len(mke.env_list),
                "info": {}
            }
            send_pickle(conn, response)
            logging.info(f"[{addr}] Environments reset upon client request.")

            if save_video and video is not None:
                frame = mke.render()
                if frame is not None:
                    import cv2
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video.write(bgr_frame)
            continue

        # Process an action command: expect a batch of actions for all environments.
        actions = msg.get("action")
        if actions is None:
            logging.warning(f"[{addr}] Received message without action.")
            continue

        # Convert actions to a numpy array (expected shape: (N, act_dim))
        actions = np.array(actions)
        new_obs_list, rewards, done_list = mke.step(actions)
        serialized_obs = [
            obs.tolist() if isinstance(obs, np.ndarray) else obs
            for obs in new_obs_list
        ]
        response = {
            "observations": serialized_obs,
            "rewards": rewards.tolist() if isinstance(rewards, np.ndarray) else rewards,
            "done": mke.done_list,
            "info": {}
        }
        send_pickle(conn, response)
        logging.info(f"[{addr}] Processed action, rewards: {rewards}, done: {mke.done_list}")

        if save_video and video is not None:
            frame = mke.render()
            if frame is not None:
                import cv2
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(bgr_frame)

    conn.close()
    if save_video and video is not None:
        video.release()
        logging.info(f"[{addr}] Client process ended. Video saved to {video_path}")
    else:
        logging.info(f"[{addr}] Client process ended.")

class BaseServer:
    def __init__(self, host='0.0.0.0', port=9999, save_video=False):
        self.host = host
        self.port = port
        self.save_video = save_video
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

            # Pass the save_video flag to the client handler process
            p = Process(target=handle_client, args=(conn, addr, self.save_video))
            p.daemon = True
            p.start()
            # test for non multi env
            # handle_client_single(conn, addr, self.save_video)

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
    
    # important to use EGL for rendering
    import multiprocessing as mp
    mp.set_start_method('spawn')
    args = parser.parse_args()

    # Initialize the BaseServer with the provided command-line arguments
    server = BaseServer(host=args.host, port=args.port, save_video=args.save_video)

    try:
        # Start the server and keep it running indefinitely
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server shutting down due to KeyboardInterrupt.")
        server.close()

if __name__ == '__main__':
    main()