import socket
import pickle
import struct
import time
import numpy as np
from tqdm import tqdm

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

# --------------------------
# DummyModel (7-dimensional action)
# --------------------------
class DummyModel:
    def eval_model(self, observation):
        """
        Receives an observation and returns a random 7D action in the range [-1, 1].
        """
        return np.random.uniform(-1, 1, size=(7,)).tolist()

class LiberoRemoteEvaluator:
    """
    - Connects to the server via socket.
    - Receives initial observation (obs / natural_language / reward / done).
    - For each episode, repeats the step loop:
      - Prints observation keys and their shapes.
      - Also prints natural language, reward, and done flag.
      - Sends an action and receives the next response.
      - If done is True, ends the episode.
    - Sends reset request -> starts next episode.
    """
    def __init__(self, host='127.0.0.1', port=7777):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        print(f"Connected to libero server at {host}:{port}")
        self.model = DummyModel()

    def evaluate(self, num_episodes=3, max_steps=50):
        init_response = recv_pickle(self.sock)
        if init_response is None:
            print("[Error] No data received initially. Connection closed.")
            return

        current_response = init_response

        for ep in range(num_episodes):
            cumulative_reward = 0.0
            pbar = tqdm(range(max_steps), desc=f"Episode {ep+1}",
                        postfix={"reward": f"{cumulative_reward:.2f}"})

            for step in pbar:
                obs = current_response.get("observation", None)
                natural_lang = current_response.get("natural_language", "")
                reward = current_response.get("reward", 0.0)
                done = current_response.get("done", False)

                # Print only keys and shapes of observation
                print(f"\n[Step {step}] ==== Observation info ====")
                if isinstance(obs, dict):
                    for k, v in obs.items():
                        # Check shape (handles ndarray, list, etc.)
                        if hasattr(v, 'shape'):
                            print(f"   key: {k}, shape: {v.shape}")
                        else:
                            shape_info = np.array(v).shape
                            print(f"   key: {k}, shape: {shape_info}")
                else:
                    print("   obs is not a dictionary (type:", type(obs), ")")

                # Print natural language, reward, and done status
                print(f"   NaturalLang: '{natural_lang}', Reward: {reward}, Done: {done}")

                cumulative_reward += reward
                pbar.set_postfix({"reward": f"{cumulative_reward:.2f}"})

                if done:
                    break

                # Generate action
                action = self.model.eval_model(obs)
                send_pickle(self.sock, {"action": action})
                response = recv_pickle(self.sock)
                if response is None:
                    pbar.close()
                    print("[Error] Connection closed by server mid-episode.")
                    return
                current_response = response

            pbar.close()
            print(f"Episode {ep+1} finished with reward: {cumulative_reward:.2f}")

            # Reset for next episode
            reset_msg = {"reset": True}
            send_pickle(self.sock, reset_msg)
            current_response = recv_pickle(self.sock)
            time.sleep(0.1)

    def set_task(self, benchmark_name="libero_90", task_id=0, metadata=None):
        """
        Reconfigure the server's task (optional feature).
        """
        msg = {
            "set_task": {
                "benchmark_name": benchmark_name,
                "task_id": task_id
            }
        }
        if metadata is not None:
            msg["metadata"] = {str(k): str(v) for k, v in metadata.items() if v is not None}
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        print("[set_task] Server responded with new initial observation.")
        return response

    def close(self):
        self.sock.close()
        print("Connection closed.")

if __name__ == '__main__':
    evaluator = LiberoRemoteEvaluator(host='127.0.0.1', port=7700)
    evaluator.set_task("libero_goal", 0) 
    evaluator.evaluate(num_episodes=2, max_steps=1)
    evaluator.close()
