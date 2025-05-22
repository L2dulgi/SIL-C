# .remoteEnv/kitchen/kitchen_client.py
import socket
import pickle
import struct
import time
import numpy as np
from tqdm import tqdm

# Dummy model that outputs a random action (assuming action dimension is 9)
class DummyModel:
    def eval_model(self, observation):
        return np.random.uniform(-1, 1, size=(9,)).tolist()

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

class KitchenRemoteEvaluator:
    """
    KitchenRemoteEvaluator provides an interface similar to the local Kitchen
    environment's evaluation API, but it internally communicates with a remote
    server. The server spawns a separate process for each client, so multiple
    evaluations can run concurrently.
    """
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm to reduce latency
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        self.model = DummyModel()

    def evaluate(self, num_episodes=3, max_steps=280):
        """
        Evaluate the remote environment for the specified number of episodes.
        Each episode runs for at most max_steps, with a tqdm progress bar displaying
        the current cumulative reward in the postfix.
        """
        init_response = recv_pickle(self.sock)
        if init_response is None:
            print("No data received. Connection closed.")
            return

        current_response = init_response

        for ep in range(num_episodes):
            cumulative_reward = 0.0
            pbar = tqdm(range(max_steps), desc=f"Episode {ep+1}", postfix={"reward": f"{cumulative_reward:.2f}"})
            for step in pbar:
                observation = current_response.get("observation")
                action = self.model.eval_model(observation)
                msg = {"action": action}
                send_pickle(self.sock, msg)
                response = recv_pickle(self.sock)
                if response is None:
                    pbar.close()
                    print("Connection closed by server.")
                    return
                reward = response.get("reward", 0.0)
                cumulative_reward += reward
                pbar.set_postfix({"reward": f"{cumulative_reward:.2f}"})
                current_response = response
                if response.get("done", False):
                    break
            pbar.close()
            print(f"Episode {ep+1} finished with reward: {cumulative_reward:.2f}")
            # Request environment reset for the next episode
            reset_msg = {"reset": True}
            send_pickle(self.sock, reset_msg)
            current_response = recv_pickle(self.sock)
            time.sleep(0.1)

    def close(self):
        self.sock.close()
        print("Connection closed.")

if __name__ == '__main__':
    evaluator = KitchenRemoteEvaluator(host='127.0.0.1', port=9999)
    evaluator.evaluate(num_episodes=10, max_steps=280)
    evaluator.close()
