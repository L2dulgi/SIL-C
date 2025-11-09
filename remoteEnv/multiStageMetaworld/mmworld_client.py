# mmworld_client.py
'''
Client script to communicate with mmworld server.
Adapted from kitchen_client.py. citeturn0file0
'''
import socket
import pickle
import struct
import time
import numpy as np
from tqdm import tqdm
import argparse

class DummyModel:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def eval_model(self, observation):
        # Return random actions in [-1,1]
        return np.random.uniform(-1, 1, size=(self.action_dim,)).tolist()

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_pickle(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack('!I', len(data)))
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

class MMWorldRemoteEvaluator:
    def __init__(self, host='127.0.0.1', port=9999, action_dim=4):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        print(f'Connected to mmworld server at {host}:{port}')
        self.model = DummyModel(action_dim)

    def evaluate(self, num_episodes=3, max_steps=1000):
        response = recv_pickle(self.sock)
        if response is None:
            print('No data received. Connection closed.')
            return
        current_obs = response.get('observation') 
        for ep in range(num_episodes):
            cumulative_reward = 0.0
            pbar = tqdm(range(max_steps), desc=f'Episode {ep+1}', postfix={'reward': f'{cumulative_reward:.2f}'})
            for step in pbar:
                action = self.model.eval_model(current_obs)
                send_pickle(self.sock, {'action': action})
                response = recv_pickle(self.sock)
                if response is None:
                    pbar.close()
                    print('Connection closed by server.')
                    return
                reward = response.get('reward') or response.get('rewards')
                cumulative_reward += float(reward if isinstance(reward, (int, float)) else 0)
                done = response.get('done') or False
                pbar.set_postfix({'reward': f'{cumulative_reward:.2f}'})
                current_obs = response.get('observation')
                if done:
                    break
            pbar.close()
            print(f'Episode {ep+1} finished with reward: {cumulative_reward:.2f}')
            send_pickle(self.sock, {'reset': True})
            response = recv_pickle(self.sock)
            current_obs = response.get('observation') 
            time.sleep(0.1)

    def close(self):
        self.sock.close()
        print('Connection closed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMWorld Remote Evaluator Client')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--action-dim', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=600)
    args = parser.parse_args()
    client = MMWorldRemoteEvaluator(host=args.host, port=args.port, action_dim=args.action_dim)
    client.evaluate(num_episodes=args.episodes, max_steps=args.max_steps)