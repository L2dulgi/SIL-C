# mmworld_server.py
'''
Server script for mmworld environment with set_task support.
Adapted from kitchen_server.py. citeturn0file2 and requires mmworld.py citeturn0file3
'''
import socket
import pickle
import struct
import numpy as np
from multiprocessing import Process
import logging
from multiStageEnv import MultiStageTask, get_task_list_equal_easy

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(processName)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

# currently, the server only supports one env per time
def handle_client(conn, addr, seed=777):
    logging.info(f'Client connected from {addr}')
    # initialize with default skill list
    default_task = get_task_list_equal_easy()[0]['skill_list']
    env = MultiStageTask(seed=seed, skill_list=default_task)
    obs = env.reset()
    send_pickle(conn, {'observation': obs, 'reward': 0.0, 'done': False})
    logging.info(f'[{addr}] Sent initial observation.')

    while True:
        request = recv_pickle(conn)
        if request is None:
            logging.info(f'[{addr}] Connection closed by client.')
            break

        # handle set_task: client provides a new skill_list
        if 'set_task' in request:
            '''
            [
                {'data_name': 'door-button-drawer-puck'},
            ]
            '''
            new_skills = request['set_task']
            if len(new_skills) == 0 or len(new_skills) > 1:
                logging.error(f'[{addr}] Invalid task list: {new_skills}')
            current_task = new_skills[0]['data_name']
            current_skill_list = current_task.split('-')
            logging.info(f'[{addr}] Setting new task: {current_task}, skill_list: {current_skill_list}')
            env = MultiStageTask(seed=seed, skill_list=current_skill_list)
            obs = env.reset()
            send_pickle(conn, {'observation': obs, 'reward': 0.0, 'done': False})
            continue

        # handle reset command
        if request.get('reset', False):
            obs = env.reset()
            send_pickle(conn, {'observation': obs, 'reward': 0.0, 'done': False})
            continue

        # handle action command
        action = request.get('action')
        if action is None:
            continue
        action = np.array(action)
        if action.ndim != 1 :
            action = action.squeeze()

        obs, reward, done, info = env.step(np.array(action))
        send_pickle(conn, {'observation': obs, 'reward': float(reward), 'done': done})

    conn.close()
    logging.info(f'[{addr}] Client process ended.')

class BaseServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(10)
        logging.info(f'Server listening on {host}:{port}')

    def serve_forever(self):
        while True:
            conn, addr = self.sock.accept()
            p = Process(target=handle_client, args=(conn, addr))
            p.daemon = True
            p.start()

    def close(self):
        self.sock.close()
        logging.info('Server socket closed.')

if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='MMWorld Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8888)
    args = parser.parse_args()
    server = BaseServer(host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.close()
