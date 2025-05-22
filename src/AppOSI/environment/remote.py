import socket
import pickle
import struct
import time
import numpy as np
from tqdm import tqdm

import typing
import os

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
    it save previous N trajectoris and return create the state using the trajectories.
    """
    def __init__(self, history_len=5):
        self.history_len = history_len
        self.history = []

    def add(self, obs):
        if len(self.history) >= self.history_len:
            self.history.pop(0)
        self.history.append(obs)

    def get_state(self, obs):
        self.add(obs)
        return np.concatenate( [self.history[0], self.history[-1]], axis=-1)
    
class HistoryEvalHelperTriple(HistoryEvalHelper):
    def get_state(self, obs):
        self.add(obs)
        first = self.history[0]
        last  = self.history[-1]
        mid_idx = len(self.history) // 2
        mid     = self.history[mid_idx]
        return np.concatenate([first, mid, last], axis=-1)


class KitchenRemoteEvaluator:
    """
    KitchenRemoteEvaluator communicates with a remote server that
    performs multi-environment evaluations.
    """
    def __init__(self, host=remote_server_ip, port=9999, obs_helper=None, eval_fn=None):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm to reduce latency.
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        print(f"Connected to server at {host}:{port}")

        # ----------------------------
        # Set the evaluation function
        # ----------------------------
        self.obs_helper = obs_helper
        self.eval_fn = eval_fn # Function to evaluate observations and return actions.

    def set_task(self, eval_configure):
        """
        Send a task-setting command to the remote server to update the evaluation configuration.
        
        Args:
            eval_configure (list): A list of dictionaries specifying new evaluation configurations.
                                For example:
                                [
                                    {'data_name': 'mbls'},
                                    {'data_name': 'mktl'},
                                ]
        Returns:
            The response from the server (new initial observations) if available.
        """
        msg = {"set_task": eval_configure}
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        if response is not None:
            print("Task settings updated on server.")
            # Save the current evaluation tasks for later use in evaluate().
            self.current_eval_tasks = eval_configure
            return response
        else:
            print("No response received for task setting.")
            return None

    def evaluate(self, num_episodes=3, max_steps=280):
        """
        Evaluate the remote multi-environment for the specified number of episodes.
        Each episode runs for at most max_steps. If evaluation tasks have been set,
        this function tracks and returns rewards per task.
        
        Returns:
            A tuple (eval_rewards, eval_dict) where:
            - eval_rewards is a list of overall mean rewards per episode (global average computed from tasks if active).
            - eval_dict is a dictionary mapping each task name to a dictionary containing:
                    "episode_rewards": list of rewards per episode for that task,
                    "avg_reward": average reward across episodes for that task.
            If no evaluation tasks are active, a default key "default" is used.
        """
        # Receive initial response from the server.
        init_response = recv_pickle(self.sock)
        if init_response is None:
            print("No data received. Connection closed.")
            return

        current_response = init_response
        num_envs = len(current_response.get("observations", []))

        # Determine if evaluation tasks are active (set via set_task)
        tasks_active = hasattr(self, 'current_eval_tasks') and self.current_eval_tasks is not None
        if tasks_active:
            # Extract task names from the evaluation configuration.
            task_names = [task.get("data_name", f"task_{i}") for i, task in enumerate(self.current_eval_tasks)]
            # Initialize eval_dict to store a list of episode rewards per task.
            eval_dict = {task_name: [] for task_name in task_names}
        else:
            eval_dict = {"default": []}

        # List to store overall mean reward per episode.
        eval_rewards = []

        for ep in range(num_episodes):
            if tasks_active:
                # Initialize cumulative rewards for each task.
                cumulative_rewards_dict = {task_name: 0.0 for task_name in task_names}
            else:
                cumulative_reward = 0.0

            pbar = tqdm(range(max_steps), desc=f"Episode {ep+1}",
                        postfix={"reward": "0.00"})
            


            for step in pbar:
                observations = np.array(current_response.get("observations", []))
                if observations.ndim == 1:  # Handle single observation
                    observations = np.expand_dims(observations, axis=0)
                
                # helper function to create the state
                if self.obs_helper is not None:
                    observations = self.obs_helper.get_state(observations)
                actions = self.eval_fn(observations)
                actions = np.array(actions).tolist()
                msg = {"action": actions}
                send_pickle(self.sock, msg)
                response = recv_pickle(self.sock)
                if response is None:
                    pbar.close()
                    print("Connection closed by server.")
                    return
                raw_rewards = response.get("rewards", None)
                if tasks_active:
                    # Assume raw_rewards is a dict mapping each task to a list of rewards.
                    if isinstance(raw_rewards, dict):
                        for task_name in task_names:
                            # Get rewards for the current task; default to zeros if not provided.
                            task_reward_list = raw_rewards.get(task_name, [0.0] * num_envs)
                            cumulative_rewards_dict[task_name] += np.mean(task_reward_list)
                    else:
                        # If raw_rewards is not a dict, use the same reward for all tasks.
                        reward_val = np.mean(raw_rewards) if raw_rewards is not None else 0.0
                        for task_name in task_names:
                            cumulative_rewards_dict[task_name] += reward_val
                    # Update progress bar with each task's cumulative reward.
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
                # Compute global average reward for the episode by averaging rewards from all tasks.
                global_reward = np.mean(list(cumulative_rewards_dict.values()))
                eval_rewards.append(global_reward)
                # Log episode rewards for each task.
                for task_name in task_names:
                    print(f"Episode {ep+1} for task '{task_name}' finished with mean reward: {cumulative_rewards_dict[task_name]:.2f}")
                    eval_dict[task_name].append(cumulative_rewards_dict[task_name])
                print(f"Episode {ep+1} finished with global mean reward: {global_reward:.2f}")
            else:
                print(f"Episode {ep+1} finished with total mean reward: {cumulative_reward:.2f}")
                eval_rewards.append(cumulative_reward)
                eval_dict["default"].append(cumulative_reward)

            # Request environment reset for the next episode.
            reset_msg = {"reset": True}
            send_pickle(self.sock, reset_msg)
            current_response = recv_pickle(self.sock)
            time.sleep(0.1)

        if tasks_active:
        # Compute average reward per task across episodes.
            for task_name in task_names:
                task_rewards = eval_dict[task_name]
                avg_reward = np.mean(task_rewards) if task_rewards else 0.0
                # Round each episode reward and the average to 3 decimal places.
                rounded_rewards = [round(r, 3) for r in task_rewards]
                avg_reward = round(avg_reward, 3)
                eval_dict[task_name] = {"episode_rewards": rounded_rewards, "avg_reward": avg_reward}
        else:
            default_rewards = eval_dict["default"]
            avg_reward = np.mean(default_rewards) if default_rewards else 0.0
            rounded_rewards = [round(r, 3) for r in default_rewards]
            avg_reward = round(avg_reward, 3)
            eval_dict["default"] = {"episode_rewards": rounded_rewards, "avg_reward": avg_reward}

        return eval_rewards, eval_dict

    def close(self):
        self.sock.close()
        print("Connection closed.")

class MMWorldRemoteEvaluator:
    def __init__(self, host=remote_server_ip, port=8888, obs_helper=None, eval_fn=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        print(f'Connected to mmworld server at {host}:{port}')
        self.obs_helper = obs_helper
        self.eval_fn = eval_fn

    def set_task(self, eval_configure):
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
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        if response is not None:
            print("Task settings updated on server.")
            # Save the current evaluation tasks for later use in evaluate().
            self.current_eval_tasks = eval_configure
            return response
        else:
            print("No response received for task setting.")
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
            print("No data received. Connection closed.")
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
                send_pickle(self.sock, {"action": action})
                response = recv_pickle(self.sock)
                if response is None:
                    print("Connection closed by server.")
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
                    print(f"Episode {ep + 1} – task '{name}': {cumulative_rewards[name]:.2f}")
                print(f"Episode {ep + 1} – global reward: {global_reward:.2f}\n")
            else:
                eval_rewards.append(cumulative_reward)
                eval_dict["default"].append(cumulative_reward)  # type: ignore[arg-type]
                print(f"Episode {ep + 1} reward: {cumulative_reward:.2f}\n")

            # --------------------------------------------------------------------
            # Reset env for next episode ----------------------------------------
            send_pickle(self.sock, {"reset": True})
            response = recv_pickle(self.sock)
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
        print('Connection closed.')

class LiberoRemoteEvaluator:
    """
    LiberoRemoteEvaluator communicates with a remote server and uses
    a user-provided eval_fn to compute actions. A debug flag toggles
    verbose printing of observations, rewards, and actions.
    """
    def __init__(self, host=remote_server_ip, port=7777, obs_helper=None ,eval_fn=None, debug=False):
        self.host = host
        self.port = port
        self.debug = debug
        
        self.obs_helper = obs_helper
        self.eval_fn = eval_fn 

        # Setup socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        print(f"Connected to Libero server at {host}:{port}")


    def set_task(self, eval_configure) :
        # benchmark_name="libero_90", task_id=0):
        """
        Optionally reset the task on the server.
        Args:
            eval_configure (list): A list of dictionaries specifying new evaluation configurations.
                For example:
                [
                    {'data_name': 'libero_90-0'},
                ]
        """
        if len(eval_configure) == 0 :
            print("[Error] No evaluation configuration provided.")
            return
        elif len(eval_configure) > 1 :
            print("[Error] Multiple evaluation configurations provided. Only the first one will be used.")
        eval_task_str = eval_configure[0].get("data_name", "libero_90-0")
        benchmark_name, task_id = eval_task_str.split("-")
        task_id = int(task_id)
        msg = {"set_task": {"benchmark_name": benchmark_name, "task_id": task_id}}
        send_pickle(self.sock, msg)
        response = recv_pickle(self.sock)
        if self.debug:
            print("[set_task] Server response:", response)
        return response

    def evaluate(self, num_episodes=3, max_steps=50):
        """
        Run evaluation for a given number of episodes and steps.

        Returns:
            List of cumulative rewards per episode.
        """
        init_response = recv_pickle(self.sock)
        if init_response is None:
            print("[Error] No initial data. Connection closed.")
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

                # print(obs.keys())
                # Wrap the observations for state (NOTE hard-coded for now)
                state_obs = np.concatenate([obs.get("robot0_gripper_qpos"), obs.get("robot0_joint_pos") ,obs.get("object-state")])
                # state_obs = np.concatenate([obs.get("robot0_joint_pos"),obs.get("robot0_gripper_qpos") ,obs.get("object-state")])
                # state_obs = np.array(obs)
                if state_obs.shape[0] < 130 :
                    state_obs = np.concatenate([state_obs, np.zeros((130-state_obs.shape[0],))])
                
                if self.obs_helper is not None:
                    state_obs = self.obs_helper.get_state(state_obs)
                
                action = self.eval_fn(state_obs[None,:])
                # print(f"action shape: {action}")

                send_pickle(self.sock, {"action": action})
                response = recv_pickle(self.sock)
                if response is None:
                    pbar.close()
                    print("[Error] Connection closed mid-episode.")
                    return episode_rewards
                current = response

                cumulative_reward += reward
                pbar.set_postfix({"reward": f"{cumulative_reward:.2f}"})

                if self.debug:
                    print(f"\n[Step {step}] Observation: {type(obs)}")
                    if isinstance(obs, dict):
                        for k, v in obs.items():
                            shape = getattr(v, 'shape', np.array(v).shape)
                            print(f"  {k}: shape {shape}")
                    print(f"  Action: {action}")
                    print(f"  Reward: {reward}, Done: {done}\n")

                if done:
                    break

            pbar.close()
            episode_rewards.append(cumulative_reward)
            print(f"Episode {ep+1} finished with reward: {cumulative_reward:.2f}")

            # Reset for next episode
            send_pickle(self.sock, {"reset": True})
            current = recv_pickle(self.sock)
            time.sleep(0.1)

        return episode_rewards, None # current Eval_dict is not used in this case

    def close(self):
        self.sock.close()
        print("Connection closed.")


# Example usage:
if __name__ == '__main__':
    evaluator = KitchenRemoteEvaluator(host='127.0.0.1', port=9999)
    # Example: change the task settings remotely before evaluation.
    new_eval_config = [
            {'data_name': 'btls'},
            {'data_name': 'mbls'},
            {'data_name': 'mkbh'},
    ]
    evaluator.set_task(new_eval_config)
    evaluator.evaluate(num_episodes=1, max_steps=280)
    evaluator.close()
