import pickle
import numpy as np
from itertools import permutations, product
from contextlib import contextmanager
from tqdm import tqdm
import random as py_rand
import gym
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# MetaWorld environment imports
# try:
    
# except ImportError:
#     print("mmworld not installed")

from mmworld.envs.mujoco.sawyer_xyz.v2.sawyer_non_stationary_v2 import SawyerNonStationaryEnvV2


# from clus.env.base_evaluator import BaseEvaluator
# from clus.env.continual_config import *

# --------------------------------------
# Task list generation functions
# --------------------------------------
def get_task_list_equal_easy(all_task_flag='full'):
    task_sets = [['puck', 'drawer', 'button', 'door']]
    ss = [0, 3, 4, 7, 8, 11]
    bound = 12
    if all_task_flag == 'full':
        ss = list(range(12))
    if all_task_flag == 'half':
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == 'third':
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == 'sixth':
        ss = [0, 10, 13, 23]
        bound = 24
    task_shuffled = []
    for task_set in task_sets:
        for i, d in enumerate(permutations(task_set)):
            if i % bound in ss:
                task_shuffled.append({
                    'skill_list': list(task_set),
                    'skill_seq': list(d)
                })
    return task_shuffled


def get_task_list_equal_normal(all_task_flag='full', only_normal=False):
    task_sets = []
    for combo in product(('box', 'puck'), ('handle', 'drawer'), ('button', 'lever'), ('door', 'stick')):
        if 'box' in combo or 'stick' in combo:
            continue
        if only_normal and 'handle' not in combo and 'lever' not in combo:
            continue
        task_sets.append(list(combo))
    ss = [0, 3, 4, 7, 8, 11]
    bound = 12
    if all_task_flag == 'full':
        ss = list(range(12))
    if all_task_flag == 'half':
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == 'third':
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == 'sixth':
        ss = [0, 10, 13, 23]
        bound = 24
    task_shuffled = []
    for task_set in task_sets:
        for i, d in enumerate(permutations(task_set)):
            if i % bound in ss:
                task_shuffled.append({
                    'skill_list': list(task_set),
                    'skill_seq': list(d)
                })
    return task_shuffled


def get_task_list_equal_hard(all_task_flag='full'):
    task_sets = []
    for combo in product(('box', 'puck'), ('handle', 'drawer'), ('button', 'lever'), ('door', 'stick')):
        task_sets.append(list(combo))
    ss = [0, 3, 4, 7, 8, 11]
    bound = 12
    if all_task_flag == 'full':
        ss = list(range(12))
    if all_task_flag == 'half':
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == 'third':
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == 'sixth':
        ss = [0, 10, 13, 23]
        bound = 24
    task_shuffled = []
    for task_set in task_sets:
        for i, d in enumerate(permutations(task_set)):
            if i % bound in ss:
                task_shuffled.append({
                    'skill_list': list(task_set),
                    'skill_seq': list(d)
                })
    return task_shuffled


def configs_task_list(configs):
    task_shuffled = get_task_list_equal_hard()
    for task in task_shuffled:
        task['data_name'] = "-".join(task['skill_seq'])
    task_refined = []
    for phase in configs:
        for tasks in phase['data_name'].split(','):
            if isinstance(tasks, str):
                tasks = [tasks]
            for task in tasks:
                for task_dict in task_shuffled:
                    if task == task_dict['data_name']:
                        task_refined.append(task_dict)
                        break
    return task_refined

class MultiStageTask(gym.Env):
    def __init__(self, seed: int, skill_list, obs_type='sensor', max_episode_length=1000, partially_observable=False):
        py_rand.seed(seed)
        self.env = SawyerNonStationaryEnvV2(skill_list)
        self.max_episode_length = max_episode_length
        self.time_steps = 0
        self.obs_type = obs_type
        self.partially_observable = partially_observable

        if self.obs_type in ('vision', 'mixed'):
            self.env._partially_observable = True
        else:
            self.env._partially_observable = partially_observable

        if self.obs_type == 'vision':
            self.observation_space = gym.spaces.Box(low=np.zeros((80, 80, 3)), 
                                                   high=np.ones((80, 80, 3)), 
                                                   dtype=np.uint8)
        elif self.obs_type == 'mixed':
            self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(low=np.zeros((80, 80, 3)), 
                                        high=np.ones((80, 80, 3)), 
                                        dtype=np.uint8),
                'sensor': self.env.observation_space
            })
        else:
            self.observation_space = self.env.observation_space

        self.action_space = self.env.action_space

    def step(self, action, action_noise=None):
        if action_noise is not None:
            sensor_obs, reward, done, info = self.env.step(action, action_noise)
        else:
            sensor_obs, reward, done, info = self.env.step(action)
        self.time_steps += 1
        if self.time_steps >= self.max_episode_length:
            done = True

        if self.obs_type == 'vision':
            obs = self.render()
        elif self.obs_type == 'mixed':
            obs = {'image': self.render(), 'sensor': sensor_obs}
        else:
            obs = sensor_obs

        info['action_noise'] = action_noise
        return obs, reward, done, info

    def reset(self):
        sensor_obs = self.env.reset()
        self.time_steps = 0
        if self.obs_type == 'vision':
            return self.render()
        elif self.obs_type == 'mixed':
            return {'image': self.render(), 'sensor': sensor_obs}
        return sensor_obs

    def render(self, mode='corner3', resolution=(224, 224)):
        return self.env.render(offscreen=True, resolution=resolution, camera_name=mode)


# --------------------------------------
# Multi-task evaluator
# --------------------------------------
class MMEvaluator:
    def __init__(
        self,
        base_evaluation_sequences,
        eval_mode='obs',
        traj_length=10,
        eval_episodes=3,
        phase_configures=None,
    ):
        print("[MMevaluator]")
        skill_embedding_path = 'data/continual_dataset/evolving_world/mm_lang_embedding.pkl'
        with open(skill_embedding_path, 'rb') as f:
            self.skill_semantics = pickle.load(f)

        self.eval_horizons = 600
        self.base_evaluation_sequences = base_evaluation_sequences
        self.threshold = len(self.base_evaluation_sequences)
        self.eval_mode = eval_mode
        self.traj_length = traj_length
        self.eval_episodes = eval_episodes

        self.env_list = []
        for idx, task in enumerate(tqdm(self.base_evaluation_sequences)):
            env = SingleTask(seed=777, skill_list=task['skill_list'])
            env.env.skill_list = task['skill_seq']
            self.env_list.append(env)
            if len(self.env_list) < self.threshold:
                continue

    def evaluate_base(self, model, eval_fn=None):
        eval_episodes = self.eval_episodes
        rew_info = {'skill_seq': [], 'skill_rew': []}
        eval_fn = model.eval_model if eval_fn is None else eval_fn

        for eval_seed in range(eval_episodes):
            obs_list, done_list, skill_idx_list = [], [], []
            for env in self.env_list:
                obs = env.reset()
                obs_list.append(obs)
                done_list.append(False)
                skill_idx_list.append(0)
            dummy_obs = np.zeros_like(obs_list[0])

            for _ in tqdm(range(self.eval_horizons)):
                skill_semantics_list = [self.skill_semantics[env.env.skill_list[min(idx, 3)]]
                                        for idx, env in enumerate(self.env_list)]
                obs = np.concatenate([obs_list, skill_semantics_list], axis=-1)
                actions = np.array(eval_fn(obs[:, None, :]))

                obs_list_new = []
                for e_idx, env in enumerate(self.env_list):
                    if done_list[e_idx]:
                        obs_list_new.append(dummy_obs)
                        continue
                    obs, rew, done, env_info = env.step(actions[e_idx].squeeze())
                    obs_list_new.append(obs)
                    if env_info.get('success', 0) == 1:
                        skill_idx_list[e_idx] += 1
                        if done:
                            done_list[e_idx] = True
                obs_list = obs_list_new
                if all(done_list):
                    break

            for env in self.env_list:
                seq = env.env.skill_list
                reward_sum = int(env.env.mode)
                rew_info['skill_seq'].append(seq)
                rew_info['skill_rew'].append(reward_sum)

        total = 0
        for i, seq in enumerate(rew_info['skill_seq']):
            rew_info['skill_rew'][i] /= eval_episodes
            print(f"[...]{seq} rew: {rew_info['skill_rew'][i]:.2f}")
            total += rew_info['skill_rew'][i]
        print("total reward:", total / len(rew_info['skill_seq']))
        return rew_info

    def evaluate_base_vid(self, model, eval_fn=None):
        # video-based evaluation kept unchanged
        ...

    def multi_metaworld_evaluate(self, model, evaluation_sequences, eval_episodes=3):
        # multi-task evaluation kept unchanged
        ...

if __name__ == '__main__':
    # Example usage:
    # mwloader = MMEvaluator(base_evaluation_sequences=<your_sequences>)
    # mwloader.evaluate_base(None, eval_fn=lambda x: np.random.uniform(-1, 1, (len(x), 4)))
    pass
