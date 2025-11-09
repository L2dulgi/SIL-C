# .remoteEnv/kitchen/kitchen.py
import numpy as np
from d4rl.kitchen.kitchen_envs import KitchenBase, OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH
from contextlib import contextmanager
from typing import Optional, Tuple
from tqdm import tqdm

import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from dm_control.mujoco import engine
from scipy.spatial.transform import Rotation as R


# List of all valid tasks in the kitchen environment
all_tasks = [
    'bottom burner', 'top burner', 'light switch',
    'slide cabinet', 'hinge cabinet', 'microwave', 'kettle'
]

# KitchenTask class: manages a sequence of subtasks
class KitchenTask:
    def __init__(self, subtasks):
        # Validate each subtask against the list of all tasks
        for subtask in subtasks:
            if subtask not in all_tasks:
                raise ValueError(f'{subtask} is not valid subtask')
        self.subtasks = subtasks

    def __repr__(self):
        # Return a string representation of the task sequence
        return f"MTKitchenTask({' -> '.join(self.subtasks)})"

# KitchenEnv class: extends KitchenBase to simulate the kitchen environment
class KitchenEnv(KitchenBase):
    render_width = 224
    render_height = 224
    render_device = 1
    camera_offset = np.array([0.1, 0., -0.1])
    site_offset = np.array([0.07, 0.0, 0.])
    wrist_fovy = 75.0

    def __init__(self, *args, **kwargs):
        self.TASK_ELEMENTS = all_tasks  # for initialization
        self.TASK_ELEMENTS_TODO = all_tasks  # for initialization
        super().__init__(*args, **kwargs)
        self.task = None
        # self.TASK_ELEMENTS = all_tasks #  04
    
    def set_task_default(self,task , noise_scale=None) :
        if type(task) != KitchenTask:
            raise TypeError(f'task should be KitchenTask but {type(task)} is given')

        if noise_scale is not None:
            print(f"[ENV] Setting noise scale from {self.robot_noise_ratio} -> {noise_scale}")
            self.robot_noise_ratio = noise_scale

        # default goal task infomation of kitchen-mixed-v0
        subtasks = [ 'microwave', 'kettle', 'bottom burner', 'light switch']
        trained_task = KitchenTask(
            subtasks=subtasks,
        )
        print("Semantic Skill Seq : " , task)
        prev_task = self.task
        prev_task_elements = self.TASK_ELEMENTS
        
        self.task = trained_task
        self.TASK_ELEMENTS = trained_task.subtasks
        self.TASK_ELEMENTS_TODO = task.subtasks
        self.tasks_to_complete = task.subtasks

    @contextmanager
    def set_task(self, task):
        if type(task) != KitchenTask:
            raise TypeError(f'task should be KitchenTask but {type(task)} is given')

        # default goal task infomation of kitchen-mixed-v0
        subtasks = [ 'microwave', 'kettle', 'bottom burner', 'light switch']
        trained_task = KitchenTask(
            subtasks=subtasks,
        )
        print("Semantic Skill Seq : " , task)
        prev_task = self.task
        prev_task_elements = self.TASK_ELEMENTS
        
        self.task = trained_task
        self.TASK_ELEMENTS = trained_task.subtasks
        self.TASK_ELEMENTS_TODO = task.subtasks
        self.tasks_to_complete = task.subtasks
        yield
        self.task = prev_task
        self.TASK_ELEMENTS = prev_task_elements
        self.tasks_to_complete = prev_task_elements
        self.TASK_ELEMENTS_TODO = prev_task_elements
        
    def set_render_options(self, width, height, device, fps=30, frame_drop=1):
        self.render_width = width
        self.render_height = height
        self.render_device = device
        self.metadata['video.frames_per_second'] = fps
        self.metadata['video.frame_drop'] = frame_drop

    def _get_task_goal_todo(self):
        new_goal = np.zeros_like(self.goal)
        for element in self.TASK_ELEMENTS_TODO:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal
        return new_goal

    def compute_reward(self, obs_dict):
        reward_dict = {}
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']

        next_goal = self._get_task_goal_todo() 
        
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
            complete = distance < BONUS_THRESH
            if complete and all_completed_so_far:
                completions.append(element)
            all_completed_so_far = all_completed_so_far and complete
        for completion in completions:
            self.tasks_to_complete.remove(completion)
        
        reward = float(len(completions))
        return reward
    
    def reset_model(self):
        ret = super().reset_model()
        self.tasks_to_complete = list(self.TASK_ELEMENTS_TODO)
        return ret # ret

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        if not self.initializing:
            a = self.act_mid + a * self.act_amp

        self.robot.step(self, a, step_duration=self.skip * self.model.opt.timestep)

        obs = self._get_obs()
        reward = self.compute_reward(self.obs_dict)
        done = not self.tasks_to_complete
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
        }
        return obs, reward, done, env_info

    def render(
        self,
        mode: str = 'rgb_array',
        width: int = None,
        height: int = None,
    ):
        if mode == 'rgb_array':
            w = width if width is not None else self.render_width
            h = height if height is not None else self.render_height
            camera = engine.MovableCamera(self.sim, h, w)
            camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            return camera.render()
        return []

    def render_wrist_view(
        self,
        mode: str = 'rgb_array',
        width: int = None,
        height: int = None,
    ):
        """
        Render an RGB frame from a virtual camera tethered to the end-effector.

        Args:
            mode: Only 'rgb_array' is supported.
            width: Optional render width override; defaults to `self.render_width`.
            height: Optional render height override; defaults to `self.render_height`.

        Returns:
            np.ndarray: The wrist-view RGB image.
        """
        if mode == 'rgb_array':
            w = width if width is not None else self.render_width
            h = height if height is not None else self.render_height
            camera_id = self.sim.model.camera_name2id("eye_in_hand") # NOTE customized camera
            camera = engine.Camera(self.sim, h, w, camera_id=camera_id)
            return camera.render()
        else :
            return []


    def render_studio(
            self,
            mode : str = 'rgb_array',
            width: int = None,
            height: int = None,
    ) :
        if mode == 'rgb_array':
            w = width if width is not None else self.render_width
            h = height if height is not None else self.render_height

            cam_overview = engine.MovableCamera(self.sim, h, w)
            cam_ovens = engine.MovableCamera(self.sim, h, w)
            cam_kettle = engine.MovableCamera(self.sim, h, w)
            cam_pannels = engine.MovableCamera(self.sim, h, w)
            cam_cabinets = engine.MovableCamera(self.sim, h, w)
            wrist_camera_id = self.sim.model.camera_name2id("eye_in_hand") # NOTE customized camera
            cam_wrist = engine.Camera(self.sim, h, w, camera_id=wrist_camera_id)

            # Set camera poses for different viewpoints
            cam_overview.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            cam_ovens.set_pose(distance=1.2, lookat=[-0.6, 0.3, 2.0], azimuth=50, elevation=-35) # actually microwave kettle
            cam_pannels.set_pose(distance=1.2, lookat=[-0., 1.3, 2.3], azimuth=65, elevation=-0,)
            cam_cabinets.set_pose(distance=1.3, lookat=[-0., 0.9, 2.7], azimuth=40, elevation=-0,)

            # Render from all cameras
            img_overview = cam_overview.render()
            img_ovens = cam_ovens.render()
            img_pannels = cam_pannels.render()
            img_cabinets = cam_cabinets.render()
            img_wrist = cam_wrist.render()

            image_dict = {
                'overview': img_overview,
                'ovens': img_ovens,
                # 'kettle': img_kettle,
                'pannels': img_pannels,
                'cabinets': img_cabinets,
                'wrist': img_wrist,
            }
            return image_dict
        return []

    def get_ee_info(self) :
        ee_id = self.sim.model.site_name2id("end_effector")
        ee_pos = np.array(self.sim.data.site_xpos[ee_id], dtype=np.float64)
        
        R_mat = np.array(self.sim.data.site_xmat[ee_id]).reshape(3, 3)
        ee_ori = R.from_matrix(R_mat).as_rotvec()   # (3,)
        return ee_pos, ee_ori
    
import numpy as np
from kitchen import KitchenEnv, KitchenTask

class MultiKitchenEnv:
    def __init__(self, semantic_flag=False, eval_configure=None):
        # Use external evaluation configuration if provided; otherwise, use defaults.
        if eval_configure is None:
            self.eval_configure = [
                {'data_name': 'mbls'}, # Default evaluation configuration
            ]
        else:
            self.eval_configure = eval_configure

        # Mapping of initial characters to full task names.
        self.initial2task_dict = {
            'm': 'microwave',
            'k': 'kettle',
            'b': 'bottom burner',
            't': 'top burner',
            'l': 'light switch',
            's': 'slide cabinet',
            'h': 'hinge cabinet',
        }
        self.semantic_flag = semantic_flag
        self.possible_evaluation = [
            'mtlh', 'mlsh', 'mktl', 'mkth', 'mksh',
            'mkls', 'mklh', 'mkbs', 'mkbh', 'mbts',
            'mbtl', 'mbth', 'mbsh', 'mbls', 'ktls',
            'klsh', 'kbts', 'kbtl', 'kbth', 'kbsh',
            'kbls', 'kblh', 'btsh', 'btls',
        ]
        self.eval_horizons = 280
        self.noise_scale = None  # Initialize noise scale

        self.env_list = []
        self.task_list = []
        self.domain_configs = []
        self.evaluation_sequences = []
        
        # Build evaluation sequences based on eval_configure.
        self._build_evaluation_sequences()
        
        # Initialize environments and their corresponding tasks.
        self._initialize_environments()
        
        # If semantic_flag is True, load the skill embedding dictionary from a pickle file.
        if self.semantic_flag:
            with open('data/evolving_kitchen/kitchen_lang_embedding.pkl', 'rb') as f:
                self.skill_embedding = pickle.load(f)
        
        # Initialize observation list, done flags, cumulative rewards, and recorded frames for video.
        self.reset_model()
        self.recorded_frames = []  # List to store frames for the video

    def _build_evaluation_sequences(self):
        """Build evaluation sequences from the current eval_configure."""
        self.evaluation_sequences = []
        for phase_configure in self.eval_configure:
            tasks = phase_configure['data_name'].split(',')
            for task in tasks:
                task = task.strip()
                if 'domain' in phase_configure:
                    self.evaluation_sequences.append({
                        'task': task,
                        'domain': phase_configure['domain']
                    })
                else:
                    self.evaluation_sequences.append({'task': task})

    def _initialize_environments(self):
        """Reinitialize environment list, task list, and domain configurations."""
        self.env_list = []
        self.task_list = []
        self.domain_configs = []
        for task_configs in self.evaluation_sequences:
            env = KitchenEnv()  
            test_task = KitchenTask(subtasks=self.initial2task(task_configs['task']))
            env.set_task_default(test_task, self.noise_scale)
            self.env_list.append(env)
            self.task_list.append(test_task)
            self.domain_configs.append(task_configs.get('domain'))
            print("Initialized task:", test_task.subtasks, "domain:", task_configs.get('domain'))

    def set_task(self, new_eval_config, noise_scale=None):
        """
        Update the evaluation configuration with a new task setting.
        This rebuilds the evaluation sequences, reinitializes the environments,
        and resets the model.

        Args:
            new_eval_config (list): A list of dictionaries specifying new evaluation configurations.
                Example: [{'data_name': 'mtlh'}, {'data_name': 'mlsh'}]
            noise_scale (float, optional): Scale of Gaussian noise to add to observations.
                                         If None, no noise is added.
        """
        print("\n\n[MultiKitchenEnv] Setting new task configuration...")
        self.eval_configure = new_eval_config
        self.noise_scale = noise_scale
        
        self._build_evaluation_sequences()
        self._initialize_environments()
        self.reset_model()

    def initial2task(self, initial=None):
        """
        Convert the task identifier into a list of full task names using the mapping.
        """
        if initial is None:
            return None
        return [self.initial2task_dict[i] for i in initial]

    def reset_model(self):
        """
        Reset all environments and initialize observation list, done flags,
        cumulative rewards, and a dummy observation.
        """
        self.obs_list = []
        self.done_list = []
        self.episode_reward = np.zeros(len(self.env_list))
        for e_idx, env in enumerate(self.env_list):
            obs = env.reset_model()
            # If semantic_flag is True, append the initial semantic embedding.
            if self.semantic_flag:
                sidx = min(int(self.episode_reward[e_idx]), 3)
                semantic_vec = self.skill_embedding[self.task_list[e_idx].subtasks[sidx]]
                obs = np.concatenate([obs, semantic_vec], axis=-1)
            self.obs_list.append(obs)
            self.done_list.append(False)
        self.dummy_obs = np.zeros_like(self.obs_list[0])
    
    def step(self, actions):
        """
        Step through each environment with the corresponding action in the batch.
        If semantic_flag is True, append the semantic embedding to the observation.

        Args:
            actions (np.ndarray): Array of actions for each environment, shape (N, act_dim).

        Returns:
            tuple: (obs_list, rewards, done_list)
        """
        if actions.ndim == 1: # dim is 1 for single action handling
            actions = actions[np.newaxis, :]
        new_obs_list = []
        rewards = np.zeros(len(self.env_list))
        for e_idx, env in enumerate(self.env_list):
            if self.done_list[e_idx]:
                new_obs_list.append(self.dummy_obs)
                continue
            obs, rew, done, _ = env.step(actions[e_idx].squeeze())
            self.episode_reward[e_idx] += rew
            rewards[e_idx] = rew
            if done:
                self.done_list[e_idx] = True

            if self.semantic_flag:
                sidx = min(int(self.episode_reward[e_idx]), 3)
                semantic_vec = self.skill_embedding[self.task_list[e_idx].subtasks[sidx]]
                obs = np.concatenate([obs, semantic_vec], axis=-1)
            new_obs_list.append(obs)
        self.obs_list = new_obs_list
        return new_obs_list, rewards, self.done_list

    def render(self):
        """
        Render a grid of environment frames and return the combined image.
        """
        images = []
        for env in self.env_list:
            img = env.render()
            img = cv2.resize(img, (224, 224))
            images.append(img)
        
        num_images = len(images)
        num_cols = min(4, num_images)
        num_rows = math.ceil(num_images / 4)
        
        total_slots = num_rows * 4
        if num_images < total_slots:
            black_image = np.zeros((224, 224, 3), dtype=np.uint8)
            images.extend([black_image] * (total_slots - num_images))
        
        rows = []
        for i in range(num_rows):
            row_images = images[i*4:(i+1)*4]
            row = np.hstack(row_images)
            rows.append(row)
        grid_image = np.vstack(rows)
        self.recorded_frames.append(grid_image)
        return grid_image

    def render_per_env(self, resize_shape=(224, 224)):
        """
        Render each environment individually.

        Args:
            resize_shape (tuple[int, int] | None): Optional (width, height) to resize frames.

        Returns:
            list[np.ndarray | None]: List of RGB frames (uint8) for each environment.
        """
        frames = []
        for env in self.env_list:
            frame = env.render()
            if frame is None:
                frames.append(None)
                continue
            frame = np.asarray(frame)
            if resize_shape is not None:
                frame = cv2.resize(frame, resize_shape)
            frames.append(frame.astype(np.uint8, copy=False))
        return frames

    def save_video(self, video_filename, fps=30):
        """
        Save the recorded frames as a video file.
        """
        if not self.recorded_frames:
            print("No frames recorded.")
            return
        
        frame_height, frame_width = self.recorded_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        
        for frame in self.recorded_frames:
            out.write(frame)
        
        out.release()
        print(f"Video saved as {video_filename}")


if __name__ == "__main__":
    # Minimal render smoke test leveraging the KitchenBase superclass implementation.
    env = KitchenEnv()
    try:
        env.reset_model()
        frame = env.render()
        wrist_frame = env.render_wrist_view()
        custom_wrist = env.render_wrist_view()
        if wrist_frame is None:
            print("Render returned None")
        else:
            pass
            # wrist_frame = np.asarray(wrist_frame)
            # print(f"Render produced frame with shape {wrist_frame.shape} and dtype {wrist_frame.dtype}")
            # cv2.imwrite("kitchen_wrist_smoke.png", cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR))
        # if custom_wrist is not None:
        #     custom_wrist = np.asarray(custom_wrist)
        #     cv2.imwrite(
        #         "kitchen_wrist_smoke_offset.png",
        #         cv2.cvtColor(custom_wrist, cv2.COLOR_RGB2BGR),
        #     )

        # Test render_studio method
        print("\n=== Testing render_studio ===")
        studio_images = env.render_studio()
        if studio_images:
            print(f"Studio camera IDs: {list(studio_images.keys())}")

            # Prepare images with camera ID labels
            labeled_images = {}
            for cam_id, img in studio_images.items():
                if img is not None:
                    img = np.asarray(img).copy()
                    print(f"  {cam_id}: shape={img.shape}, dtype={img.dtype}")

                    # Add camera ID text to the image
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.putText(img_bgr, cam_id.upper(), (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                    # Store labeled image for grid (convert back to RGB)
                    labeled_images[cam_id] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Concatenate all images into a grid
            sorted_keys = sorted(labeled_images.keys())
            images_list = [labeled_images[key] for key in sorted_keys]

            num_images = len(images_list)

            # Arrange in 2 rows: 3 in first row, remaining in second row with padding
            if num_images >= 3:
                row1 = np.hstack(images_list[:3])
                if num_images > 3:
                    # Stack remaining images
                    remaining_images = images_list[3:]
                    row2 = np.hstack(remaining_images)
                    # Pad row2 to match row1 width
                    if row2.shape[1] < row1.shape[1]:
                        pad_width = row1.shape[1] - row2.shape[1]
                        padding = np.zeros((row2.shape[0], pad_width, 3), dtype=row2.dtype)
                        row2 = np.hstack([row2, padding])
                    grid_image = np.vstack([row1, row2])
                else:
                    grid_image = row1
            else:
                # Less than 3 images, just stack horizontally
                grid_image = np.hstack(images_list)

            # Save concatenated grid only
            cv2.imwrite("kitchen_studio_grid.png", cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
            print(f"Saved concatenated grid: shape={grid_image.shape}")
            print(f"  Grid layout ({num_images} cameras):")
            if num_images >= 3:
                print(f"    Row 1: {sorted_keys[:3]}")
                if num_images > 3:
                    print(f"    Row 2: {sorted_keys[3:]}")
            else:
                print(f"    Single row: {sorted_keys}")

    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
