import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio

# Import envs first to initialize isaacgym modules
from envs import *

# Import torch and utils after isaacgym modules are initialized
import torch
import torch.nn.functional as F
from utils.model import *
from utils.buffer import ExperienceBuffer
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder

# Dynamic task class loading
import importlib
import inspect

def get_task_class(task_name):
    """
    Dynamically load task class by name.
    Searches through all modules in the envs package for classes that match the task name.
    Handles different naming conventions (Base_Walk vs BaseWalk, etc.)
    """
    # Generate possible class name variations
    possible_names = [task_name]
    
    # Handle underscore to camelCase conversion (Base_Walk -> BaseWalk)
    if '_' in task_name:
        camel_case = ''.join(word.capitalize() for word in task_name.split('_'))
        possible_names.append(camel_case)
    
    # Handle camelCase to underscore conversion (BaseWalk -> Base_Walk)
    if not '_' in task_name and any(c.isupper() for c in task_name[1:]):
        import re
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', task_name).lower()
        snake_case = snake_case[0].upper() + snake_case[1:]  # Capitalize first letter
        possible_names.append(snake_case)
    
    # First try to get from the envs module (which imports all task classes)
    try:
        envs_module = importlib.import_module('envs')
        for name, obj in inspect.getmembers(envs_module):
            if inspect.isclass(obj) and name in possible_names:
                return obj
    except Exception as e:
        print(f"Error loading from envs module: {e}")
    
    # If not found, try to import from specific paths
    task_paths = [
        f"envs.T1.{task_name.lower()}",
        f"envs.K1.{task_name.lower()}",
        f"envs.{task_name}",
    ]
    
    for path in task_paths:
        try:
            module = importlib.import_module(path)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name in possible_names:
                    return obj
        except ImportError:
            continue
        except Exception as e:
            print(f"Error loading from {path}: {e}")
            continue
    
    return None


class Runner:

    def __init__(self, test=False):
        self.test = test
        # prepare the environment
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        task_name = self.cfg["basic"]["task"]
        # Extract task name from path (e.g., "T1/T1" -> "T1")
        if "/" in task_name:
            task_name = task_name.split("/")[-1]
        
        # Dynamically load the task class
        task_class = get_task_class(task_name)
        if task_class is None:
            raise ValueError(f"Unknown task: {task_name}. Could not find a class named '{task_name}' in the envs package.")
        
        self.env = task_class(self.cfg)

        self.device = self.cfg["basic"]["rl_device"]
        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.model = ActorCritic(self.env.num_actions, self.env.num_obs, self.env.num_privileged_obs).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._load()

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        if not self.test:
            self.cfg["viewer"]["record_video"] = False

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load(self):
        if not self.cfg["basic"]["checkpoint"]:
            return
        if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
            # Look for models in hierarchical structure: logs/robot_type/task_name/**/*.pth
            task_name = self.cfg["basic"]["task"]
            robot_type = self._get_robot_type(task_name)
            
            # First try: exact task in robot-specific folder
            task_log_pattern = os.path.join("logs", robot_type, task_name, "**/*.pth")
            task_models = sorted(glob.glob(task_log_pattern, recursive=True), key=os.path.getmtime)
            
            if task_models:
                self.cfg["basic"]["checkpoint"] = task_models[-1]
            else:
                # Second try: any task in robot-specific folder
                robot_log_pattern = os.path.join("logs", robot_type, "**/*.pth")
                robot_models = sorted(glob.glob(robot_log_pattern, recursive=True), key=os.path.getmtime)
                
                if robot_models:
                    self.cfg["basic"]["checkpoint"] = robot_models[-1]
                else:
                    # Fallback: all logs if no robot-specific models found
                    self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)
        try:
            self.env.curriculum_prob = model_dict["curriculum"]
        except Exception as e:
            print(f"Failed to load curriculum: {e}")
        try:
            self.optimizer.load_state_dict(model_dict["optimizer"])
        except Exception as e:
            print(f"Failed to load optimizer: {e}")

    def train(self):
        self.recorder = Recorder(self.cfg)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        for it in range(self.cfg["basic"]["max_iterations"]):
            # within horizon_length, env.step() is called with same act
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                with torch.no_grad():
                    dist = self.model.act(obs)
                    act = dist.sample()
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                self.buffer.update_data("actions", n, act)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])
                self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))

            with torch.no_grad():
                old_dist = self.model.act(self.buffer["obses"])
                old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

            mean_value_loss = 0
            mean_actor_loss = 0
            mean_bound_loss = 0
            mean_entropy = 0
            for n in range(self.cfg["runner"]["mini_epochs"]):
                values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
                last_values = self.model.est_value(obs, privileged_obs)
                with torch.no_grad():
                    self.buffer["rewards"][self.buffer["time_outs"]] = values[self.buffer["time_outs"]]
                    advantages = discount_values(
                        self.buffer["rewards"],
                        self.buffer["dones"] | self.buffer["time_outs"],
                        values,
                        last_values,
                        self.cfg["algorithm"]["gamma"],
                        self.cfg["algorithm"]["lam"],
                    )
                    returns = values + advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                value_loss = F.mse_loss(values, returns)

                dist = self.model.act(self.buffer["obses"])
                actions_log_prob = dist.log_prob(self.buffer["actions"]).sum(dim=-1)
                actor_loss = surrogate_loss(old_actions_log_prob, actions_log_prob, advantages)

                bound_loss = torch.clip(dist.loc - 1.0, min=0.0).square().mean() + torch.clip(dist.loc + 1.0, max=0.0).square().mean()

                entropy = dist.entropy().sum(dim=-1)

                loss = (
                    value_loss
                    + actor_loss
                    + self.cfg["algorithm"]["bound_coef"] * bound_loss
                    + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(dist.scale / old_dist.scale)
                        + 0.5 * (torch.square(old_dist.scale) + torch.square(dist.loc - old_dist.loc)) / torch.square(dist.scale)
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                mean_value_loss += value_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_bound_loss += bound_loss.item()
                mean_entropy += entropy.mean()
            mean_value_loss /= self.cfg["runner"]["mini_epochs"]
            mean_actor_loss /= self.cfg["runner"]["mini_epochs"]
            mean_bound_loss /= self.cfg["runner"]["mini_epochs"]
            mean_entropy /= self.cfg["runner"]["mini_epochs"]
            self.recorder.record_statistics(
                {
                    "value_loss": mean_value_loss,
                    "actor_loss": mean_actor_loss,
                    "bound_loss": mean_bound_loss,
                    "entropy": mean_entropy,
                    "kl_mean": kl_mean,
                    "lr": self.learning_rate,
                    "curriculum/mean_lin_vel_level": self.env.mean_lin_vel_level,
                    "curriculum/mean_ang_vel_level": self.env.mean_ang_vel_level,
                    "curriculum/max_lin_vel_level": self.env.max_lin_vel_level,
                    "curriculum/max_ang_vel_level": self.env.max_ang_vel_level,
                },
                it,
            )

            if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                self.recorder.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "curriculum": self.env.curriculum_prob,
                    },
                    it + 1,
                )
            print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))

    def play(self):
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        while True:
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.loc
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
            if self.cfg["viewer"]["record_video"]:
                record_time -= self.env.dt
                if record_time < 0:
                    record_time += self.cfg["viewer"]["record_interval"]
                    self.interrupt = False
                    signal.signal(signal.SIGINT, self.interrupt_handler)
                    with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as self.writer:
                        for frame in self.env.camera_frames:
                            self.writer.append_data(frame)
                    if self.interrupt:
                        raise KeyboardInterrupt
                    signal.signal(signal.SIGINT, signal.default_int_handler)

    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True

    def _get_robot_type(self, task_name):
        """Determine robot type from task name."""
        # Check if task name starts with K1 or T1
        if task_name.startswith("K1"):
            return "K1"
        elif task_name.startswith("T1"):
            return "T1"
        else:
            # Default fallback - could be extended for other robot types
            return "Unknown"
