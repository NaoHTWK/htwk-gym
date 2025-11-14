import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import wandb
import yaml


class Recorder:

    def __init__(self, cfg):
        self.cfg = cfg
        name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # Create logs in robot-type/task-name hierarchy
        task_name = self.cfg["basic"]["task"]
        
        # Determine robot type from task name
        robot_type = self._get_robot_type(task_name)
        
        # Create hierarchical structure: logs/robot_type/task_name/timestamp
        robot_log_dir = os.path.join("logs", robot_type)
        task_log_dir = os.path.join(robot_log_dir, task_name)
        self.dir = os.path.join(task_log_dir, name)
        os.makedirs(self.dir, exist_ok=True)
        self.model_dir = os.path.join(self.dir, "nn")
        os.mkdir(self.model_dir)
        self.writer = SummaryWriter(os.path.join(self.dir, "summaries"))
        if self.cfg["runner"]["use_wandb"]:
            # Sanitize project name for wandb (remove invalid characters)
            project_name = self._sanitize_project_name(self.cfg["basic"]["task"])
            wandb.init(
                project=project_name,
                dir=self.dir,
                name=name,
                notes=self.cfg["basic"]["description"],
                config=self.cfg,
            )

        self.episode_statistics = {}
        self.last_episode = {}
        self.last_episode["steps"] = []
        self.episode_steps = None

        with open(os.path.join(self.dir, "config.yaml"), "w") as file:
            yaml.dump(self.cfg, file)

    def _sanitize_project_name(self, task_name):
        """Sanitize task name for wandb project name by removing invalid characters."""
        # Replace invalid characters with underscores
        invalid_chars = ['/', '\\', '#', '?', '%', ':']
        sanitized = task_name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        return sanitized

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

    def record_episode_statistics(self, done, ep_info, it, write_record=False):
        if self.episode_steps is None:
            self.episode_steps = torch.zeros_like(done, dtype=int)
        else:
            self.episode_steps += 1
        for val in self.episode_steps[done]:
            self.last_episode["steps"].append(val.item())
        self.episode_steps[done] = 0

        for key, value in ep_info.items():
            if self.episode_statistics.get(key) is None:
                self.episode_statistics[key] = torch.zeros_like(value)
            self.episode_statistics[key] += value
            if self.last_episode.get(key) is None:
                self.last_episode[key] = []
            for done_value in self.episode_statistics[key][done]:
                self.last_episode[key].append(done_value.item())
            self.episode_statistics[key][done] = 0

        if write_record:
            for key in self.last_episode.keys():
                path = ("" if key == "steps" or key == "reward" else "episode/") + key
                value = self._mean(self.last_episode[key])
                self.writer.add_scalar(path, value, it)
                if self.cfg["runner"]["use_wandb"]:
                    wandb.log({path: value}, step=it)
                self.last_episode[key].clear()

    def record_statistics(self, statistics, it):
        for key, value in statistics.items():
            self.writer.add_scalar(key, float(value), it)
            if self.cfg["runner"]["use_wandb"]:
                wandb.log({key: float(value)}, step=it)

    def save(self, model_dict, it):
        path = os.path.join(self.model_dir, "model_{}.pth".format(it))
        print("Saving model to {}".format(path))
        torch.save(model_dict, path)

    def _mean(self, data):
        if len(data) == 0:
            return 0.0
        else:
            return sum(data) / len(data)

    def log_video(self, frames, it, dt):
        """Log video frames to wandb.
        
        Args:
            frames: List of camera frames in RGBA format (height, width, 4)
            it: Current iteration step
            dt: Simulation timestep for calculating FPS
        """
        if not self.cfg["runner"]["use_wandb"]:
            return
        
        if len(frames) == 0:
            return
        
        import numpy as np
        
        # Convert frames from RGBA to RGB format
        # frames are stored as (height, width, 4) arrays
        video_array = []
        for frame in frames:
            # frame shape is (height, width, 4) for RGBA
            # Convert to RGB by taking first 3 channels
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                rgb_frame = frame[:, :, :3]
            else:
                rgb_frame = frame
            # Ensure uint8 format (0-255 range)
            if rgb_frame.dtype != np.uint8:
                rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
            video_array.append(rgb_frame)
        
        # Stack frames: (time, height, width, channels)
        video_tensor = np.stack(video_array, axis=0)
        
        # Calculate FPS from simulation timestep (same as in play mode)
        fps = int(1.0 / dt)
        
        # Log to wandb
        wandb.log({"video/training": wandb.Video(video_tensor, fps=fps, format="mp4")}, step=it)
