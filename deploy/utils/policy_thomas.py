import numpy as np
import torch
from utils.observation_controller import get_controller


class Policy:
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"])
            self.policy.eval()
        except Exception as e:
            print(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()
        # Initialize observation controller for live control
        self.obs_controller = get_controller()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.gait_frequency = self.cfg["policy"]["gait_frequency"]
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw):
        self.gait_frequency = self.obs_controller.get_value(9)
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        
        # Use live-controlled walk commands from observation controller
        # Fallback to remote control service if not available
        try:
            self.commands[0] = self.obs_controller.get_vx_cmd()
            self.commands[1] = self.obs_controller.get_vy_cmd()
            self.commands[2] = self.obs_controller.get_vyaw_cmd()
        except:
            # Fallback to remote control service values
            self.commands[0] = vx
            self.commands[1] = vy
            self.commands[2] = vyaw
            
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        self.obs[0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.obs[3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.obs[6] = self.commands[0] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        self.obs[7] = self.commands[1] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        self.obs[8] = self.commands[2] * self.cfg["policy"]["normalization"]["ang_vel"] * (self.gait_frequency > 1.0e-8)
        
        # Use live-controlled values for obs[9:16]
        self.obs[9] = self.obs_controller.get_value(9)
        self.obs[10] = self.obs_controller.get_value(10)
        self.obs[11] = self.obs_controller.get_value(11)
        self.obs[12] = self.obs_controller.get_value(12)
        self.obs[13] = self.obs_controller.get_value(13)
        self.obs[14] = self.obs_controller.get_value(14)
        self.obs[15] = self.obs_controller.get_value(15)
        
        self.obs[16] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[17] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[18:30] = (dof_pos - self.default_dof_pos)[11:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.obs[30:42] = dof_vel[11:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[42:54] = self.actions

        self.actions[:] = self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets
