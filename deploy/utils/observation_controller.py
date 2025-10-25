import json
import os
import threading
import time
from typing import Dict, Any
import numpy as np


class ObservationController:
    """Manages observation values for real-time control between Streamlit and deployment script."""
    
    def __init__(self, config_file: str = "live_observation_values.json"):
        self.config_file = config_file
        self.lock = threading.Lock()
        self._file_mtime = 0  # Track file modification time
        self._default_values = {
            "obs_9": 1.7,   # gait_frequency
            "obs_10": 0.0,  # foot_yaw_left
            "obs_11": 0.0,  # foot_yaw_right
            "obs_12": 0.1,  # body_pitch_target
            "obs_13": 0.0,  # body_roll_target
            "obs_14": 0.0,  # feet_offset_x_target
            "obs_15": 0.0,  # feet_offset_y_target
            "vx": 0.0,      # forward velocity command
            "vy": 0.0,      # lateral velocity command
            "vyaw": 0.0,    # yaw velocity command
        }
        self._load_values()
    
    def _load_values(self):
        """Load current values from file or use defaults."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self._values = json.load(f)
                # Ensure all keys exist
                for key, default_val in self._default_values.items():
                    if key not in self._values:
                        self._values[key] = default_val
                # Update file modification time
                self._file_mtime = os.path.getmtime(self.config_file)
            else:
                self._values = self._default_values.copy()
                self._save_values()
        except Exception as e:
            print(f"Warning: Could not load observation values: {e}")
            self._values = self._default_values.copy()
    
    def _save_values(self):
        """Save current values to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._values, f, indent=2)
            # Update file modification time after saving
            self._file_mtime = os.path.getmtime(self.config_file)
        except Exception as e:
            print(f"Warning: Could not save observation values: {e}")
    
    def _check_and_reload_if_modified(self):
        """Check if the config file has been modified and reload if necessary."""
        try:
            if os.path.exists(self.config_file):
                current_mtime = os.path.getmtime(self.config_file)
                if current_mtime > self._file_mtime:
                    # File has been modified, reload it
                    with open(self.config_file, 'r') as f:
                        new_values = json.load(f)
                    # Ensure all keys exist
                    for key, default_val in self._default_values.items():
                        if key not in new_values:
                            new_values[key] = default_val
                    self._values = new_values
                    self._file_mtime = current_mtime
        except Exception as e:
            # If there's an error reading the file, just continue with current values
            pass
    
    def get_value(self, obs_index: int) -> float:
        """Get the current value for a specific observation index."""
        with self.lock:
            # Check if file has been modified and reload if necessary
            self._check_and_reload_if_modified()
            key = f"obs_{obs_index}"
            return self._values.get(key, self._default_values.get(key, 0.0))
    
    def set_value(self, obs_index: int, value: float):
        """Set the value for a specific observation index."""
        with self.lock:
            key = f"obs_{obs_index}"
            self._values[key] = value
            self._save_values()
    
    def get_all_values(self) -> Dict[str, float]:
        """Get all current observation values."""
        with self.lock:
            # Check if file has been modified and reload if necessary
            self._check_and_reload_if_modified()
            return self._values.copy()
    
    def set_all_values(self, values: Dict[str, float]):
        """Set all observation values at once."""
        with self.lock:
            self._values.update(values)
            self._save_values()
    
    def reset_to_defaults(self):
        """Reset all values to defaults."""
        with self.lock:
            self._values = self._default_values.copy()
            self._save_values()
    
    def get_values_array(self, num_observations: int = 54) -> np.ndarray:
        """Get observation values as a numpy array with the specified size."""
        obs_array = np.zeros(num_observations, dtype=np.float32)
        with self.lock:
            # Check if file has been modified and reload if necessary
            self._check_and_reload_if_modified()
            for i in range(9, 16):  # obs[9] to obs[15]
                key = f"obs_{i}"
                obs_array[i] = self._values.get(key, self._default_values.get(key, 0.0))
        return obs_array
    
    def get_vx_cmd(self) -> float:
        """Get forward velocity command."""
        with self.lock:
            self._check_and_reload_if_modified()
            return self._values.get("vx", self._default_values.get("vx", 0.0))
    
    def get_vy_cmd(self) -> float:
        """Get lateral velocity command."""
        with self.lock:
            self._check_and_reload_if_modified()
            return self._values.get("vy", self._default_values.get("vy", 0.0))
    
    def get_vyaw_cmd(self) -> float:
        """Get yaw velocity command."""
        with self.lock:
            self._check_and_reload_if_modified()
            return self._values.get("vyaw", self._default_values.get("vyaw", 0.0))
    
    def set_vx_cmd(self, value: float):
        """Set forward velocity command."""
        with self.lock:
            self._values["vx"] = value
            self._save_values()
    
    def set_vy_cmd(self, value: float):
        """Set lateral velocity command."""
        with self.lock:
            self._values["vy"] = value
            self._save_values()
    
    def set_vyaw_cmd(self, value: float):
        """Set yaw velocity command."""
        with self.lock:
            self._values["vyaw"] = value
            self._save_values()


# Global instance for easy access
_controller = None

def get_controller() -> ObservationController:
    """Get the global observation controller instance."""
    global _controller
    if _controller is None:
        _controller = ObservationController()
    return _controller 