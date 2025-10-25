import os
import glob
import yaml
import argparse
import torch
from utils.model_thomas import *

def get_robot_type(task_name):
    """Determine robot type from task name."""
    # Check if task name starts with K1 or T1
    if task_name.startswith("K1"):
        return "K1"
    elif task_name.startswith("T1"):
        return "T1"
    else:
        # Default fallback - could be extended for other robot types
        return "Unknown"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        # Look for models in hierarchical structure: logs/robot_type/task_name/**/*.pth
        task_name = args.task
        robot_type = get_robot_type(task_name)
        
        # First try: exact task in robot-specific folder
        task_log_pattern = os.path.join("logs", robot_type, task_name, "**/*.pth")
        task_models = sorted(glob.glob(task_log_pattern, recursive=True), key=os.path.getmtime)
        
        if task_models:
            cfg["basic"]["checkpoint"] = task_models[-1]
        else:
            # Second try: any task in robot-specific folder
            robot_log_pattern = os.path.join("logs", robot_type, "**/*.pth")
            robot_models = sorted(glob.glob(robot_log_pattern, recursive=True), key=os.path.getmtime)
            
            if robot_models:
                cfg["basic"]["checkpoint"] = robot_models[-1]
            else:
                # Fallback: all logs if no robot-specific models found
                cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(model_dict["model"])

    model.eval()
    script_module = torch.jit.script(model.actor)
    save_path = os.path.splitext(cfg["basic"]["checkpoint"])[0] + ".pt"
    script_module.save(save_path)
    print(f"Saved model to {save_path}")
