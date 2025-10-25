#!/usr/bin/env python3
# export2tflite.py
#
# Usage:
#   python export2tflite.py --task mytask
#   python export2tflite.py --task mytask --checkpoint logs/run42/model_1200.pth --quant int8
#
# Requirements (pip):
#   torch onnx onnx-tf tensorflow==2.16.1 pyyaml

import os, glob, yaml, argparse, shutil, tempfile, pathlib
import numpy as np
import torch, onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from utils.model_thomas import ActorCritic          # <-- changed to use Thomas model with odometry

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

# --------------------------------------------------------------------------- #
def export_pytorch_to_onnx(model, onnx_path, input_shape):
    dummy = torch.randn(*input_shape, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['obs'], output_names=['action'],
        dynamic_axes={'obs': {0: 'N'}, 'action': {0: 'N'}},
        opset_version=18
    )
    print(f"[✓] ONNX written to {onnx_path}")

def export_onnx_to_savedmodel(onnx_path, saved_dir):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_dir)
    print(f"[✓] SavedModel at {saved_dir}")

def export_savedmodel_to_tflite(saved_dir, tflite_path, quant, rep_shape):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)

    if quant is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quant == "int8":
        def rep_dataset():
            for _ in range(128):
                yield [np.random.rand(*rep_shape).astype(np.float32)]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    pathlib.Path(tflite_path).write_bytes(tflite_model)
    print(f"[✓] TFLite model saved to {tflite_path}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Name of the task YAML.")
    parser.add_argument("--checkpoint", help="Override path to .pth checkpoint.")
    parser.add_argument("--quant", choices=[None, "dr", "int8"], default=None,
                        help="Quantisation: dr = dynamic-range, int8 = full int8.")
    parser.add_argument("--obs-shape", nargs="+", type=int, default=None,
                        help="Override observation tensor shape, e.g. --obs-shape 1 84 84")
    args = parser.parse_args()

    # ---------- 1.  Read task config ------------------------------------------------
    cfg_path = os.path.join("envs", f"{args.task}.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.checkpoint:
        cfg["basic"]["checkpoint"] = args.checkpoint
    if not cfg["basic"]["checkpoint"] or str(cfg["basic"]["checkpoint"]) in {"-1", "-1"}:
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
                cfg["basic"]["checkpoint"] = sorted(
                    glob.glob(os.path.join("logs", "**/*.pth"), recursive=True),
                    key=os.path.getmtime
                )[-1]
    ckpt = cfg["basic"]["checkpoint"]
    print(f"[*] Loading checkpoint {ckpt}")

    # ---------- 2.  Restore PyTorch model ------------------------------------------
    model = ActorCritic(
        cfg["env"]["num_actions"],
        cfg["env"]["num_observations"],
        cfg["env"]["num_privileged_obs"]
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    actor = model.actor          # the subnet you want to deploy

    # ---------- 3.  Decide I/O tensor shapes ---------------------------------------
    obs_shape = args.obs_shape or [1, cfg["env"]["num_observations"]]
    # e.g. for images you might need [1, 3, 84, 84]
    rep_shape = obs_shape        # for quant representative set

    stem = os.path.splitext(ckpt)[0]
    onnx_path   = stem + ".onnx"
    saved_dir   = stem + "_saved"
    tflite_path = stem + ".tflite"

    # ---------- 4.  Export chain ----------------------------------------------------
    export_pytorch_to_onnx(actor, onnx_path, obs_shape)
    export_onnx_to_savedmodel(onnx_path, saved_dir)
    export_savedmodel_to_tflite(saved_dir, tflite_path, args.quant, rep_shape)

    # ---------- 5.  Quick smoke-test -----------------------------------------------
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    print(f"[✓] TFLite input  : {interpreter.get_input_details()[0]['shape']}")
    print(f"[✓] TFLite output : {interpreter.get_output_details()[0]['shape']}")

    # ---------- 6.  Verify outputs match -------------------------------------------
    # Generate test input
    test_input = torch.randn(*obs_shape, dtype=torch.float32)
    
    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = actor(test_input).numpy()
    
    # Get TFLite output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], test_input.numpy())
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - tflite_output))
    print(f"[*] Maximum difference between PyTorch and TFLite outputs: {max_diff:.6f}")
    if max_diff < 1e-5:
        print("[✓] Outputs match within acceptable tolerance")
    else:
        print("[!] Warning: Significant difference between outputs")
