import streamlit as st
import numpy as np
import yaml
import os
import time
from utils.policy_thomas import Policy
from utils.observation_controller import get_controller

# Page configuration
st.set_page_config(
    page_title="Policy Observation Editor",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Policy Observation Editor - Live Control")
st.markdown("Modify observation values 9-15 and walk commands (vx, vy, vyaw) for the robot policy in real-time")

# Initialize observation controller
obs_controller = get_controller()

# Sidebar for configuration
st.sidebar.header("Configuration")
config_file = st.sidebar.selectbox(
    "Select Configuration File",
    [f for f in os.listdir("configs") if f.endswith('.yaml') or f.endswith('.yml')],
    index=0 if os.path.exists("configs") else None
)

# Initialize policy if config is selected
policy = None
if config_file:
    try:
        with open(os.path.join("configs", config_file), "r", encoding="utf-8") as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        policy = Policy(cfg)
        st.sidebar.success(f"✅ Loaded config: {config_file}")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load config: {e}")
        policy = None

# Status indicator
st.sidebar.header("🔄 Live Status")
if os.path.exists("live_observation_values.json"):
    st.sidebar.success("✅ Live control active")
    st.sidebar.info("Values will be applied to running deployment script")
else:
    st.sidebar.warning("⚠️ No live control file found")

# Main content
if policy is not None:
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 Live Observation Values")
        
        # Define the observation parameters with their descriptions
        obs_params = {
            9: {"name": "Gait Frequency", "description": "Normalized gait frequency", "default": 1.7},
            10: {"name": "Foot Yaw (Left)", "description": "Normalized left foot yaw", "default": 0.0},
            11: {"name": "Foot Yaw (Right)", "description": "Normalized right foot yaw", "default": 0.0},
            12: {"name": "Body Pitch Target", "description": "Normalized body pitch target", "default": 0.1},
            13: {"name": "Body Roll Target", "description": "Normalized body roll target", "default": 0.0},
            14: {"name": "Feet Offset X", "description": "Normalized feet offset in X direction", "default": 0.0},
            15: {"name": "Feet Offset Y", "description": "Normalized feet offset in Y direction", "default": 0.0}
        }
        
        # Create sliders for each observation value
        obs_values = {}
        for idx, param in obs_params.items():
            st.subheader(f"obs[{idx}]: {param['name']}")
            st.caption(param['description'])
            
            # Get current value from controller
            current_value = obs_controller.get_value(idx)
            
            # Get normalization factor from config
            norm_key = None
            if idx == 9:
                norm_key = "gait_frequency"
            elif idx in [10, 11]:
                norm_key = "foot_yaw"
            elif idx == 12:
                norm_key = "body_pitch_target"
            elif idx == 13:
                norm_key = "body_roll_target"
            elif idx == 14:
                norm_key = "feet_offset_x_target"
            elif idx == 15:
                norm_key = "feet_offset_y_target"
            
            norm_factor = policy.cfg["policy"]["normalization"].get(norm_key, 1.0)
            
            # Create slider
            normalized_value = st.slider(
                f"Value (normalized)",
                min_value=-2.0,
                max_value=2.0,
                value=float(current_value),
                step=0.01,
                key=f"obs_{idx}"
            )
            
            # Update controller with new value
            if normalized_value != current_value:
                obs_controller.set_value(idx, normalized_value)
            
            # Show raw value
            raw_value = normalized_value / norm_factor if norm_factor != 0 else normalized_value
            st.metric("Raw Value", f"{raw_value:.4f}")
            
            obs_values[idx] = normalized_value
            
            st.divider()
        
        # Walk Command Controls
        st.header("🚶 Walk Commands")
        st.markdown("Control the robot's movement commands in real-time")
        
        # Define walk command parameters
        walk_params = {
            "vx": {"name": "Forward Velocity", "description": "Forward/backward movement", "default": 0.0, "min": -1.0, "max": 1.0},
            "vy": {"name": "Lateral Velocity", "description": "Left/right movement", "default": 0.0, "min": -1.0, "max": 1.0},
            "vyaw": {"name": "Yaw Velocity", "description": "Rotation left/right", "default": 0.0, "min": -3.0, "max": 3.0}
        }
        
        # Create sliders for walk commands
        walk_values = {}
        for cmd, param in walk_params.items():
            st.subheader(f"{cmd.upper()}: {param['name']}")
            st.caption(param['description'])
            
            # Get current value from controller
            if cmd == "vx":
                current_value = obs_controller.get_vx_cmd()
            elif cmd == "vy":
                current_value = obs_controller.get_vy_cmd()
            elif cmd == "vyaw":
                current_value = obs_controller.get_vyaw_cmd()
            
            # Create slider
            new_value = st.slider(
                f"Value",
                min_value=param["min"],
                max_value=param["max"],
                value=float(current_value),
                step=0.01,
                key=f"walk_{cmd}"
            )
            
            # Update controller with new value
            if new_value != current_value:
                if cmd == "vx":
                    obs_controller.set_vx_cmd(new_value)
                elif cmd == "vy":
                    obs_controller.set_vy_cmd(new_value)
                elif cmd == "vyaw":
                    obs_controller.set_vyaw_cmd(new_value)
            
            walk_values[cmd] = new_value
            st.divider()
    
    with col2:
        st.header("🎮 Live Control Panel")
        
        # Real-time status
        st.subheader("🔄 Real-time Status")
        
        # Auto-refresh every 2 seconds
        if st.button("🔄 Refresh Status") or 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        # Show current values from controller
        st.subheader("Current Values (Live)")
        
        # Observation values
        st.write("**Observation Values:**")
        current_values = obs_controller.get_all_values()
        for idx in range(9, 16):
            key = f"obs_{idx}"
            value = current_values.get(key, 0.0)
            st.metric(f"obs[{idx}]", f"{value:.4f}")
        
        # Walk commands
        st.write("**Walk Commands:**")
        vx = obs_controller.get_vx_cmd()
        vy = obs_controller.get_vy_cmd()
        vyaw = obs_controller.get_vyaw_cmd()
        st.metric("VX (Forward)", f"{vx:.4f}")
        st.metric("VY (Lateral)", f"{vy:.4f}")
        st.metric("VYaw (Rotation)", f"{vyaw:.4f}")
        
        # Control buttons
        st.subheader("⚙️ Quick Controls")
        
        col_reset, col_default = st.columns(2)
        
        with col_reset:
            if st.button("🔄 Reset to Defaults"):
                obs_controller.reset_to_defaults()
                st.success("Reset to defaults!")
                st.rerun()
        
        with col_default:
            if st.button("📊 Load Current Values"):
                st.rerun()
        
        # Export section
        st.header("💾 Export")
        
        # Create a dictionary with the current values
        export_data = {
            "observation_values": current_values,
            "walk_commands": {
                "vx": obs_controller.get_vx_cmd(),
                "vy": obs_controller.get_vy_cmd(),
                "vyaw": obs_controller.get_vyaw_cmd()
            },
            "config_file": config_file,
            "timestamp": str(np.datetime64('now'))
        }
        
        st.json(export_data)
        
        # Download button for the values
        import json
        st.download_button(
            label="📥 Download Current Values",
            data=json.dumps(export_data, indent=2),
            file_name=f"observation_values_{config_file.replace('.yaml', '')}.json",
            mime="application/json"
        )
        
        # Show the live control file content
        st.subheader("📄 Live Control File")
        if os.path.exists("live_observation_values.json"):
            with open("live_observation_values.json", 'r') as f:
                file_content = f.read()
            st.code(file_content, language="json")
        else:
            st.warning("Live control file not found")

else:
    st.warning("⚠️ Please select a configuration file in the sidebar to start editing observation values.")
    
    # Show available configs
    if os.path.exists("configs"):
        st.subheader("Available Configuration Files:")
        config_files = [f for f in os.listdir("configs") if f.endswith('.yaml') or f.endswith('.yml')]
        if config_files:
            for config in config_files:
                st.write(f"- {config}")
        else:
            st.write("No configuration files found in the configs directory.")
    else:
        st.write("Configs directory not found.")

# Footer
st.markdown("---")
st.markdown("**Note:** This app provides real-time control of observation values 9-15 and walk commands (vx, vy, vyaw). Changes are immediately applied to the running deployment script via the shared observation controller.")
st.markdown("**Usage:** 1. Start the deployment script (`python deploy_thomas.py`) 2. Use this app to control the observation values and walk commands in real-time") 