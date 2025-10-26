# Deploy on Booster Robot

This directory contains scripts and utilities for deploying trained policies on the Booster robot, including support for real-time parameter editing.

## Quickstart: Setup & Run Deployment Script

Follow these steps to set up your environment and deploy a policy on the robot:

1. **Copy the `deploy/` folder to your robot (Intel Board recommended):**
   ```sh
   $ scp -r deploy/ <username>@<robot_ip>:/<destination>/
   ```

2. **SSH into the robot and set up your environment:**
   ```sh
   $ ssh <username>@<robot_ip>
   $ cd /<destination>/deploy
   $ python3 -m venv venv
   $ source venv/bin/activate
   $ pip install -r requirements.txt
   ```
   - **Install the Booster Robotics SDK:**  
     Follow the [Booster Robotics SDK Guide](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f) and complete the [Compile Sample Programs and Install Python SDK](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-EI5fdtSucoJWO4xd49QcE5CInSf) section.


3. **Prepare the robot:**
   - Power on the robot.
   - Switch robot to **PREP Mode**.
   - Place the robot in a stable standing position in an open area.

4. **Deploy the policy:**
   - **For basic walking:**
     ```sh
     $ python deploy_base_walk.py --config=Base_Walk.yaml --net=127.0.0.1
     ```
   - **For parameterized walking (with real-time editing):**
     ```sh
     $ python deploy_parameter_walk.py --config=Parameter_Walk.yaml --net=127.0.0.1
     $ streamlit run streamlit_observation_editor.py
     ```
     - Open your browser at `http://<robot_ip>:8501` to access the web-based control interface.
     - Use interface sliders to adjust gait parameters and commands in real time.

5. **Exit Safely:**
   - Press `Ctrl+C` to stop deployment scripts.
   - Switch robot back to **PREP Mode** before turning off or moving the robot.

---

### Notes

- **Configuration files:**  
  All config files are in `configs/` (e.g. `Base_Walk.yaml`, `Parameter_Walk.yaml`). Each contains model paths, control gains, normalization, and limits.

- **Real-Time Observation Controls:**  
  The Streamlit interface lets you adjust gait frequency, foot yaw, body pitch/roll, feet offset, and walk commands on the fly (requires `deploy_parameter_walk.py`).

- **Network interface (`--net`):**  
  Use `127.0.0.1` if running on the Intel Board. Otherwise, specify the proper FastDDS/network address if deploying remotely or in simulation.

- **SDK & Policy Troubles:**  
  Ensure the Booster SDK is installed correctly and that model files exist in `models/`. For Streamlit issues, make sure `live_observation_values.json` is being created.

- **Robot Safety:**  
  Always enter/exit PREP Mode carefully and check surroundings before starting motion.

---
