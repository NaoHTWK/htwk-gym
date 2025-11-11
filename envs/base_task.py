import sys
from isaacgym import gymapi, gymutil

# Import torch after isaacgym modules are initialized
import torch
from utils.terrain import Terrain


class BaseTask:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.terrain = Terrain(self.gym, self.sim, self.device, self.cfg["terrain"])

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.set_viewer()

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        sim_cfg = self.cfg["sim"]
        sim_device = self.cfg["basic"]["sim_device"]
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(sim_device)

        # env device is GPU only if sim is on GPU, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda":
            self.device = sim_device
        else:
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        self.headless = self.cfg["basic"]["headless"]
        self.graphics_device_id = self.sim_device_id
        if self.headless and not self.cfg["viewer"]["record_video"]:
            self.graphics_device_id = -1

        self.sim_params = gymapi.SimParams()

        # assign general sim parameters
        self.sim_params.dt = sim_cfg["dt"]
        self.sim_params.num_client_threads = sim_cfg.get("num_client_threads", 0)
        self.sim_params.use_gpu_pipeline = sim_device_type == "cuda"
        self.sim_params.substeps = sim_cfg.get("substeps", 2)

        # assign up-axis
        if sim_cfg["up_axis"] == "z":
            self.up_axis_idx = 2
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
        elif sim_cfg["up_axis"] == "y":
            self.up_axis_idx = 1
            self.sim_params.up_axis = gymapi.UP_AXIS_Y
        else:
            raise ValueError(f"Invalid physics up-axis: {sim_cfg['up_axis']}")

        # assign gravity
        self.sim_params.gravity = gymapi.Vec3(*sim_cfg["gravity"])

        # configure physics parameters
        if sim_cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
            # set the parameters
            if "physx" in sim_cfg:
                for opt in sim_cfg["physx"].keys():
                    if opt == "contact_collection":
                        setattr(self.sim_params.physx, opt, gymapi.ContactCollection(sim_cfg["physx"][opt]))
                    else:
                        setattr(self.sim_params.physx, opt, sim_cfg["physx"][opt])
                setattr(self.sim_params.physx, "use_gpu", sim_device_type == "cuda")
        elif sim_cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
            # set the parameters
            if "flex" in sim_cfg:
                for opt in sim_cfg["flex"].keys():
                    setattr(self.sim_params.flex, opt, sim_cfg["flex"][opt])
        else:
            raise ValueError(f"Invalid physics engine backend: {sim_cfg['physics_engine']}")

        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

    def set_viewer(self):
        self.viewer = None
        self.camera = None
        if not self.headless:
            # if running with a viewer, set up keyboard shortcuts and camera
            self.enable_viewer_sync = True
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            position = self.cfg["viewer"]["pos"]
            lookat = self.cfg["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(position[0], position[1], position[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def render(self):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

        if self.cfg["viewer"]["record_video"]:
            if self.viewer is None:
                if self.device != "cpu":
                    self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
            if self.camera is None:
                camera_props = gymapi.CameraProperties()
                camera_props.width = 1280
                camera_props.height = 720
                camera_props.use_collision_geometry = False
                self.camera = self.gym.create_camera_sensor(self.envs[self.cfg["viewer"]["record_env_idx"]], camera_props)
                self.camera_frames = []
            cam_pos = gymapi.Vec3(
                *(x + y for x, y in zip(self.root_states[self.cfg["viewer"]["record_env_idx"], 0:3].tolist(), self.cfg["viewer"]["pos"]))
            )
            cam_target = gymapi.Vec3(*self.root_states[self.cfg["viewer"]["record_env_idx"], 0:3].tolist())
            self.gym.set_camera_location(self.camera, self.envs[self.cfg["viewer"]["record_env_idx"]], cam_pos, cam_target)
            self.gym.render_all_camera_sensors(self.sim)
            img = self.gym.get_camera_image(self.sim, self.envs[self.cfg["viewer"]["record_env_idx"]], self.camera, gymapi.IMAGE_COLOR)
            self.camera_frames.append(img.reshape(img.shape[0], -1, 4))
