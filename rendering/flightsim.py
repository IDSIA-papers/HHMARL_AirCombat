"""FlightSim (https://github.com/ardian-selmonaj/FlightSim), a Skybolt-based flight simulator, as realistic
renderer of AircraftEnv and, optionally, as JSBSim simulator of its aircraft.

FlightSim is a separate repository. Clone and set it up (setup_windows.bat or setup_linux.sh) next to this
repository, or set the environment variable FLIGHTSIM_DIR to its folder. It is only imported when used.
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from env.utils import get_root_dir

# Minimum FLIGHTSIM_API_VERSION of the FlightSim build
FLIGHTSIM_API_VERSION = 10

# Colors of the labels and trails per team (linear RGB)
TEAM_COLORS = {"B": (0.05, 0.3, 1.0), "R": (1.0, 0.06, 0.04)}

FEET_TO_METERS = 0.3048


class FlightSimClosed(Exception):
    """Raised when the FlightSim window was closed or escape was pressed."""


def import_skybolt():
    """Imports the skybolt module of FlightSim, see the module docstring."""
    flightsim_dir = Path(os.environ.get("FLIGHTSIM_DIR", Path(get_root_dir()).parent / "FlightSim")).resolve()
    if not (flightsim_dir / "flightsim_env.py").is_file():
        raise ImportError(f"FlightSim not found in '{flightsim_dir}'. Clone and set up FlightSim there, "
                          "or set the environment variable FLIGHTSIM_DIR to its folder.")
    if str(flightsim_dir) not in sys.path:
        sys.path.append(str(flightsim_dir))
    import flightsim_env  # noqa: F401  sets up the paths of the FlightSim build for skybolt
    import skybolt

    version = getattr(skybolt, "FLIGHTSIM_API_VERSION", 0)
    if version < FLIGHTSIM_API_VERSION:
        raise ImportError(f"The FlightSim build in '{flightsim_env.SKYBOLT_DIR}' is too old (API version {version}, "
                          f"needed {FLIGHTSIM_API_VERSION}). Update FlightSim and build it again, see its README.")
    return skybolt


class FlightSimBridge:
    """Shows the aircraft of an AircraftEnv in FlightSim, created by AircraftEnv for config["flightsim"].

    Args:
        env: the AircraftEnv.
        jsbsim_states: "python" if the JSBSim of the env simulates the aircraft and FlightSim only shows them,
            "c++" if the JSBSim of FlightSim simulates them (see FlightSimFDM). Both use JSBSim 1.3.0 at sim_freq.
        manual_agent: ID of the aircraft flown by a human with the joysticks (FlightSim's Tools/joysticks.json) or the keyboard
            (mission model_human), or None. Its F-16 is the one of FlightSim's Assets/Movers/JSBSimManual, whose rudder is
            corrected, also with jsbsim_states="c++" (MANUAL_TEMPLATES). The RL agents keep the F-16 of their training.
        window: (x, y, width, height) of the FlightSim window.
        speed: simulated seconds per real second, or None to run as fast as possible.
        camera_view: "Cockpit", "Follow", "Overview" (all aircraft) or "Free", or None for "Cockpit" with manual_agent,
            otherwise "Overview".
        trail_length: length in m of the flown paths shown behind the aircraft (key T toggles them, L the labels).
        templates: FlightSim entity templates per team, e.g. {"R": "SepecatJaguarRedBig"}, replacing the defaults
            of TEMPLATES. With jsbsim_states="c++", the templates need a 'jsbSimAircraft' component.
        tracer_interval: JSBSim steps between the tracers (short dark lines) shown while the cannon of an aircraft fires.
        headless: run FlightSim without a window, e.g. to compare jsbsim_states "c++" and "python".
        recording_dir: folder of the videos recorded with key R (MP4 with 30 frames per second of simulated time, showing the view
            of the window without the help message and the red dot shown while recording), by default "recordings" of this repository.
        minimap: show the map of the aircraft around the camera target in the bottom right corner of the window (key M toggles it):
            arrows in the team colors pointing in their headings, with their IDs and distances (3D) to the camera target, and dots along the paths of the other aircraft in the last 30 s.
        minimap_range: distance in m from the center of the minimap to its outer ring. Aircraft farther away are shown on the ring
            with their distance.
        minimap_heading_up: True: the heading of the camera target points up on the minimap, False: north (key N toggles it),
            None: heading-up with manual_agent, otherwise north-up.
        wez: indicate the weapon engagement zones (WEZ) of the cannons (CannonSim) in the cockpit and follow views (F1, F2, key Z
            toggles it): a light green circle ahead of the camera target while an opponent is in its WEZ, and a red beam from the
            direction of each opponent that has the camera target in its WEZ, fading out towards it (a red border around the view
            for an opponent behind the camera).
    """

    TEMPLATES = {
        "python": {"B": "SepecatJaguarBlue", "R": "SepecatJaguarRed"},
        "c++": {"B": "SepecatJaguarJsbBlue", "R": "SepecatJaguarJsbRed"},
    }
    # JSBSim models of the "c++" templates
    CPP_MODELS = ("f16",)
    # "c++" templates of the manual agent, with the F-16 of FlightSim's Assets/Movers/JSBSimManual
    MANUAL_TEMPLATES = {"B": "SepecatJaguarJsbBlueManual", "R": "SepecatJaguarJsbRedManual"}

    def __init__(self, env, jsbsim_states="python", manual_agent=None, window=(10, 30, 1600, 900), speed=1.0,
                 camera_view=None, trail_length=30000.0, templates=None, tracer_interval=5, headless=False, recording_dir=None,
                 minimap=True, minimap_range=20000.0, minimap_heading_up=None, wez=True):
        if jsbsim_states not in self.TEMPLATES:
            raise ValueError(f"jsbsim_states must be 'python' or 'c++', not '{jsbsim_states}'")
        if manual_agent is not None and manual_agent not in env.agents:
            raise ValueError(f"Manual agent {manual_agent} is not an aircraft of the env: {list(env.agents)}")
        if headless and manual_agent is not None:
            raise ValueError("A manual agent needs the FlightSim window, so it cannot be used with headless")

        self.sb = import_skybolt()
        self.jsbsim_states = jsbsim_states
        self.manual_agent = manual_agent
        self.speed = speed
        self.tracer_interval = tracer_interval
        self.templates = {**self.TEMPLATES[jsbsim_states], **(templates or {})}
        self.ids = {aid: i + 1 for i, aid in enumerate(env.agents)}  # FlightSim aircraft IDs
        self.center = (env.center_lon, env.center_lat)
        self.dt = 1.0 / env.sim_freq

        x, y, width, height = window
        self.app = self.sb.FlightSim(x, y, width, height, manual=manual_agent is not None, multiwindow=False,
                                     dynamicsRate=env.sim_freq, headless=headless)
        self.app.select_camera_view(camera_view or ("Cockpit" if manual_agent is not None else "Overview"))
        self.app.trail_length = trail_length
        self.app.minimap_visible = minimap
        self.app.minimap_range = minimap_range
        self.app.minimap_heading_up = manual_agent is not None if minimap_heading_up is None else minimap_heading_up
        self.app.wez_visible = wez
        self.app.recording_directory = recording_dir or os.path.join(get_root_dir(), "recordings")
        for aid, fid in self.ids.items():
            self.app.set_aircraft_label(fid, aid)
            self.app.set_aircraft_color(fid, TEAM_COLORS.get(aid[0], (1.0, 1.0, 1.0)))
            self.app.set_aircraft_team(fid, env.agents[aid].tid)  # aircraft of other teams are opponents, as in the env

        if jsbsim_states == "c++":
            for agent in env.agents.values():
                if agent.model not in self.CPP_MODELS:
                    raise ValueError(f"{agent.aid} uses the JSBSim model '{agent.model}', but the FlightSim templates "
                                     f"simulate {self.CPP_MODELS}. Use jsbsim_states='python'.")
                agent.fdm_factory = self.create_fdm
        elif manual_agent is not None:
            import flightsim_env
            human = env.agents[manual_agent]
            if (flightsim_env.MANUAL_AIRCRAFT_DIR / human.model).is_dir():
                human.aircraft_path = str(flightsim_env.MANUAL_AIRCRAFT_DIR)

        self._hidden = set()  # IDs of the env aircraft that are hidden because they died
        self._substeps = 0
        self._sim_time = 0.0
        self._wall_start = time.perf_counter()

    def create_fdm(self, agent):
        """AircraftSim.fdm_factory for jsbsim_states="c++"."""
        return FlightSimFDM(self, agent)

    def fdm_template(self, agent):
        """@returns the template of the FlightSim aircraft that simulates the agent with jsbsim_states="c++"."""
        return self.MANUAL_TEMPLATES[agent.tid] if agent.aid == self.manual_agent else self.templates[agent.tid]

    def manual_controls(self):
        """@returns the controls of the manual aircraft from the joysticks or the keyboard: aileron, elevator, rudder, throttle."""
        return np.asarray(self.app.get_manual_controls(), dtype=np.float64)

    def consume_fire(self):
        """@returns True if the fire button or key is held, or was pressed since the last call."""
        return self.app.consume_fire_requests() > 0 or self.app.fire_pressed

    def on_reset(self, env):
        """Shows the aircraft at their initial states, called at the end of AircraftEnv.reset()."""
        self.app.clear_trails()
        self.app.clear_tracers()
        # The WEZ indications show the zone of the cannons, which AircraftSim.reload() has created
        cannon = next(iter(env.agents.values())).cannon
        self.app.wez_range = cannon.Rmax_wez
        self.app.wez_angle = cannon.omega_wez
        for aid, agent in env.agents.items():
            fid = self.ids[aid]
            if self.jsbsim_states == "python":
                if fid in self.app.get_aircraft_ids():
                    self.app.set_aircraft_state(fid, *self._state(agent))
                    self.app.set_aircraft_visible(fid, True)
                else:
                    self.app.add_aircraft(fid, self.templates[agent.tid], *self._state(agent))
                self._show_throttle(fid, agent)
            # With "c++", FlightSimFDM has created the aircraft in AircraftSim.reload()
        self._hidden.clear()

        if self.manual_agent is not None:
            self.app.set_manual_aircraft(self.ids[self.manual_agent])
        else:
            camera_agent = env.rl_agent_ids[0] if env.rl_agent_ids else next(iter(env.agents))
            self.app.set_camera_target(self.ids[camera_agent])

        self.app.step(0)  # updates the visuals without advancing the simulation
        self._render()
        self._sim_time = 0.0
        self._wall_start = time.perf_counter()

    def before_substep(self, env):
        """Called by AircraftEnv.run_substep() before the aircraft are advanced by one JSBSim step."""
        # Aircraft that were shot down or crashed disappear and are no longer simulated
        for aid, agent in env.agents.items():
            if not agent.alive and aid not in self._hidden:
                self._hidden.add(aid)
                self.app.set_aircraft_visible(self.ids[aid], False)
                if self.jsbsim_states == "c++":
                    self.app.set_dynamics_enabled(self.ids[aid], False)

        if self.manual_agent is not None:
            human = env.agents[self.manual_agent]
            if human.alive:
                human.set_property_values(human.mission.action_var, self.manual_controls())
            human.mission.fire_cannon(human, env, self.consume_fire())

        if self.jsbsim_states == "c++":
            # Advances all JSBSim aircraft of FlightSim, FlightSimFDM.run() does nothing
            self.app.step(1)

    def after_substep(self, env):
        """Called by AircraftEnv.run_substep() after the aircraft were advanced by one JSBSim step."""
        if self.jsbsim_states == "python":
            for aid, agent in env.agents.items():
                if agent.alive:
                    self.app.set_aircraft_state(self.ids[aid], *self._state(agent))
                    self._show_throttle(self.ids[aid], agent)
            self.app.step(1)

        self._substeps += 1
        if self._substeps % self.tracer_interval == 0:
            for aid, agent in env.agents.items():
                if agent.alive and agent.cannon.alive:
                    self.app.fire_tracer(self.ids[aid])

        self._sim_time += self.dt
        self._present()

    def close(self):
        if self.app is not None:
            self.app.stop_recording()  # finishes the video of key R
        self.app = None

    def _state(self, agent):
        """@returns position, roll/pitch/yaw and north/east/down velocity of a Python JSBSim aircraft for FlightSim,
        as the FlightSim JSBSim plugin computes them."""
        get = agent.ac_sim.get_property_value
        position = self.sb.LatLonAlt(get("position/lat-geod-rad"), get("position/long-gc-rad"), get("position/h-sl-meters"))
        rpy = self.sb.Vector3(get("attitude/roll-rad"), get("attitude/pitch-rad"), get("attitude/heading-true-rad"))
        velocity = self.sb.Vector3(get("velocities/v-north-fps") * FEET_TO_METERS,
                                   get("velocities/v-east-fps") * FEET_TO_METERS,
                                   get("velocities/v-down-fps") * FEET_TO_METERS)
        return position, rpy, velocity

    def _show_throttle(self, fid, agent):
        """Passes the throttle lever of a Python JSBSim aircraft to FlightSim, whose nozzles show it while it is the camera target.
        The JSBSim aircraft of FlightSim (jsbsim_states="c++") show their own."""
        self.app.set_aircraft_throttle(fid, agent.ac_sim.get_property_value("fcs/throttle-cmd-norm"))

    def _present(self):
        """Renders a frame, keeping the simulation in real time (or at speed), if not behind schedule.
        While recording, the frames of the video are always rendered, so that the video does not repeat frames."""
        if self.speed is None:
            self._render()
            return
        now = time.perf_counter()
        target = self._wall_start + self._sim_time / self.speed
        render = now <= target
        if now - target > 0.25:
            # Far behind, e.g. after the window was moved: continue from now instead of catching up.
            # While recording, the frames of the video are rendered anyway.
            self._wall_start += now - target
            target = now
            render = not self.app.recording
        if render or self.app.recording_frame_due:
            self._render()
            now = time.perf_counter()
            if now < target:
                time.sleep(target - now)
        # Otherwise the frame is skipped to catch up

    def _render(self):
        if not self.app.render():
            raise FlightSimClosed("FlightSim window closed")


class FlightSimFDM:
    """Replaces the jsbsim.FGFDMExec of an AircraftSim with the JSBSim of a FlightSim aircraft, for jsbsim_states="c++".

    Like AircraftSim.reload() creates a new FGFDMExec, each instance creates a new FlightSim aircraft, which replaces
    the previous one of the agent. FlightSimBridge advances all FlightSim aircraft together, so run() does nothing.
    """

    def __init__(self, bridge, agent):
        self.app = bridge.app
        self.id = bridge.ids[agent.aid]
        lon, lat = bridge.center
        # The initial conditions are set by run_ic(), this is just the position until then
        position = bridge.sb.LatLonAlt(math.radians(lat), math.radians(lon), 5000.0)
        self.app.add_aircraft(self.id, bridge.fdm_template(agent), position)
        # The flight controls are set by the env (and for the manual agent from the joysticks or the keyboard by FlightSimBridge)
        self.app.set_external_control(self.id, True)

    def set_debug_level(self, level):
        pass

    def load_model(self, model):
        if model not in FlightSimBridge.CPP_MODELS:
            raise ValueError(f"FlightSim simulates the JSBSim models {FlightSimBridge.CPP_MODELS}, not '{model}'")
        return True

    def set_dt(self, dt):
        if not math.isclose(dt, self.app.dynamics_step_size, rel_tol=1e-9):
            raise ValueError(f"JSBSim step {dt} s differs from the FlightSim step {self.app.dynamics_step_size} s")

    def query_property_catalog(self, check):
        return self.app.query_property_catalog(self.id, check)

    def get_property_value(self, name):
        return self.app.get_property(self.id, name)

    def set_property_value(self, name, value):
        self.app.set_property(self.id, name, float(value))

    def run_ic(self):
        return self.app.run_ic(self.id)

    def run(self):
        return True

    def get_sim_time(self):
        return self.app.get_property(self.id, "simulation/sim-time-sec")

    def get_propulsion(self):
        return _FlightSimPropulsion(self.app, self.id)


class _FlightSimPropulsion:
    """The part of jsbsim.FGPropulsion used by AircraftSim."""

    def __init__(self, app, aircraft_id):
        self.app = app
        self.id = aircraft_id

    def get_num_engines(self):
        return self.app.get_num_engines(self.id)

    def get_engine(self, index):
        return _FlightSimEngine(self.app, self.id, index)

    def get_steady_state(self):
        return self.app.get_steady_state(self.id)


class _FlightSimEngine:
    """The part of jsbsim.FGEngine used by AircraftSim."""

    def __init__(self, app, aircraft_id, index):
        self.app = app
        self.id = aircraft_id
        self.index = index

    def init_running(self):
        self.app.init_engine_running(self.id, self.index)
