import logging
from abc import ABC, abstractmethod
import numpy as np
import importlib.util
import random
import os
import jsbsim
from env.catalog import Catalog, Property
from env.utils import LLA2NEU, get_AO_TA_R

#rgb colors used for rendering
COLORS = {"B": [0,0,1,1], "R": [1,0,0,1], "G": [0,1,0,1], "Y": [1,1,0,1]}
JSBSIM_DIR = os.path.dirname(importlib.util.find_spec('jsbsim').origin)
os.environ["JSBSIM_DEBUG"]=str(0)

class BaseSim(ABC):
    """
    Base Instance for simulation. 
    tid = team ID, team color
    aid = agent ID, e.g. B1 for agent 1 in team blue
    dt = simulation frequency
    """
    def __init__(self, tid: str, aid:str, dt: float = 1/60) -> None:
        self.log = logging.getLogger(__name__)
        self.__tid = tid
        self.__aid = aid
        self.__dt = dt
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)
        self.model = ""

    @property
    def tid(self) -> str:
        return self.__tid
    
    @property
    def aid(self) -> str:
        return self.__aid
    
    @property
    def color_code(self) -> str:
        return COLORS[self.__tid]
    
    @property
    def color_name(self) -> str:
        return self.__tid

    @property
    def dt(self) -> float:
        return self.__dt
    
    def get_geodetic(self):
        """(longitude, latitude, altitude), unit: °, m"""
        return self._geodetic

    def get_position(self):
        """(north, east, up), unit: m"""
        return self._position

    def get_rpy(self, degree=False):
        """(roll, pitch, yaw), unit: rad"""
        return self._posture * 180 / np.pi if degree else self._posture

    def get_velocity(self):
        """(v_north, v_east, v_up), unit: m/s"""
        return self._velocity
    
    def get_norm_velocity_2d(self):
        """norm (v_north, v_east), unit: m/s"""
        return np.linalg.norm(self._velocity[:2])
    
    def get_distance_to_origin_3d(self):
        """distance in m to origin (lat0, lon0, alt0)"""
        return np.linalg.norm(self.get_position())
    
    def get_distance_to_origin_2d(self):
        """distance in m to origin (lat0, lon0)"""
        return np.linalg.norm(self.get_position()[:2])
    
    def get_distance_to_origin_lat(self):
        """distance in m to origin lat0"""
        return np.linalg.norm(self.get_position()[0])
    
    def get_distance_to_origin_lon(self):
        """distance in m to origin lon0"""
        return np.linalg.norm(self.get_position()[1])

    def get_altitude(self):
        """height in m above sea level."""
        # return np.linalg.norm(self.get_position()[-1])
        return self.get_property_value(Catalog.position_h_sl_m)
    
    def get_distance_to_target(self, target):
        return np.linalg.norm(target.get_position() - self.get_position())

    def reload(self):
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)

    @abstractmethod
    def run(self, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass

    def log(self):
        lon, lat, alt = self.get_geodetic()
        roll, pitch, yaw = self.get_rpy() * 180 / np.pi
        log_msg = f"{self.aid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
        log_msg += f"Name={self.model.upper()},"
        log_msg += f"Team={self.color_name}"
        return log_msg

class AircraftSim(BaseSim):
    """
    Aircraft simulation through JSBSim.
    """

    ALIVE = 0
    CRASH = 1           # extreme state / overload
    SHOTDOWN = 2        # hit by missile / cannon

    def __init__(self, 
                 tid: str, 
                 aid: str,
                 mission: object,
                 learning: bool,
                 model: str = 'f16',
                 num_missiles: int = 10,
                 init_state: dict = {},
                 origin: tuple = (0.0, 0.0, 0.0),
                 sim_freq: int = 100,
                 sim_steps:int = 10,
                 unlimited_fuel: bool = True,
                 **kwargs
                 ) -> None:
        """
        Creates an instance of JSBSim, loads an aircraft and sets initial conditions
        Args:
            - tid (str): team color
            - aid (str): agent id
            - mission(obj): Task or pre-trained Model
            - learning (bool): is a learning RL agent or pre-trained.
            - model (str): aircraft model to be loaded. Default = 'f16'
            - num_missiles (int): number of missiles the aircraft can fire. Default = 10
            - init_state (dict): initial properties
            - origin (tuple): combat field origin (longitude, latitude, altitude). Default = `(0.0, 0.0, 0.0)
            - sim_freq (int): JSBSim integration frequency. Default = 60.
        """

        super().__init__(tid, aid, 1/sim_freq)
        self.sim_steps = sim_steps
        self.model = model
        self.mission = mission
        self.is_learning = learning
        self.init_state = init_state
        self.lon0, self.lat0, self.alt0 = origin
        self.unlimited_fuel = unlimited_fuel

        self.partners = [] #AirCraftSim
        self.opponents = [] #AirCraftSim
        self.cannon = None
        self.missiles_max = num_missiles
        self.ac_sim = None  # JSBSim instance, created by reload()
        # Creates the JSBSim instance on reload(): None for jsbsim.FGFDMExec, otherwise a function
        # fdm_factory(aircraft_sim) returning an object with the same methods, e.g. rendering.flightsim.FlightSimFDM
        self.fdm_factory = None
        # Folder of the JSBSim model of jsbsim.FGFDMExec, None for the aircraft of the jsbsim package. FlightSimBridge sets the
        # one of FlightSim for the manual agent, whose F-16 has a corrected rudder.
        self.aircraft_path = None

        #self.reload()

    @property
    def alive(self):
        return self._status == AircraftSim.ALIVE

    @property
    def crashed(self):
        return self._status == AircraftSim.CRASH

    @property
    def shotdown(self):
        return self._status == AircraftSim.SHOTDOWN

    def crash(self):
        self._status = AircraftSim.CRASH

    def shootdown(self):
        self._status = AircraftSim.SHOTDOWN

    def get_status(self):
        return self._status

    def reload(self, new_state = None, new_origin = None, mission = None):
        """
        Reload aircraft.
        """
        super().reload()
        self.cannon = CannonSim(self.dt, self.sim_steps, self.missiles_max)
        self._status = AircraftSim.ALIVE
        if mission is not None: self.mission = mission

        # load and reset properties
        self.ac_sim = jsbsim.FGFDMExec(JSBSIM_DIR) if self.fdm_factory is None else self.fdm_factory(self)
        self.ac_sim.set_debug_level(0)
        if self.aircraft_path is not None:
            self.ac_sim.set_aircraft_path(self.aircraft_path)
        self.ac_sim.load_model(self.model)
        self.ac_sim.set_dt(self.dt)
        Catalog.add_jsbsim_props(self.ac_sim.query_property_catalog(""))
        self.reset_properties()
        if self.unlimited_fuel:
            self.refill()

        # assign new properties
        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin
        for key, value in self.init_state.items():
            self.set_property_value(Catalog[key], value)
        success = self.ac_sim.run_ic()
        if not success:
            raise RuntimeError("JSBSim failed to init simulation conditions.")
        
        # propulsion init running
        propulsion = self.ac_sim.get_propulsion()
        n = propulsion.get_num_engines()
        for i in range(n):
            propulsion.get_engine(i).init_running()
        propulsion.get_steady_state()
        # update inner property
        self.update_properties()

    def run(self):
        """
        Runs JSBSim simulation.
        """
        if self.alive:
            result = self.ac_sim.run()
            if not result:
                raise RuntimeError("JSBSim failed.")
            self.update_properties()
            if self.unlimited_fuel:
                self.refill()
        if self.cannon.alive:
            self.cannon.run()

    def close(self):
        """
        Closes simulation and instances.
        """
        if self.ac_sim:
            self.ac_sim = None
        self.partners.clear()
        self.opponents.clear()
        self.mission = None

    def refill(self):
        """
        Refill tank to simulate unlimited fuel (freezed).
        """
        if self.model == 'f16':
            content = 1500
        elif self.model == 'f15':
            content = 2000
        elif self.model == 'f22':
            content = 6500
        elif self.model == 'A4':
            content = 1500
        else:
            content = 1500
        self.set_property_value(Catalog.propulsion_tank_contents_lbs, content)

    def reset_properties(self):
        """
        Reset initial conditions
        """
        default_props = {
            Catalog.ic_long_gc_deg: 0.0,    # geodesic longitude [deg]
            Catalog.ic_lat_geod_deg: 0.0,   # geodesic latitude  [deg]
            Catalog.ic_h_sl_ft: 20000,      # altitude above mean sea level [ft]
            Catalog.ic_psi_true_deg: 0.0,   # initial (true) heading [deg] (0, 360)
            Catalog.ic_u_fps: 800.0,        # body frame x-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_v_fps: 0.0,          # body frame y-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_w_fps: 0.0,          # body frame z-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_p_rad_sec: 0.0,      # roll rate  [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_q_rad_sec: 0.0,      # pitch rate [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_r_rad_sec: 0.0,      # yaw rate   [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_roc_fpm: 0.0,        # initial rate of climb [ft/min]
            Catalog.ic_terrain_elevation_ft: 0,
            Catalog.gear_gear_cmd_norm: 0.0 # gear up
        }
        for prop, value in default_props.items():
            self.set_property_value(prop, value)

    def update_properties(self):
        # update position
        self._geodetic[:] = self.get_property_values([
            Catalog.position_long_gc_deg,
            Catalog.position_lat_geod_deg,
            Catalog.position_h_sl_m
        ])
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
        # update posture
        self._posture[:] = self.get_property_values([
            Catalog.attitude_roll_rad,
            Catalog.attitude_pitch_rad,
            Catalog.attitude_heading_true_rad,
        ])
        # update velocity
        self._velocity[:] = self.get_property_values([
            Catalog.velocities_v_north_mps,
            Catalog.velocities_v_east_mps,
            Catalog.velocities_v_up_mps,
        ])

    def set_property_values(self, props, values):
        """Set the values of the specified properties
        """
        if not len(props) == len(values):
            raise ValueError("mismatch between properties and values size")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def set_property_value(self, prop, value):
        """Set the values of the specified property
        """
        # set value in property bounds
        if isinstance(prop, Property):
            if value < prop.min:
                value = prop.min
            elif value > prop.max:
                value = prop.max

            self.ac_sim.set_property_value(prop.name_jsbsim, value)

            if "W" in prop.access:
                if prop.update:
                    prop.update(self)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def get_property_values(self, props):
        """Get the values of the specified properties
        """
        return [self.get_property_value(prop) for prop in props]

    def get_property_value(self, prop):
        """Get the value of the specified property from the JSBSim simulation
        """
        if isinstance(prop, Property):
            if prop.access == "R":
                if prop.update:
                    prop.update(self)
            return self.ac_sim.get_property_value(prop.name_jsbsim)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def get_sim_time(self):
        """
        Get JSBSim simulation time
        """
        return self.ac_sim.get_sim_time()

class CannonSim(object):
    """
    Instance to simulate Weapon Engagement Zone (WEZ). If opponent is inside the WEZ cone, 
    there is a 80% probability that it gets destroyed.
    """
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2
    HIT_PROBABILITY = 0.8   # per JSBSim step while an opponent is inside the WEZ cone

    def __init__(self, dt:float, sim_steps:int, max_missiles:int):
        self._status = CannonSim.INACTIVE
        self.dt = dt
        self.sim_steps = sim_steps
        self.parent_aircraft = None

        self.cannon_shots = self.max_shots = max_missiles
        self.omega_wez = 10     # (half) cone angle in °
        self.Rmax_wez = 3500   # m
        self._Rhit_prob = CannonSim.HIT_PROBABILITY   # hit probability
        self.t_max = 90 * self.sim_steps*self.dt
        self.shot_step = 0
        self._held = False  # fires while held, see fire()
        self.target = None  # opponent shot down by the last burst (status HIT)

    @property
    def ready(self):
        """cannon ready to be launched"""
        return self._status == CannonSim.INACTIVE

    @property
    def alive(self):
        """cannon still firing"""
        return self._status == CannonSim.LAUNCHED

    @property
    def success(self):
        """cannon shots have hit the target"""
        return self._status == CannonSim.HIT

    @property
    def done(self):
        """rocket is already exploded"""
        return self._status == CannonSim.HIT \
            or self._status == CannonSim.MISS
    
    @property
    def ammunition(self):
        """remaining ammunition"""
        return round(float(self.cannon_shots / self.max_shots), 3)

    def get_distance_angle(self, agent=None, target=None) -> float:
        if agent==None:
            agent = self.parent_aircraft
        ego_state = np.array([*agent.get_position(), *agent.get_velocity()])
        enm_state = np.array([*target.get_position(), *target.get_velocity()])
        ata, _, dist = get_AO_TA_R(ego_state, enm_state)
        return np.rad2deg(ata), dist

    def launch(self, parent: AircraftSim, env):
        steps_condition = False
        if getattr(env, "hl_sim_steps", False):
            steps_condition = abs(env.steps-parent.cannon.shot_step) > int(env.hl_sim_steps + 5)
        else:
            steps_condition = abs(env.steps-parent.cannon.shot_step) > parent.cannon.t_max*10
        if self.cannon_shots > 0:
            if parent.cannon.shot_step == 0 or steps_condition:
                self.parent_aircraft = parent
                self._t = 0
                self._status = CannonSim.LAUNCHED
                self.cannon_shots -= 1
                parent.cannon.shot_step = env.steps
                return True
        return False

    def fire(self, parent: AircraftSim, env):
        """Fires while it is called before every JSBSim step, until cease_fire(), e.g. while the fire button of the human is held.
        Unlike launch(), a burst lasts only as long as it is held, the next one can follow at once, and the ammunition is unlimited."""
        if self.alive:
            return True
        self.parent_aircraft = parent
        self._t = 0
        self._held = True
        self._status = CannonSim.LAUNCHED
        self.shot_step = env.steps
        return True

    def cease_fire(self):
        """Stops the burst of fire()."""
        if self._held:
            self._held = False
            if self.alive:
                self._status = CannonSim.INACTIVE

    def revert_cannon(self):
        """Reset cannon status to be ready."""
        if self.done:
            self._status = CannonSim.INACTIVE

    def run(self):
        self._t += self.dt
        for opp in self.parent_aircraft.opponents:
            ata, dist = self.get_distance_angle(target=opp)

            if (self._t > self.t_max) and not self._held:
                self._status = CannonSim.MISS
                # self.parent_aircraft.cannon = None

            elif (abs(ata) < self.omega_wez) and (dist < self.Rmax_wez):
                if opp.alive and random.choices([True, False], [self._Rhit_prob, 1-self._Rhit_prob])[0]:
                    self._status = CannonSim.HIT
                    self.target = opp
                    opp.shootdown()
                    # self.parent_aircraft.cannon = None
            # else:
            #     self.parent_aircraft.cannon_shots -= 1

            if self._status == CannonSim.MISS or self._status == CannonSim.HIT:
                break
        
    def close(self):
        self.parent_aircraft = None