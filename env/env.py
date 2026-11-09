import logging
import importlib
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from typing import Any, Tuple
from env.sim import AircraftSim
from env.utils import parse_config, LLA2NEU, NEU2LLA
from rendering.scene import Scene


class AircraftEnv(gym.Env):
    """Base class for low-level aircraft combat training.

    Optional config entries besides the mission config:
      config_overrides (dict): deep-merged into the mission config, e.g. {"battle_field_center": [lon, lat, 0.0]}.
      flightsim (dict): shows the aircraft in FlightSim, with the arguments of rendering.flightsim.FlightSimBridge,
        e.g. {"jsbsim_states": "python", "manual_agent": "B2"}. The manual agent is flown by a human with the joysticks or the
        keyboard (mission model_human). It is added as a copy of the first aircraft of its team if the config has no such aircraft.
    """

    metadata = {"render.modes": ["simple", "fg"]}

    def __init__(self, config: dict):
        self.log = logging.getLogger(__name__)
        self.config = parse_config(config["mission_config"], config.get("config_overrides"))
        flightsim_options = config.get("flightsim")
        if flightsim_options and flightsim_options.get("manual_agent"):
            self._add_human_agent(flightsim_options["manual_agent"])
        self.hierarchy_level = config["hierarchy_level"]
        self.training_level = config["training_level"]
        self.render_samples = config.get("render_samples", 4)
        self.rnd_init_states = getattr(self.config, "rnd_init", True)
        self.horizon = getattr(self.config, "horizon", 1000)
        self.sim_freq = getattr(self.config, "sim_freq", 100)
        self.sim_steps = getattr(self.config, "sim_steps", 10)
        self.center_lon, self.center_lat, self.center_alt = getattr(
            self.config, "battle_field_center", (0.0, 0.0, 0.0)
        )
        self.steps = 0
        self.scene = None
        self._alive_last_step = set()
        self.load_missions()
        self.load_agents()
        self.load_spaces()
        self.seed()
        self.flightsim = None
        if flightsim_options is not None:
            from rendering.flightsim import FlightSimBridge
            self.flightsim = FlightSimBridge(self, **flightsim_options)

    def _add_human_agent(self, aid):
        """Lets a human fly aircraft aid, see the class docstring."""
        aircraft = self.config.aircraft
        if aid in aircraft:
            aircraft[aid] = {**aircraft[aid], "mission": "model_human"}
        else:
            team = [cfg for other, cfg in aircraft.items() if other[0] == aid[0]]
            assert team, f"Manual agent {aid}: the config has no aircraft of team {aid[0]} to copy"
            aircraft[aid] = {**team[0], "mission": "model_human"}

    def load_missions(self):
        # Lookup order matters: missions module hosts trainable Tasks (with cannon firing),
        # models module hosts non-trainable opponent policies. A mission string of the form
        # "task_cannon" maps to CannonTask in missions; "model_strategy" -> StrategyModel in models.
        missions_mod = importlib.import_module("env.missions")
        models_mod = importlib.import_module("env.models")
        self.missions = {}
        for config in self.config.aircraft.values():
            mission = config.get("mission")
            assert mission, "Specify mission per aircraft in .yaml config."
            if mission in self.missions:
                continue
            mission_type, mission_name = mission.split("_")[0], mission.split("_")[1]
            class_name = f"{mission_name.capitalize()}{mission_type.capitalize()}"
            cls = getattr(missions_mod, class_name, None) or getattr(models_mod, class_name, None)
            if cls is None:
                raise NotImplementedError(f"Unknown Mission: {class_name}")
            self.missions[mission] = cls(self.config)

    def load_agents(self):
        self.agents = {}
        self.teams = {}
        self.rl_agent_ids = []
        self.opp_agent_ids = []
        self.human_agent_ids = []  # aircraft flown by a human, which are not opponents (see _add_human_agent)
        origin = getattr(self.config, "battle_field_center", (0.0, 0.0, 0.0))
        for aid, config in self.config.aircraft.items():
            mission = config.get("mission")
            is_learning = mission.split("_")[0] == "task"
            is_human = mission == "model_human"
            self.agents[aid] = AircraftSim(
                tid=aid[0],
                aid=aid,
                mission=self.missions[mission],
                learning=is_learning,
                model=config.get("model", "f16"),
                num_missiles=config.get("num_missiles", 5),
                init_state=config.get("init_state"),
                origin=origin,
                sim_freq=self.sim_freq,
                sim_steps=self.sim_steps,
                unlimited_fuel=config.get("unlimited_fuel", True),
            )
            if is_learning:
                self.rl_agent_ids.append(aid)
            elif is_human:
                self.human_agent_ids.append(aid)
            else:
                self.opp_agent_ids.append(aid)
            self.teams.setdefault(aid[0], []).append(aid)

        for aid, agent in self.agents.items():
            agent.partners = [self.agents[a] for a in self.teams[agent.tid] if a != aid]
            agent.opponents = [
                self.agents[a]
                for tid, team in self.teams.items() if tid != agent.tid
                for a in team
            ]

    def load_spaces(self):
        obs, act = {}, {}
        for aid, agent in self.agents.items():
            if agent.is_learning:
                obs[aid] = agent.mission.observation_space
                act[aid] = agent.mission.action_space
        self.observation_space = gym.spaces.Dict(obs)
        self.action_space = gym.spaces.Dict(act)
        self.log.info(f"Observation space={self.observation_space}, Action space={self.action_space}")

    def reset_agent_states(self, init_states=None):
        """Reloads the aircraft at their initial states: init_states[aid] if given (see to_init_state),
        otherwise the init_state of the config, randomized if rnd_init."""
        init_states = init_states or {}
        lvl1 = self.training_level == 1
        hier1 = self.hierarchy_level == 1
        rng = self.np_random
        for aid, agent in self.agents.items():
            if aid in init_states:
                agent.reload(new_state=self.to_init_state(init_states[aid]))
                continue
            state = dict(self.config.aircraft[aid].get("init_state") or {})
            if self.rnd_init_states:
                for sk in state:
                    if sk == "ic_psi_true_deg":
                        state[sk] = int(rng.integers(0, 350))
                    elif sk in ("ic_long_gc_deg", "ic_lat_geod_deg"):
                        if hier1:
                            state[sk] = rng.uniform(-0.15, 0.15)
                        elif lvl1:
                            state[sk] = rng.uniform(-0.06, 0.06)
                        else:
                            state[sk] = rng.uniform(-0.13, 0.13)
                    elif sk == "ic_h_sl_ft":
                        state[sk] = int(rng.integers(25000, 26000)) if lvl1 else int(rng.integers(22000, 28000))
                    elif sk == "ic_u_fps":
                        is_B = agent.tid == "B"
                        state[sk] = (
                            int(rng.integers(700, 900)) if (lvl1 and is_B)
                            else int(rng.integers(500, 700)) if lvl1
                            else int(rng.integers(500, 1000))
                        )
                    else:
                        raise ValueError(f"Init state key {sk} not randomized.")
            agent.reload(new_state=self._relocate(state))

    def _relocate(self, state):
        """Moves an initial state of the configs, whose positions are around longitude and latitude 0,
        to the same north/east offset from the battle field center."""
        if (self.center_lon, self.center_lat) == (0.0, 0.0) or "ic_long_gc_deg" not in state:
            return state
        north, east, _ = LLA2NEU(state["ic_long_gc_deg"], state["ic_lat_geod_deg"], 0.0)
        lon, lat, _ = NEU2LLA(north, east, 0.0, self.center_lon, self.center_lat, 0.0)
        return {**state, "ic_long_gc_deg": float(lon), "ic_lat_geod_deg": float(lat)}

    def to_init_state(self, spec):
        """Converts an initial state relative to the battle field center into JSBSim initial conditions.
        spec: "north" and "east" in m from the battle field center (default 0), or "lat" and "lon" in degrees,
        "alt" in m above sea level, "speed" in m/s along the aircraft axis, "heading" in degrees (default 0),
        and optionally "roll" and "pitch" in degrees."""
        if "lat" in spec or "lon" in spec:
            lon, lat = spec["lon"], spec["lat"]
        else:
            lon, lat, _ = NEU2LLA(spec.get("north", 0.0), spec.get("east", 0.0), 0.0, self.center_lon, self.center_lat, 0.0)
        state = {
            "ic_long_gc_deg": float(lon),
            "ic_lat_geod_deg": float(lat),
            "ic_h_sl_ft": spec["alt"] / 0.3048,
            "ic_psi_true_deg": spec.get("heading", 0.0) % 360,
            "ic_u_fps": spec["speed"] / 0.3048,
        }
        if "roll" in spec:
            state["ic_phi_deg"] = spec["roll"]
        if "pitch" in spec:
            state["ic_theta_deg"] = spec["pitch"]
        return state

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
        """options: {"init_states": {aid: spec}} sets the initial states of the given aircraft, see to_init_state()."""
        if seed is not None:
            self.seed(seed)
        self.steps = 0
        self.reset_agent_states((options or {}).get("init_states"))
        for mission in self.missions.values():
            mission.reset(self)
        self._alive_last_step = {aid for aid in self.agents}
        if self.flightsim is not None:
            self.flightsim.on_reset(self)
        return self.get_obs(), {}

    def step(self, action: Any) -> Tuple[dict, dict, dict, dict, dict]:
        self.steps += 1
        timed_out = self.steps >= self.horizon

        for aid, agent in self.agents.items():
            if not agent.alive:
                continue
            act = agent.mission.get_action(action=action.get(aid), agent=agent, env=self)
            agent.set_property_values(agent.mission.action_var, act)

        for _ in range(self.sim_steps):
            self.run_substep(self.agents.values())

        obs = {}
        terminated = {}
        for aid, agent in self.agents.items():
            obs.update(self.get_obs(aid))
            if agent.is_learning:
                terminated[aid] = bool(agent.mission.get_termination(self, aid))

        rewards = {}
        for aid, agent in self.agents.items():
            rewards.update(self.get_reward(aid, action))
            agent.mission.step(self, agent)

        info = self._combat_info()
        self._alive_last_step = {aid for aid, agent in self.agents.items() if agent.alive}
        terminated["all"] = all(terminated[aid] for aid in self.rl_agent_ids) or self._opponents_eliminated()
        truncated = {aid: timed_out for aid in self.rl_agent_ids}
        truncated["all"] = timed_out
        self.log.debug(f"step={self.steps}, term={terminated}, trunc={truncated}")
        return obs, rewards, terminated, truncated, info

    def run_substep(self, agents):
        """Advances the given aircraft by one JSBSim step (1/sim_freq seconds)."""
        if self.flightsim is not None:
            self.flightsim.before_substep(self)
        for agent in agents:
            agent.run()
        if self.flightsim is not None:
            self.flightsim.after_substep(self)

    def get_obs(self, aid: str = None) -> dict:
        if aid is not None:
            agent = self.agents[aid]
            obs = {aid: agent.mission.get_obs(self, agent)} if agent.is_learning else {}
        else:
            obs = {
                aid: agent.mission.get_obs(self, agent)
                for aid, agent in self.agents.items() if agent.is_learning
            }
        for k, v in obs.items():
            assert not np.isnan(v).any(), self.log.error(f"Observation of agent {k} contains NaN: {v}")
        return obs

    def get_reward(self, aid: str, action: dict, raw_value: bool = False) -> dict:
        if not self.agents[aid].is_learning:
            return {}
        rew = self.agents[aid].mission.get_reward(self, aid, action)
        return rew if raw_value else {aid: rew}

    def _opponents_eliminated(self) -> bool:
        return any(
            all(not self.agents[aid].alive for aid in team_aids)
            for team_aids in self.teams.values()
        )

    def _combat_info(self) -> dict:
        opponents = [self.agents[aid] for aid in self.opp_agent_ids]
        friendlies = [self.agents[aid] for aid in self.rl_agent_ids]
        return {
            "opp_killed":   any(a.shotdown and a.aid in self._alive_last_step for a in opponents),
            "opp_crashed":  any(a.crashed  and a.aid in self._alive_last_step for a in opponents),
            "agent_killed": any(a.shotdown and a.aid in self._alive_last_step for a in friendlies),
            "agent_crashed":any(a.crashed  and a.aid in self._alive_last_step for a in friendlies),
        }

    def close(self):
        for agent in self.agents.values():
            agent.close()
        self.agents.clear()
        if self.flightsim is not None:
            self.flightsim.close()
            self.flightsim = None

    def render(self, viewpoint=0):
        if self.flightsim is not None:
            return  # FlightSim renders after every JSBSim step, see FlightSimBridge
        if not self.scene:
            self.scene = Scene(self)
        self.scene.render(viewpoint)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class AircraftEnvHier(AircraftEnv):
    """AircraftEnv for high-level commander (hierarchy_level==2) training.

    One environment step is a *high-level* step: the commander assigns a low-level
    policy (cannon/track/escape) to each aircraft, and that policy is rolled out for
    `hl_sim_steps` low-level steps. Rewards accumulate over the inner steps.
    """

    def __init__(self, config: dict, inference: bool = False, rendering: bool = False, render_view: int = 0):
        super().__init__(config)
        self.hl_sim_steps = getattr(self.config, "hl_sim_steps", 15)
        self.inference = inference
        self.rendering = rendering
        self.render_view = render_view
        assert len(self.rl_agent_ids) > 0 and len(self.opp_agent_ids) > 0, \
            f"RL agents {len(self.rl_agent_ids)} and Opp agents {len(self.opp_agent_ids)}"

    def step(self, action):
        self.steps += 1
        timed_out = self.steps >= self.horizon
        rewards = {aid: 0.0 for aid in self.rl_agent_ids}

        # 1. Commander assigns a low-level policy to every (alive) aircraft.
        #    RL agents use the learned high-level `action`; opponents use a scripted strategy.
        for aid, agent in self.agents.items():
            if agent.alive:
                agent.mission.set_policy(action=action.get(aid, None), agent=agent, env=self)

        # 2. Roll the chosen low-level policies out for hl_sim_steps low-level steps,
        #    accumulating per-agent reward and stopping early on a terminal event.
        terminated_event = False
        for _ in range(self.hl_sim_steps):
            if self.rendering:
                self.render(viewpoint=self.render_view)
            alive_before = {aid for aid, agent in self.agents.items() if agent.alive}
            if self.flightsim is None:
                for aid, agent in self.agents.items():
                    if not agent.alive:
                        continue
                    agent.mission.run_action(agent=agent, env=self)
            else:
                # FlightSim shows all aircraft at the same time, so they are simulated in lockstep:
                # all aircraft set their controls first, then all are advanced together.
                for agent in self.agents.values():
                    if agent.alive:
                        agent.mission.prepare_action(agent=agent, env=self)
                for _ in range(self.sim_steps):
                    self.run_substep([agent for agent in self.agents.values() if agent.alive])
            for aid, agent in self.agents.items():
                # Agents alive before this low-level step, so that an agent shot down or crashed in it gets the penalty
                # of EventDrivenReward once. Its terminations are checked here, so that a crash ends it at once.
                if agent.is_learning and aid in alive_before:
                    agent.mission.get_termination(self, aid)
                    rewards[aid] += self.get_reward(aid, action, raw_value=True)
                if agent.alive:
                    agent.mission.step(self, agent)
            if self._opponents_eliminated() or all(
                not self.agents[aid].alive for aid in self.rl_agent_ids
            ):
                terminated_event = True
                break

        # 3. Per-agent termination/truncation for the learning agents.
        terminated = {
            aid: bool(self.agents[aid].mission.get_termination(self, aid))
            for aid in self.rl_agent_ids
        }
        truncated = {aid: (timed_out and not terminated[aid]) for aid in self.rl_agent_ids}

        obs = {}
        for aid in self.rl_agent_ids:
            obs.update(self.get_obs(aid))

        info = self._combat_info()
        self._alive_last_step = {aid for aid, agent in self.agents.items() if agent.alive}

        # The time limit is a truncation, not a terminal state (the learner bootstraps the value after it)
        all_done = (
            all(terminated[aid] for aid in self.rl_agent_ids)
            or self._opponents_eliminated()
            or terminated_event
        )
        terminated["all"] = all_done
        truncated["all"] = timed_out and not all_done

        self.log.debug(f"hl_step={self.steps}, term={terminated}, trunc={truncated}")
        return obs, rewards, terminated, truncated, info

    def get_obs(self, aid: str = None) -> dict:
        """Like the base get_obs, but a dead learning agent yields a fixed-shape zero
        vector instead of querying JSBSim (whose state is undefined once destroyed)."""
        if aid is not None:
            agent = self.agents[aid]
            if not agent.is_learning:
                return {}
            if not agent.alive:
                shape = self.observation_space[aid].shape
                return {aid: np.zeros(shape, dtype=np.float32)}
            return {aid: agent.mission.get_obs(self, agent)}
        return {
            aid: self.get_obs(aid)[aid]
            for aid, agent in self.agents.items() if agent.is_learning
        }


class AircombatMARL:
    """Adapts AircraftEnv to the interface expected by the learner/workers.

    hierarchy_level==2 selects AircraftEnvHier (high-level commander training);
    any other value selects the base low-level AircraftEnv.
    """

    def __init__(
        self,
        task=None,
        seed=None,
        **kwargs,
    ):
        config = {"mission_config": task, **kwargs}  # kwargs carries hierarchy_level/training_level
        hierarchy_level = kwargs.get("hierarchy_level", 0)
        if hierarchy_level == 2:
            self.env = AircraftEnvHier(
                config,
                inference=kwargs.get("inference", False),
                rendering=kwargs.get("rendering", False),
                render_view=kwargs.get("render_view", 0),
            )
        else:
            self.env = AircraftEnv(config)
        if seed is not None:
            self.env.seed(seed)
        self.agent_keys = list(self.env.rl_agent_ids)
        self.first_agent = self.agent_keys[0]
        self.n_agents = len(self.agent_keys)
        
        # The commander uses spaces.Discrete; low-level tasks use MultiDiscrete.
        self._act_space = self.env.action_space[self.first_agent]
        self._is_discrete = isinstance(self._act_space, gym.spaces.Discrete)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed, options)

    def step(self, actions):
        return self.env.step(actions)

    def get_obs_size(self):
        return self.env.observation_space[self.first_agent].shape[0]

    def get_state_size(self):
        return self.env.observation_space[self.first_agent].shape[0]

    def get_state(self):
        """State of the centralized critic: the obs of the first alive RL agent (the obs of a dead agent is zeros in
        AircraftEnvHier), which also contains its closest opponents and friendlies."""
        aid = next((aid for aid in self.agent_keys if self.env.agents[aid].alive), self.first_agent)
        return self.env.get_obs(aid)[aid]

    def get_action_size(self):
        if self._is_discrete:
            return [int(self._act_space.n)]
        return list(self._act_space.nvec)

    def get_action_space(self):
        if self._is_discrete:
            return ("discrete", int(self._act_space.n))
        return ("multi_discrete", list(self._act_space.nvec))

    def get_avail_actions(self):
        n = int(self._act_space.n) if self._is_discrete else len(self._act_space.nvec)
        return {aid: np.ones(n, dtype=np.float32) for aid in self.agent_keys}

    def get_agents(self):
        return self.agent_keys

    def close(self):
        self.env.close()
