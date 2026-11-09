import logging
import os
import numpy as np
import torch
from abc import ABC, abstractmethod
from gymnasium import spaces

from env.catalog import Catalog as c
from env.utils import get_AO_TA_R, LLA2NEU, get_separate_deltas, get_root_dir
from env.terminations import ExtremeState, Overload, LowAltitude, DeadAgent
from env.rewards import (
    EventDrivenReward,
    ShootReward,
    ShootPenaltyReward,
    HeadingReward,
    DistanceReward,
    FleeReward,
)


LAST_ACTIONS = False


class BaseMission(ABC):
    """
    A Mission is either a task for trainable RL agents or a model
    (pre-trained or rule-based) for non-trainable agents.
    BaseMission defines the low-level observation/action/termination/reward layout.
    """

    def __init__(self, config):
        self.log = logging.getLogger(__name__)
        self.config = config
        self.shift = 4 if LAST_ACTIONS else 0
        self.load()

    def load(self):
        opp_obs_len = 13
        self.obs_len = 17 + self.shift + opp_obs_len

        self.velocity_max = 670
        self.map_size = getattr(self.config, 'map_size', 50) * 1000
        self.loc_scale, self.loc_offset = self.shift_range(-self.map_size/2, self.map_size/2, -1, 1)
        self.loc_delta_scale, self.loc_delta_offset = self.shift_range(-self.map_size, self.map_size, -1, 1)

        self.altitude_min = getattr(self.config, 'altitude_limit', 2500)
        self.altitude_max = getattr(self.config, 'altitude_limit_max', 10000)
        self.alt_scale, self.alt_offset = self.shift_range(self.altitude_min, self.altitude_max, 0, 1)
        self.altitude_delta = self.altitude_max - self.altitude_min
        self.alt_delta_scale, self.alt_delta_offset = self.shift_range(-self.altitude_delta, self.altitude_delta, -1, 1)

        self.vel_scale, self.vel_offset = self.shift_range(-self.velocity_max, self.velocity_max, -1, 1)
        self.vel_delta_scale, self.vel_delta_offset = self.shift_range(-2*self.velocity_max, 2*self.velocity_max, -1, 1)

        self.aspect_target_scale, self.aspect_target_offset = self.shift_range(0, np.pi, 0, 1)

        self.load_obs_space()
        self.load_obs_vars()
        self.load_act_space()
        self.load_act_vars()
        self.load_terminations()
        self.load_rewards()
        self.load_model()
        self.load_render_vars()

    @abstractmethod
    def load_rewards(self):
        raise NotImplementedError

    def load_obs_space(self):
        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.obs_len,))

    def load_obs_vars(self):
        self.state_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.velocities_v_north_mps,
            c.velocities_v_east_mps,
            c.velocities_v_up_mps,
            c.velocities_vc_mps,
            c.aero_alpha_rad,
            c.aero_beta_rad,
            c.velocities_p_rad_sec,
            c.velocities_q_rad_sec,
            c.velocities_r_rad_sec,
            c.attitude_phi_rad,
            c.attitude_theta_rad,
            c.attitude_psi_rad,
        ]

    def get_obs(self, env, agent, **kwargs):
        norm_obs = np.zeros(self.obs_len, dtype=np.float32)
        ego_state = np.array(agent.get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[3:6])])

        norm_obs[0] = ego_cur_ned[0] * self.loc_scale + self.loc_offset
        norm_obs[1] = ego_cur_ned[1] * self.loc_scale + self.loc_offset
        norm_obs[2] = ego_cur_ned[2] * self.alt_scale + self.alt_offset
        norm_obs[3] = ego_state[3] * self.vel_scale + self.vel_offset
        norm_obs[4] = ego_state[4] * self.vel_scale + self.vel_offset
        norm_obs[5] = ego_state[5] * self.vel_scale + self.vel_offset
        norm_obs[6:9] = ego_state[9:12]
        norm_obs[9], norm_obs[10] = np.cos(ego_state[12]), np.sin(ego_state[12])
        norm_obs[11], norm_obs[12] = np.cos(ego_state[13]), np.sin(ego_state[13])
        norm_obs[13], norm_obs[14] = np.cos(ego_state[14]), np.sin(ego_state[14])

        norm_obs[15] = int(agent.cannon.alive)
        norm_obs[16] = agent.cannon.ammunition

        if LAST_ACTIONS:
            acts = np.array(agent.get_property_values(self.action_var))
            norm_obs[17:21] = acts

        opponent = self.get_closest_opp(agent)
        if opponent:
            opponent = opponent[0][0]
            opp_state = np.array(opponent.get_property_values(self.state_var))
            opp_cur_ned = LLA2NEU(*opp_state[:3], env.center_lon, env.center_lat, env.center_alt)
            opp_feature = np.array([*opp_cur_ned, *(opp_state[3:6])])
            a_a, a_t, _, side = get_AO_TA_R(ego_feature, opp_feature, return_side=True)
            delta_x, delta_y, delta_z, delta_vx, delta_vy, delta_vz = get_separate_deltas(ego_feature, opp_feature)

            norm_obs[17+self.shift] = delta_x * self.loc_delta_scale + self.loc_delta_offset
            norm_obs[18+self.shift] = delta_y * self.loc_delta_scale + self.loc_delta_offset
            norm_obs[19+self.shift] = delta_z * self.alt_delta_scale + self.alt_delta_offset

            norm_obs[20+self.shift] = delta_vx * self.vel_delta_scale + self.vel_delta_offset
            norm_obs[21+self.shift] = delta_vy * self.vel_delta_scale + self.vel_delta_offset
            norm_obs[22+self.shift] = delta_vz * self.vel_delta_scale + self.vel_delta_offset

            norm_obs[23+self.shift] = a_a * self.aspect_target_scale + self.aspect_target_offset
            norm_obs[24+self.shift] = a_t * self.aspect_target_scale + self.aspect_target_offset
            norm_obs[25+self.shift] = side

            norm_obs[26+self.shift], norm_obs[27+self.shift] = np.cos(opp_state[13]), np.sin(opp_state[13])
            norm_obs[28+self.shift], norm_obs[29+self.shift] = np.cos(opp_state[14]), np.sin(opp_state[14])

        assert not np.isnan(norm_obs).any(), self.log.error(f"Observation contains NaN values: {norm_obs}")
        return norm_obs

    def load_act_space(self):
        # aileron, elevator, rudder, throttle, shoot (bool)
        self.action_space = spaces.MultiDiscrete([17, 11, 9, 9, 2])

    def load_act_vars(self):
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ]

    def load_terminations(self):
        self.terminations = [
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            DeadAgent(self.config),
        ]

    def load_render_vars(self):
        self.render_var = [
            c.position_lat_gc_rad,
            c.position_long_gc_rad,
            c.position_h_sl_m,
            c.attitude_phi_rad,
            c.attitude_theta_rad,
            c.attitude_psi_rad,
        ]

    def load_model(self):
        """Load pre-trained or rule-based model for non-rl agents."""
        pass

    def reset(self, env):
        for rew in self.rewards:
            rew.reset(self, env)

    def step(self, env, agent):
        pass

    def get_action(self, action, agent, env):
        """Convert discrete action index into continuous value."""
        norm_act = np.zeros(4)
        act_space = spaces.MultiDiscrete([17, 11, 9, 9, 2])
        scale, offset = self.shift_range(0, act_space.nvec[0]-1, -0.8, 0.8)
        norm_act[0] = action[0]*scale + offset
        scale, offset = self.shift_range(0, act_space.nvec[1]-1, -0.8, 0.8)
        norm_act[1] = action[1]*scale + offset
        scale, offset = self.shift_range(0, act_space.nvec[2]-1, -0.5, 0.5)
        norm_act[2] = action[2]*scale + offset
        scale, offset = self.shift_range(0, act_space.nvec[3]-1, 0.3, 0.9)
        norm_act[3] = action[3]*scale + offset
        return norm_act

    def get_reward(self, env, agent_id, action):
        reward = 0.0
        for rew in self.rewards:
            reward += rew.get_reward(self, env, agent_id, action)
        return np.round(reward, 5)

    def get_termination(self, env, agent_id, info={}, get_info=False):
        done = not env.agents[agent_id].alive
        success = True
        if not done:
            for condition in self.terminations:
                d, s, info = condition.get_termination(self, env, agent_id, info)
                done = done or d
                success = success and s
                if done:
                    break
        return (done, info) if get_info else done

    def get_closest_opp(self, agent):
        opps = [[opp, agent.get_distance_to_target(opp)] for opp in agent.opponents if opp.alive]
        return sorted(opps, key=lambda x: x[1])

    def get_closest_friendly(self, agent):
        fris = [[fri, agent.get_distance_to_target(fri)] for fri in agent.partners]
        return sorted(fris, key=lambda x: x[1])

    def shift_range(self, a, b, c, d):
        scale = (d - c) / (b - a)
        offset = c - a * scale
        return scale, offset


class TrackTask(BaseMission):
    """Engage Task: track (pursue) target. Parent class of CannonTask and EscapeTask."""

    def load_rewards(self):
        self.rewards = [
            ShootPenaltyReward(self.config),
            EventDrivenReward(self.config),
            HeadingReward(self.config),
        ]

    def get_action(self, action, agent, env):
        shoot_action = action[4]
        action = super().get_action(action, agent, env)
        self._cannon(shoot_action, agent, env)
        return action

    def _cannon(self, shoot_action, agent, env):
        if agent.alive and shoot_action and agent.cannon.ready:
            self.shot_fired[agent.aid] = agent.cannon.launch(parent=agent, env=env)
        else:
            self.shot_fired[agent.aid] = False

    def reset(self, env):
        self.shot_fired = {aid: False for aid in env.agents.keys()}
        return super().reset(env)

    def step(self, env, agent):
        if agent.cannon.done:
            agent.cannon.revert_cannon()


class CannonTask(TrackTask):
    """Attack Task for aggressively destroying opponents."""

    def load_rewards(self):
        self.rewards = [
            ShootReward(self.config),
            ShootPenaltyReward(self.config),
            EventDrivenReward(self.config),
            HeadingReward(self.config),
        ]


class EscapeTask(TrackTask):
    """Defend Task to escape from opponent."""

    def load_rewards(self):
        self.rewards = [
            ShootPenaltyReward(self.config),
            EventDrivenReward(self.config),
            DistanceReward(self.config),
            HeadingReward(self.config),
            FleeReward(self.config),
        ]


class HierTask(BaseMission):
    """High-level commander policy (the agent being trained).

    The commander's own obs/action space is high-level (Discrete over cannon/track/
    escape). Each high-level action selects a pre-trained low-level policy whose
    multi-discrete control output is what actually drives the aircraft. Unlike
    HierModel (whose policy choice comes from a scripted strategy in set_policy),
    HierTask receives its choice from the high-level policy being learned.
    """

    HL_ACTION_MAPPING = {0: 'cannon', 1: 'track', 2: 'escape'}
    # Low-level pre-trained policies, keyed by task -> runs/<dir>. These are the
    # default f16 policies; for a non-f16 aircraft model the trailing "_f16" token
    # is swapped for the model name (e.g. cannon_L4_A4) in _policy_paths_for_model.
    POLICY_PATHS = {'cannon': 'cannon_L4_f16', 'track': 'track_L4_f16', 'escape': 'escape_L2_f16'}
    # Fixed low-level control action layout (BaseMission.load_act_space) that the
    # dispatched low-level policies output: aileron, elevator, rudder, throttle, shoot.
    LOW_LEVEL_NVEC = [17, 11, 9, 9, 2]
    # Fixed observation window: the commander always sees the N closest opponents and
    # friendlies, regardless of roster size (e.g. unchanged obs shape in 3v3 vs 7v7).
    NUM_OBS_OPPS = 3
    NUM_OBS_FRIENDS = 2

    def load_rewards(self):
        self.rewards = [
            EventDrivenReward(self.config),
            ShootReward(self.config),
        ]

    def load_act_space(self):
        # High-level commander action: which low-level policy to run.
        self.action_space = spaces.Discrete(3)
        self.hl_action_mapping = self.HL_ACTION_MAPPING

    def load_obs_space(self):
        # Actual roster sizes (still used for the reset consistency check). The obs
        # window itself is fixed (see below), so the obs shape is roster-independent.
        self.num_rl_agents = getattr(self.config, 'num_rl_agents', 3)
        self.num_opp_agents = getattr(self.config, 'num_opp_agents', 3)
        
        # Fixed observation window: at most NUM_OBS_OPPS opponents and NUM_OBS_FRIENDS
        # friendlies (the closest ones).
        self.own_obs_len = 12
        self.opp_obs = self.fri_obs = 10
        self.opp_obs_len = self.opp_obs * self.NUM_OBS_OPPS
        self.fri_obs_len = self.fri_obs * self.NUM_OBS_FRIENDS
        self.observation_space = spaces.Box(
            low=-1, high=1.0,
            shape=(self.own_obs_len + self.opp_obs_len + self.fri_obs_len,),
        )

    def _policy_paths_for_model(self, model):
        """Resolve the low-level policy run dirs for a given aircraft model.

        The default POLICY_PATHS target f16. For any other model (e.g. A4) the
        trailing "_f16" token is swapped for the model name, yielding e.g.
        cannon_L4_A4 / track_L4_A4 / escape_L2_A4.
        """
        if model == 'f16':
            return dict(self.POLICY_PATHS)
        return {
            task: path[:-len('_f16')] + f'_{model}' if path.endswith('_f16') else f'{path}_{model}'
            for task, path in self.POLICY_PATHS.items()
        }

    def _validate_model_policies(self, model, policy_paths):
        """Ensure every pre-trained policy for a non-f16 model exists, else abort.

        The default f16 policies are assumed present (validation happens at load
        time anyway). For heterogeneous combat where an aircraft requests a
        different model in hier.yaml, the matching <task>_L*_<model> runs must
        have been pre-trained; if any is missing we raise so training aborts with
        a clear message instead of silently driving an A4 with an f16 policy.
        """
        runs_dir = os.path.join(get_root_dir(), 'runs')
        missing = [
            name for name in policy_paths.values()
            if not os.path.isdir(os.path.join(runs_dir, name))
        ]
        if missing:
            raise FileNotFoundError(
                f"No pre-trained '{model}' low-level policies found for hierarchical "
                f"training: missing runs/{', runs/'.join(missing)}. Aircraft requesting "
                f"model '{model}' in hier.yaml require these policies to be pre-trained "
                f"under '{model}' flight dynamics. Aborting training."
            )

    def load_model(self):
        """Instantiate the pre-trained low-level actor modules (built like StrategyModel).

        These run on the low-level obs/action layout (BaseMission.obs_len / LOW_LEVEL_NVEC),
        which differs from this commander's own high-level obs/action space above.

        One policy set is loaded per distinct aircraft model among the RL agents
        that this commander drives (the task_hier aircraft). For non-f16 models the
        matching <task>_L*_<model> runs are validated to exist first (see
        _validate_model_policies), so heterogeneous combat with e.g. an A4 only
        proceeds when A4 low-level policies were actually pre-trained.
        """
        from env.models import build_low_level_actors
        from algorithm.networks import act_with_policies
        self._act_with_policies = act_with_policies

        aircraft = getattr(self.config, 'aircraft', {})
        models = {
            cfg.get('model', 'f16')
            for cfg in aircraft.values()
            if cfg and cfg.get('mission') == 'task_hier'
        }
        models.add('f16')  # always available as the default fallback

        self.policies_by_model = {}
        for model in models:
            policy_paths = self._policy_paths_for_model(model)
            if model != 'f16':
                self._validate_model_policies(model, policy_paths)
            self.policies_by_model[model], self.actor_args = build_low_level_actors(
                self.obs_len, self.LOW_LEVEL_NVEC, policy_paths,
            )
        # Back-compat alias: f16 policy set used when an agent's model is unknown.
        self.policies = self.policies_by_model['f16']

    def reset(self, env):
        # Aircraft flown by a human (env.human_agent_ids) replace RL agents, so all aircraft are counted
        assert self.num_rl_agents + self.num_opp_agents == len(env.agents), \
            f"Specified num agents {self.num_rl_agents + self.num_opp_agents} does not match the {len(env.agents)} aircraft of the env. Adjust your configuration!"
        self.policy_to_use = {aid: None for aid in env.agents.keys()}
        return super().reset(env)

    def step(self, env, agent):
        if agent.cannon.done:
            agent.cannon.revert_cannon()

    def get_obs(self, env, agent, **kwargs):
        own_obs = np.zeros(self.own_obs_len, dtype=np.float32)
        fri_obs = np.zeros(self.fri_obs_len, dtype=np.float32)
        opp_obs = np.zeros(self.opp_obs_len, dtype=np.float32)

        ego_state = np.array(agent.get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[3:6])])

        # location
        own_obs[0] = ego_cur_ned[0] * self.loc_scale + self.loc_offset
        own_obs[1] = ego_cur_ned[1] * self.loc_scale + self.loc_offset
        own_obs[2] = ego_cur_ned[2] * self.alt_scale + self.alt_offset

        # velocity
        own_obs[3] = ego_state[3] * self.vel_scale + self.vel_offset
        own_obs[4] = ego_state[4] * self.vel_scale + self.vel_offset
        own_obs[5] = ego_state[5] * self.vel_scale + self.vel_offset

        # orientation
        own_obs[6], own_obs[7] = np.cos(ego_state[12]), np.sin(ego_state[12])
        own_obs[8], own_obs[9] = np.cos(ego_state[13]), np.sin(ego_state[13])
        own_obs[10], own_obs[11] = np.cos(ego_state[14]), np.sin(ego_state[14])

        # Only the NUM_OBS_OPPS closest (alive) opponents are observed; get_closest_opp
        # returns them distance-sorted, so we stop once the window is full.
        opp_idx = 0
        for opp, _ in super().get_closest_opp(agent):
            if opp_idx >= self.NUM_OBS_OPPS:
                break
            if not opp.alive:
                continue
            opp_state = np.array(opp.get_property_values(self.state_var))
            opp_cur_ned = LLA2NEU(*opp_state[:3], env.center_lon, env.center_lat, env.center_alt)
            opp_feature = np.array([*opp_cur_ned, *(opp_state[3:6])])

            a_a, a_t, _, side = get_AO_TA_R(ego_feature, opp_feature, return_side=True)
            delta_x_o, delta_y_o, delta_z_o, _, _, _ = get_separate_deltas(ego_feature, opp_feature)

            base = opp_idx * self.opp_obs

            opp_obs[base+0] = delta_x_o * self.loc_delta_scale + self.loc_delta_offset
            opp_obs[base+1] = delta_y_o * self.loc_delta_scale + self.loc_delta_offset
            opp_obs[base+2] = delta_z_o * self.loc_delta_scale + self.loc_delta_offset

            opp_obs[base+3] = a_a * self.aspect_target_scale + self.aspect_target_offset
            opp_obs[base+4] = a_t * self.aspect_target_scale + self.aspect_target_offset
            opp_obs[base+5] = side

            opp_obs[base+6], opp_obs[base+7] = np.cos(opp_state[13]), np.sin(opp_state[13])
            opp_obs[base+8], opp_obs[base+9] = np.cos(opp_state[14]), np.sin(opp_state[14])

            opp_idx += 1

        # Only the NUM_OBS_FRIENDS closest (alive) friendlies are observed.
        fri_idx = 0
        for fri, _ in super().get_closest_friendly(agent):
            if fri_idx >= self.NUM_OBS_FRIENDS:
                break
            if not fri.alive:
                continue
            fri_state = np.array(fri.get_property_values(self.state_var))
            fri_cur_ned = LLA2NEU(*fri_state[:3], env.center_lon, env.center_lat, env.center_alt)
            fri_feature = np.array([*fri_cur_ned, *(fri_state[3:6])])
            a_a, a_t, _, side = get_AO_TA_R(ego_feature, fri_feature, return_side=True)
            delta_x_f, delta_y_f, delta_z_f, _, _, _ = get_separate_deltas(ego_feature, fri_feature)

            base = fri_idx * self.fri_obs

            fri_obs[base+0] = delta_x_f * self.loc_delta_scale + self.loc_delta_offset
            fri_obs[base+1] = delta_y_f * self.loc_delta_scale + self.loc_delta_offset
            fri_obs[base+2] = delta_z_f * self.loc_delta_scale + self.loc_delta_offset

            fri_obs[base+3] = a_a * self.aspect_target_scale + self.aspect_target_offset
            fri_obs[base+4] = a_t * self.aspect_target_scale + self.aspect_target_offset
            fri_obs[base+5] = side

            fri_obs[base+6], fri_obs[base+7] = np.cos(fri_state[13]), np.sin(fri_state[13])
            fri_obs[base+8], fri_obs[base+9] = np.cos(fri_state[14]), np.sin(fri_state[14])

            fri_idx += 1

        assert not np.isnan(own_obs).any(), self.log.error(f"Own Observation contains NaN values: {own_obs}")
        assert not np.isnan(fri_obs).any(), self.log.error(f"Fri Observation contains NaN values: {fri_obs}")
        assert not np.isnan(opp_obs).any(), self.log.error(f"Opp Observation contains NaN values: {opp_obs}")
        return np.concatenate((own_obs, opp_obs, fri_obs), axis=0, dtype=np.float32)

    def set_policy(self, action, agent, env):
        """Assign the low-level policy chosen by the learned high-level action.

        `action` is the commander's discrete choice (0=cannon, 1=track, 2=escape).
        This is the key difference from HierModel, which derives the choice from a
        scripted strategy instead.
        """
        policy_type = self.hl_action_mapping[int(action)]
        policies = self.policies_by_model.get(getattr(agent, 'model', 'f16'), self.policies)
        self.policy_to_use[agent.aid] = policies[policy_type]

    def prepare_action(self, agent, env):
        """Sets the flight controls chosen by the selected low-level policy."""
        # Low-level obs/control use BaseMission's layout (30-dim obs, multi-discrete action).
        ll_obs = super().get_obs(env=env, agent=agent)
        ll_action = self.get_ll_action(ll_obs, agent)
        if bool(ll_action[4]):
            self._cannon(agent, env)
        ll_action = super().get_action(action=ll_action, agent=agent, env=env)
        agent.set_property_values(self.action_var, ll_action)

    def run_action(self, agent, env):
        self.prepare_action(agent, env)
        for _ in range(env.sim_steps):
            agent.run()

    def get_ll_action(self, obs, agent):
        """Sample deterministic action from the chosen pre-trained low-level policy."""
        actors = self.policy_to_use[agent.aid]
        assert actors is not None, f"No low-level policy selected for agent {agent.aid}"
        obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            actions, _, _, _ = self._act_with_policies(
                self.actor_args, actors, obs_t, rnn_hxs=None, masks=None, deterministic=True,
            )
        return actions[0, 0].cpu().numpy()

    def _cannon(self, agent, env):
        if agent.alive and agent.cannon.ready:
            agent.cannon.launch(parent=agent, env=env)
