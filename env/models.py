import os
import random
import numpy as np
import torch
from gymnasium import spaces

from env.catalog import Catalog as c
from env.missions import BaseMission
from env.base_actor.baseline_actor import BaselineActor
from env.utils import get_root_dir, in_range_rad, load_pretrained_actors
from args import Args


class TrackModel(BaseMission):
    """Pre-trained agent to track (pursue) given target."""

    def load_act_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def load_obs_space(self):
        # Non-RL agents do not need a public obs space.
        pass

    def load_obs_vars(self):
        self.state_var = [
            c.delta_altitude,
            c.delta_heading,
            c.delta_velocities_u,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.velocities_u_mps,
            c.velocities_v_mps,
            c.velocities_w_mps,
            c.velocities_vc_mps,
            c.position_h_sl_m,
        ]

    def load_rewards(self):
        self.rewards = []

    def load_model(self):
        self.model_path = os.path.join(get_root_dir(), "env", "base_actor", "baseline_actor.pt")
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
        self.actor.eval()

    def reset(self, env):
        self.rnn_states = np.zeros((1, 1, 128))
        super().reset(env)

    def get_obs(self, env, agent, **kwargs):
        delta_value = kwargs["kwargs"].get("delta", None)
        assert delta_value is not None, self.log.error(f"Delta Value in {self.__class__.__name__} is missing.")
        obs = agent.get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000
        norm_obs[1] = in_range_rad(delta_value[1])
        norm_obs[2] = delta_value[2] / 340
        norm_obs[3] = obs[9] / 5000
        norm_obs[4] = np.sin(obs[3])
        norm_obs[5] = np.cos(obs[3])
        norm_obs[6] = np.sin(obs[4])
        norm_obs[7] = np.cos(obs[4])
        norm_obs[8] = obs[5] / 340
        norm_obs[9] = obs[6] / 340
        norm_obs[10] = obs[7] / 340
        norm_obs[11] = obs[8] / 340
        return np.expand_dims(norm_obs, axis=0)

    def get_action(self, action=None, agent=None, env=None):
        delta_value = self.set_delta_value(agent)
        observation = self.get_obs(None, agent, kwargs={"delta": delta_value})
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        return self.shift_action(_action.detach().cpu().numpy().squeeze())

    def shift_action(self, action):
        """The baseline actor was trained under a different action space; remap to this env's action ranges."""
        norm_act = np.zeros(4)
        scale, offset = self.shift_range(0, self.action_space.nvec[0]-1, -1, 1)
        norm_act[0] = action[0]*scale + offset
        norm_act[1] = action[1]*scale + offset
        norm_act[2] = action[2]*scale + offset
        scale, offset = self.shift_range(0, self.action_space.nvec[3]-1, 0.4, 0.9)
        norm_act[3] = action[3]*scale + offset
        return norm_act

    def set_delta_value(self, agent):
        ego_x, ego_y, ego_z = agent.get_position()
        ego_vx, ego_vy, _ = agent.get_velocity()
        opp_x, opp_y, opp_z = agent.opponents[0].get_position()
        delta_altitude = opp_z - ego_z
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = opp_x - ego_x, opp_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        delta_velocity = (
            agent.opponents[0].get_property_value(c.velocities_u_mps)
            - agent.get_property_value(c.velocities_u_mps)
        )
        return np.array([delta_altitude, delta_heading, delta_velocity])


class L1maneuverModel(TrackModel):
    """Randomly turn left or right; randomly shoot cannon."""

    def __init__(self, config):
        super().__init__(config)
        self.turn_interval = np.random.randint(20, 60)
        self.step_interval = self.turn_interval / 0.1
        self.target_heading_list = [np.pi/3, np.pi, -np.pi/3]
        self.target_altitude = 6000
        self.target_velocity = 250

    def reset(self, env):
        self._step = 0
        self._i = 0
        self._rnd_head = 0
        self._rnd_vel = 0
        self.turn_interval = np.random.randint(20, 60)
        self.step_interval = self.turn_interval / 0.1
        self.init_heading = np.random.choice([0, 2*np.pi])
        super().reset(env)

    def set_delta_value(self, agent):
        cur_heading = agent.get_property_value(c.attitude_heading_true_rad)
        if self._step % self.step_interval == 0:
            self._i = (self._i + 1) % 3
            self._rnd_head = np.random.randint(-900, 900)
            self._rnd_vel = np.random.randint(-30, 30)

        if self.init_heading:
            delta_heading = self.init_heading - self.target_heading_list[self._i] - cur_heading
        else:
            delta_heading = self.target_heading_list[self._i] - cur_heading
        delta_altitude = self.target_altitude + self._rnd_head - agent.get_property_value(c.position_h_sl_m)
        delta_velocity = self.target_velocity + self._rnd_vel - agent.get_property_value(c.velocities_u_mps)
        self._step += 1
        return np.array([delta_altitude, delta_heading, delta_velocity])

    def get_action(self, action=None, agent=None, env=None):
        if (agent.cannon.ready
                and self._step % self.step_interval in range(100, 130)
                and random.choices([True, False], weights=[0.4, 0.6], k=1)[0]):
            agent.cannon.launch(parent=agent, env=env)
        return super().get_action(action, agent, env)

    def step(self, env, agent):
        if agent.cannon.done:
            agent.cannon.revert_cannon()


class L2maneuverModel(TrackModel):
    """Tracks and shoots (rather) precisely with cannon."""

    def reset(self, env):
        self.rnd_alt = random.uniform(-250, 250)
        self.rnd_head = random.uniform(-25, 25)
        self.rnd_vel = random.uniform(-100, 100)
        super().reset(env)

    def get_action(self, action=None, agent=None, env=None):
        if agent.cannon.ready:
            self.check_cannon(agent, env)

        delta_value = self.set_delta_value(agent)
        delta_value[0] = delta_value[0] + self.rnd_alt
        delta_value[1] = delta_value[1] + np.deg2rad(self.rnd_head)
        delta_value[2] = delta_value[2] + self.rnd_vel

        observation = self.get_obs(None, agent, kwargs={"delta": delta_value})
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        return self.shift_action(_action.detach().cpu().numpy().squeeze())

    def check_cannon(self, agent, env):
        ata, dist = agent.cannon.get_distance_angle(agent=agent, target=agent.opponents[0])
        if (abs(ata) < agent.cannon.omega_wez+5) and (dist < agent.cannon.Rmax_wez+500):
            agent.cannon.launch(parent=agent, env=env)

    def step(self, env, agent):
        if agent.cannon.done:
            agent.cannon.revert_cannon()


class SpmaneuverModel(BaseMission):
    """
    Self-play opponent.
    """

    def load_rewards(self):
        self.rewards = []

    def load_model(self):
        policy_name = getattr(self.config, "sp_policy_name", "cannon_L2_f16")
        from algorithm.networks import build_actor_critic, act_with_policies
        self._act_with_policies = act_with_policies

        self.actor_args = Args()
        self.actor_args.use_ReLU = True
        self.actor_args.avail_actions = False
        self.actor_args.torch_compile = False

        device = torch.device("cpu")
        action_space = ("multi_discrete", list(self.action_space.nvec))
        self.actors, _ = build_actor_critic(
            self.actor_args,
            obs_dim=self.obs_len,
            action_space=action_space,
            state_dim=self.obs_len,
            num_agents=1,
            device=device,
        )
        ckpt = load_pretrained_actors(policy_name, device=device)
        self.actors.load_state_dict(ckpt["actors_state_dict"])
        self.actors.eval()

    def step(self, env, agent):
        if agent.cannon.done:
            agent.cannon.revert_cannon()

    def get_action(self, action=None, agent=None, env=None):
        obs = self.get_obs(env=env, agent=agent)
        obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            actions, _, _, _ = self._act_with_policies(
                self.actor_args, self.actors, obs_t,
                rnn_hxs=None, masks=None, deterministic=True,
            )
        act = actions[0, 0].cpu().numpy()
        if bool(act[4]):
            self._cannon(agent, env)
        return super().get_action(action=act, agent=agent, env=env)

    def _cannon(self, agent, env):
        if agent.alive and agent.cannon.ready:
            agent.cannon.launch(parent=agent, env=env)


class StrategyModel(BaseMission):
    """
    L4 opponent that picks low-level pre-trained policies with a mixed strategy.
    """

    # Map the high-level discrete choice to the task whose policy we run.
    HL_ACTION_MAPPING = {0: 'cannon', 1: 'track', 2: 'escape'}
    # In L4 training only the f16 model is used (better performance than A4).
    POLICY_PATHS = {'cannon': 'cannon_L3_f16', 'track': 'track_L3_f16', 'escape': 'escape_L1_f16'}

    def load_model(self):
        from algorithm.networks import build_actor_critic, act_with_policies
        self._act_with_policies = act_with_policies

        # Single shared config for all actors.
        self.actor_args = Args()
        self.actor_args.use_ReLU = True
        self.actor_args.avail_actions = False
        self.actor_args.torch_compile = False

        device = torch.device("cpu")
        action_space = ("multi_discrete", list(self.action_space.nvec))
        self.policies = {}
        for task, policy_name in self.POLICY_PATHS.items():
            actors, _ = build_actor_critic(
                self.actor_args,
                obs_dim=self.obs_len,
                action_space=action_space,
                state_dim=self.obs_len,
                num_agents=1,
                device=device,
            )
            ckpt = load_pretrained_actors(policy_name, device=device)
            actors.load_state_dict(ckpt["actors_state_dict"])
            actors.eval()
            self.policies[task] = actors

    def reset(self, env):
        self.policy_set_step = 0
        self.policy_to_use = {aid: None for aid in env.agents.keys()}
        return super().reset(env)

    def step(self, env, agent):
        if agent.cannon.done:
            agent.cannon.revert_cannon()

    def load_rewards(self):
        self.rewards = []

    #load_act_space / load_obs_space are inherited from BaseMission.

    def get_action(self, action, agent, env):
        # Switch strategy every (env.horizon / 3) steps (horizon 600 -> every 200 steps).
        if self.policy_set_step == 0 or abs(env.steps - self.policy_set_step) > (env.horizon / 3):
            self.set_policy(action, agent, env)
            self.policy_set_step = env.steps

        assert self.policy_to_use[agent.aid] is not None, f"No policy was selected for agent {agent.aid}"

        obs = super().get_obs(env=env, agent=agent)
        act = self.get_policy_action(obs=obs, agent=agent)
        if bool(act[4]):
            self._cannon(agent, env)
        return super().get_action(action=act, agent=agent, env=env)

    def set_policy(self, action, agent, env):
        act = random.choices([0, 1, 2], weights=[0.4, 0.3, 0.3], k=1)[0]
        self.policy_to_use[agent.aid] = self.policies[self.HL_ACTION_MAPPING[act]]

    def get_policy_action(self, obs, agent):
        actors = self.policy_to_use[agent.aid]
        obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            actions, _, _, _ = self._act_with_policies(
                self.actor_args, actors, obs_t, rnn_hxs=None, masks=None, deterministic=True,
            )
        return actions[0, 0].cpu().numpy()

    def _cannon(self, agent, env):
        if agent.alive and agent.cannon.ready:
            agent.cannon.launch(parent=agent, env=env)


def build_low_level_actors(obs_len, action_nvec, policy_paths, device=torch.device("cpu")):
    """Instantiate and load the low-level actor modules used by hierarchical commanders.

    policy_paths: {task: run_dir_name}. Returns ({task: actors}, actor_args) where each
    `actors` is a ModuleDict ready for algorithm.networks.act_with_policies. Mirrors how
    StrategyModel.load_model builds its policies (same Args flags), so the high-level
    commander and the L4 opponent share one instantiation path.
    """
    from algorithm.networks import build_actor_critic

    actor_args = Args()
    actor_args.use_ReLU = True
    actor_args.avail_actions = False
    actor_args.torch_compile = False

    action_space = ("multi_discrete", list(action_nvec))
    policies = {}
    for task, policy_name in policy_paths.items():
        actors, _ = build_actor_critic(
            actor_args, obs_dim=obs_len, action_space=action_space,
            state_dim=obs_len, num_agents=1, device=device,
        )
        ckpt = load_pretrained_actors(policy_name, device=device)
        actors.load_state_dict(ckpt["actors_state_dict"])
        actors.eval()
        policies[task] = actors
    return policies, actor_args


class HierModel(BaseMission):
    """High-level commander opponent that follows a mixed strategy over L4 policies."""

    HL_ACTION_MAPPING = {0: 'cannon', 1: 'track', 2: 'escape'}
    # Low-level pre-trained policies, keyed by task -> runs/<dir>.
    POLICY_PATHS = {'cannon': 'cannon_L4_f16', 'track': 'track_L4_f16', 'escape': 'escape_L2_f16'}

    def __init__(self, config):
        super().__init__(config)
        self.opp_hier_strategy = getattr(config, 'opp_hier_strategy', "mixed")

    def load_model(self):
        from algorithm.networks import act_with_policies
        self._act_with_policies = act_with_policies
        self.policies, self.actor_args = build_low_level_actors(
            self.obs_len, self.action_space.nvec, self.POLICY_PATHS,
        )

    def reset(self, env):
        self.policy_to_use = {aid: None for aid in env.agents.keys()}
        return super().reset(env)

    def step(self, env, agent):
        if agent.cannon.done:
            agent.cannon.revert_cannon()

    def load_rewards(self):
        self.rewards = []

    # load_act_space / load_obs_space are inherited from BaseMission: the opponent
    # runs the low-level policies, so it uses the low-level obs/action layout.

    def set_policy(self, action, agent, env):
        if self.opp_hier_strategy == "mixed":
            act = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2], k=1)[0]
        elif self.opp_hier_strategy == "aggressive":
            act = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1], k=1)[0]
        else:
            act = np.random.randint(0, 3)
        self.policy_to_use[agent.aid] = self.policies[self.HL_ACTION_MAPPING[act]]

    def run_action(self, agent, env):
        ll_obs = self.get_obs(env=env, agent=agent)
        ll_action = self.get_ll_action(ll_obs, agent)
        if bool(ll_action[4]):
            self._cannon(agent, env)
        ll_action = self.get_action(action=ll_action, agent=agent, env=env)
        agent.set_property_values(self.action_var, ll_action)
        for _ in range(env.sim_steps):
            agent.run()

    def get_ll_action(self, obs, agent):
        actors = self.policy_to_use[agent.aid]
        assert actors is not None, f"No low-level policy selected for opponent {agent.aid}"
        obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            actions, _, _, _ = self._act_with_policies(
                self.actor_args, actors, obs_t, rnn_hxs=None, masks=None, deterministic=True,
            )
        return actions[0, 0].cpu().numpy()

    def _cannon(self, agent, env):
        if agent.alive and agent.cannon.ready:
            agent.cannon.launch(parent=agent, env=env)
