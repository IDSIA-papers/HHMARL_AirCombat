import logging
import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from env.catalog import Catalog as c
from env.utils import get_AO_TA_R, get2d_AO_TA_R, LLA2NEU


class BaseRewardFunction(ABC):
    """Base RewardFunction class."""

    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger(__name__)
        self.reward_scale = getattr(self.config, f'{self.__class__.__name__}_scale', 1.0)
        self.is_potential = getattr(self.config, f'{self.__class__.__name__}_potential', False)
        self.pre_rewards = defaultdict(float)

    def reset(self, task, env):
        if self.is_potential:
            self.pre_rewards.clear()
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_learning:
                    self.pre_rewards[agent_id] = self.get_reward(task, env, agent_id)

    @abstractmethod
    def get_reward(self, task, env, agent_id, action=None):
        raise NotImplementedError

    def _process(self, new_reward, agent_id, render_items=()):
        reward = new_reward * self.reward_scale
        if self.is_potential:
            reward, self.pre_rewards[agent_id] = reward - self.pre_rewards[agent_id], reward
        return reward

    def get_orientation_metrics(self, env, agent_id, deg=False, two_d=False):
        state_vars = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.velocities_v_north_mps,
            c.velocities_v_east_mps,
            c.velocities_v_up_mps,
            c.attitude_roll_rad,
            c.velocities_p_rad_sec,
            c.velocities_q_rad_sec,
        ]
        ego_state = np.array(env.agents[agent_id].get_property_values(state_vars))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[3:6])])

        opp_state = np.array(env.agents[agent_id].opponents[0].get_property_values(state_vars))
        opp_cur_ned = LLA2NEU(*opp_state[:3], env.center_lon, env.center_lat, env.center_alt)
        opp_feature = np.array([*opp_cur_ned, *(opp_state[3:6])])
        if two_d:
            a_a, a_t, dist = get2d_AO_TA_R(ego_feature, opp_feature, return_side=False)
        else:
            a_a, a_t, dist = get_AO_TA_R(ego_feature, opp_feature, return_side=False)
        if deg:
            a_a, a_t, a_r, a_rr, a_pr = np.rad2deg(a_a), np.rad2deg(a_t), np.rad2deg(ego_state[6]), np.rad2deg(ego_state[7]), np.rad2deg(ego_state[8])
        else:
            a_r, a_rr, a_pr = ego_state[6], ego_state[7], ego_state[8]
        return a_a, a_t, a_r, a_rr, a_pr, dist

    def get_delta_values(self, env, agent_id, in_km=True):
        state_vars = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.velocities_v_north_mps,
            c.velocities_v_east_mps,
            c.velocities_v_up_mps,
        ]
        ego_state = np.array(env.agents[agent_id].get_property_values(state_vars))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[3:6])])

        opp_state = np.array(env.agents[agent_id].opponents[0].get_property_values(state_vars))
        opp_cur_ned = LLA2NEU(*opp_state[:3], env.center_lon, env.center_lat, env.center_alt)
        opp_feature = np.array([*opp_cur_ned, *(opp_state[3:6])])

        ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
        enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = opp_feature
        delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
        delta_vx, delta_vy, delta_vz = enm_vx - ego_vx, enm_vy - ego_vy, enm_vz - ego_vz
        if in_km:
            return abs(delta_x/1000), abs(delta_y/1000), abs(delta_z/1000), abs(delta_vx), abs(delta_vy), abs(delta_vz)
        return abs(delta_x), abs(delta_y), abs(delta_z), abs(delta_vx), abs(delta_vy), abs(delta_vz)


class EventDrivenReward(BaseRewardFunction):
    """Penalty when shot down or crashed."""

    def get_reward(self, task, env, agent_id, action=None):
        reward = 0
        if env.agents[agent_id].shotdown:
            reward -= 5
        elif env.agents[agent_id].crashed:
            reward -= 5
        return self._process(reward, agent_id)


class ShootReward(BaseRewardFunction):
    """Reward for destroying an opponent by cannon."""

    def __init__(self, config):
        super().__init__(config)
        self.orientation_kill = getattr(self.config, f'{self.__class__.__name__}_orientation_kill', True)

    def get_reward(self, task, env, agent_id, action=None):
        reward = 0
        if env.agents[agent_id].cannon and env.agents[agent_id].cannon.success:
            if self.orientation_kill:
                _, a_t, _, _, _, _ = self.get_orientation_metrics(env, agent_id, deg=False)
                scale, shift = task.shift_range(-1, 1, 5, 10)
                reward += float(np.cos(a_t) * scale + shift)
            else:
                reward += 5
        return self._process(reward, agent_id)


class ShootPenaltyReward(BaseRewardFunction):
    """Penalty for firing cannon (constant or posture-aware)."""

    def __init__(self, config):
        super().__init__(config)
        self.posture_punish = getattr(self.config, f'{self.__class__.__name__}_posture_punish', False)

    def get_reward(self, task, env, aid, action=None):
        a_a, _, _, _, _, dist = self.get_orientation_metrics(env, aid, deg=True)
        reward = 0
        if self.posture_punish:
            if a_a <= 10 and dist < 3600:
                if not env.agents[aid].cannon.alive:
                    reward -= 0.1
            elif a_a > 90 or dist > 15000:
                if task.shot_fired[aid]:
                    reward -= 0.1
        else:
            if task.shot_fired[aid]:
                reward -= 0.1
        return self._process(reward, aid)


class HeadingReward(BaseRewardFunction):
    """Geometric reward over heading / orientation / roll / pitch."""

    def __init__(self, config):
        super().__init__(config)
        self.track = getattr(self.config, f'{self.__class__.__name__}_track', True)
        self.opposite = getattr(self.config, f'{self.__class__.__name__}_opposite', False)

    def get_reward(self, task, env, agent_id, action=None):
        a_a, a_t, a_r, a_rr, a_pr, dist = self.get_orientation_metrics(env, agent_id, deg=True)
        a_a_opp, _, _, _, _, _ = self.get_orientation_metrics(env, env.agents[agent_id].opponents[0].aid, deg=True)
        delta_x, delta_y, delta_z, delta_vx, delta_vy, delta_vz = self.get_delta_values(env, agent_id, in_km=True)

        if self.track:
            head_scale = 10
            head_a = math.exp(-((a_a/head_scale)**2))
            head_t = math.exp(-((a_t/head_scale)**2))

            head_a_opp = math.exp(-((a_a_opp/head_scale)**2))

            def delta_loc(x):
                dist_scale = 4
                dist_offset = 2
                return 1 if x < dist_offset else math.exp(-(((x-dist_offset)/dist_scale)**4))

            def delta_vel(x):
                vel_scale = 100
                vel_offset = 30
                return 1 if x < vel_offset else math.exp(-(((x-vel_offset)/vel_scale)**2))

            h = 0.3 * head_a + 0.7 * head_t
            d = 0.333 * delta_loc(delta_x) + 0.333 * delta_loc(delta_y) + 0.333 * delta_loc(delta_z)
            v = 0.5 * delta_vel(delta_vx) + 0.5 * delta_vel(delta_vy)

            roll_scale = 35
            roll_rr_scale = 25
            pitch_rr_scale = 15
            roll_r = math.exp(-((a_r/roll_scale)**4))
            roll_rr = math.exp(-((a_rr/roll_rr_scale)**2))
            pitch_rr = math.exp(-((a_pr/pitch_rr_scale)**2))
            
            # reward = h*(d + v + 0.2*(roll_r-0.5)*2 + 0.6*(roll_rr-0.5)*2 + 0.2*(pitch_rr-0.5)*2)
            reward = h*(d + v + 0.2*(roll_r-0.5)*2 + 0.6*(roll_rr-0.5)*2 + 0.2*(pitch_rr-0.5)*2) - 0.1*head_a_opp*head_a*(0.5 * delta_loc(delta_x) + 0.5 * delta_loc(delta_y))

        elif self.opposite:
            head_scale = 40
            head_a = 1-math.exp(-((a_a/head_scale)**2))
            head_t = math.exp(-((a_t/head_scale)**2))
            h = 0.5*head_a+0.5*head_t

            roll_scale = 60
            roll_rr_scale = 25
            pitch_rr_scale = 20
            roll_r = math.exp(-((a_r/roll_scale)**2)) - 1.0
            roll_rr = math.exp(-((a_rr/roll_rr_scale)**4)) - 0.5
            pitch_rr = math.exp(-((a_pr/pitch_rr_scale)**4)) - 0.5
            reward = h*(roll_rr + pitch_rr + 0.5*roll_r)

        else:
            head_scale = 30
            head_a = math.exp(-((a_a/head_scale)**2))
            head_t = math.exp(-((a_t/(head_scale))**2))
            h = 0.7 * head_a + 0.3 * head_t

            roll_scale = 60
            roll_rr_scale = 25
            pitch_rr_scale = 20
            roll_r = math.exp(-((a_r/roll_scale)**2)) -1.0
            roll_rr = math.exp(-((a_rr/roll_rr_scale)**4)) - 0.5
            pitch_rr = math.exp(-((a_pr/pitch_rr_scale)**4)) - 0.5
            reward = h*(roll_rr + pitch_rr + 0.5*roll_r)

        return self._process(reward, agent_id)


class DistanceReward(BaseRewardFunction):
    """Reward increasing with horizontal distance from opponent."""

    def __init__(self, config):
        super().__init__(config)
        self.map_size = getattr(self.config, 'map_size', 50)
        self.half_map = self.map_size / 2

    def get_reward(self, task, env, agent_id, action=None):
        delta_x, delta_y, _, _, _, _ = self.get_delta_values(env, agent_id, in_km=True)
        scale, shift = task.shift_range(0, 50, -5, 10)
        x = delta_x * scale + shift
        y = delta_y * scale + shift
        reward = 0.5*x + 0.5*y
        return self._process(reward, agent_id)


class FleeReward(BaseRewardFunction):
    """Motivation to flee: stay inside boundaries, keep altitude, gain speed."""

    def __init__(self, config):
        super().__init__(config)
        self.map_size = getattr(self.config, 'map_size', 50)
        self.half_map = self.map_size / 2

    def get_reward(self, task, env, agent_id, action=None):
        def boundary_pen(d):
            margin = self.half_map - d
            bnd_offset = 13
            bnd_scale = 3
            if margin > bnd_offset:
                return 0
            return -(1 - math.exp(-(((bnd_offset - margin)/bnd_scale)**2)))

        dist_lon = env.agents[agent_id].get_distance_to_origin_lon() / 1000
        dist_lat = env.agents[agent_id].get_distance_to_origin_lat() / 1000
        b = 0.5*boundary_pen(dist_lat) + 0.5*boundary_pen(dist_lon)

        def altitude_pen(alt):
            alt_low = getattr(self.config, 'altitude_limit', 2500) / 1000
            alt_high = getattr(self.config, 'altitude_limit_max', 10000) / 1000
            alt_offset = 2
            alt_scale = 1.5
            low_margin = alt - alt_low
            high_margin = alt_high - alt
            low_pen = 0 if low_margin > alt_offset else -(1 - math.exp(-(((alt_offset - low_margin)/alt_scale)**2)))
            high_pen = 0 if high_margin > alt_offset else -(1 - math.exp(-(((alt_offset - high_margin)/alt_scale)**2)))
            return low_pen + high_pen

        a = altitude_pen(env.agents[agent_id].get_altitude() / 1000)

        def speed_rew(v):
            v_scale = 1200
            v_offset = 800
            if v < v_offset:
                return 0
            return 1 - math.exp(-(((v - v_offset)/v_scale)**2))

        v = speed_rew(env.agents[agent_id].get_norm_velocity_2d() * 3.6)

        reward = a + b + v
        return self._process(reward, agent_id)