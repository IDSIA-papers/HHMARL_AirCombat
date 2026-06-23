import logging
import math
from abc import ABC, abstractmethod
from env.catalog import Catalog as c


class BaseTerminationCondition(ABC):
    """Base TerminationCondition class."""

    def __init__(self, config):
        self.log = logging.getLogger(__name__)
        self.config = config

    @abstractmethod
    def get_termination(self, task, env, agent_id, info={}):
        raise NotImplementedError


class ExtremeState(BaseTerminationCondition):
    """End episode on extreme dynamic state or out-of-boundary."""

    def __init__(self, config):
        super().__init__(config)
        self.map_size = getattr(self.config, 'map_size', 50)

    def get_termination(self, task, env, agent_id, info={}):
        agent = env.agents[agent_id]
        done = (
            bool(agent.get_property_value(c.detect_extreme_state))
            or bool(agent.get_distance_to_origin_2d() > (self.map_size/2)*1000)
        )
        if done:
            agent.crash()
            self.log.info(f'{agent_id} is on an extreme state! Total Steps={env.steps}')
        return done, False, info


class Overload(BaseTerminationCondition):
    """End episode if acceleration limits are exceeded."""

    def __init__(self, config):
        super().__init__(config)
        self.acceleration_limit_x = getattr(config, 'acceleration_limit_x', 10.0)
        self.acceleration_limit_y = getattr(config, 'acceleration_limit_y', 10.0)
        self.acceleration_limit_z = getattr(config, 'acceleration_limit_z', 10.0)

    def get_termination(self, task, env, agent_id, info={}):
        agent = env.agents[agent_id]
        done = self._judge_overload(agent)
        if done:
            agent.crash()
            self.log.info(f'{agent_id} acceleration is too high! Total Steps={env.steps}')
        return done, False, info

    def _judge_overload(self, agent):
        if agent.get_property_value(c.simulation_sim_time_sec) <= 10:
            return False
        return (
            math.fabs(agent.get_property_value(c.accelerations_n_pilot_x_norm)) > self.acceleration_limit_x
            or math.fabs(agent.get_property_value(c.accelerations_n_pilot_y_norm)) > self.acceleration_limit_y
            or math.fabs(agent.get_property_value(c.accelerations_n_pilot_z_norm) + 1) > self.acceleration_limit_z
        )


class LowAltitude(BaseTerminationCondition):
    """End episode if altitude is outside [altitude_limit, altitude_limit_max]."""

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 2500)
        self.altitude_limit_max = getattr(config, 'altitude_limit_max', 10000)

    def get_termination(self, task, env, agent_id, info={}):
        height = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        done = height <= self.altitude_limit or height > self.altitude_limit_max
        if done:
            env.agents[agent_id].crash()
            self.log.info(f'{agent_id} altitude is too low. Total Steps={env.steps}')
        return done, False, info


class DeadAgent(BaseTerminationCondition):
    """End episode if the current agent has been shot down or crashed."""

    def get_termination(self, task, env, agent_id, info={}):
        if env.agents[agent_id].shotdown:
            self.log.info(f'{agent_id} has been shot down! Total Steps={env.steps}')
            return True, False, info
        if env.agents[agent_id].crashed:
            self.log.info(f'{agent_id} has crashed! Total Steps={env.steps}')
            return True, False, info
        return False, False, info