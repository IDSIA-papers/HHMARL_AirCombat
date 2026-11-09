import numpy as np
import torch
from multiprocessing import Pipe, Process
from env.env import AircombatMARL

def validate_decay_schedule(value, name):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be of the form [[t0, l0], [t1, l1]]")
    try:
        (t0, l0), (t1, l1) = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be of the form [[t0, l0], [t1, l1]]") from exc
    t0 = int(t0)
    t1 = int(t1)
    l0 = float(l0)
    l1 = float(l1)
    if t1 <= t0:
        raise ValueError(f"{name} requires t1 > t0")
    return ((t0, l0), (t1, l1))

def get_decayed_value(step, base_value, decay_schedule):
    if decay_schedule is None:
        return base_value
    (t0, l0), (t1, l1) = decay_schedule
    if step <= t0:
        return l0
    if step >= t1:
        return l1
    alpha = (step - t0) / (t1 - t0)
    return l0 + alpha * (l1 - l0)

class Logger:
    def __init__(self, log_dir, args):
        import os
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_every_step = args.log_every_step
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    def log_episode_stats(self, ep_rewards, ep_lengths, ep_stats, num_episodes, step):
        """Log rollout episode statistics if enough episodes have accumulated.

        Returns True if the threshold was reached and stats were logged, False otherwise.
        """
        if len(ep_rewards) < self.log_every_step:
            return False
        mean_ep_reward = np.mean(ep_rewards)
        mean_ep_length = np.mean(ep_lengths)
        step_rewards = [
            reward / length if length > 0 else 0.0
            for reward, length in zip(ep_rewards, ep_lengths)
        ]
        mean_step_reward = np.mean(step_rewards) if step_rewards else 0.0
        weighted_step_reward = mean_ep_reward / mean_ep_length if mean_ep_length > 0 else 0.0

        self.writer.add_scalar("rollout/mean_ep_reward", mean_ep_reward, step)
        self.writer.add_scalar("rollout/mean_ep_length", mean_ep_length, step)
        self.writer.add_scalar("rollout/mean_step_reward", mean_step_reward, step)
        self.writer.add_scalar("rollout/weighted_step_reward", weighted_step_reward, step)
        self.writer.add_scalar("rollout/num_episodes", num_episodes, step)
        if ep_stats:
            for key in ("opp_killed", "opp_crashed", "agent_killed", "agent_crashed"):
                self.writer.add_scalar(
                    f"rollout/{key}",
                    np.mean([1 if info.get(key, False) else 0 for info in ep_stats]),
                    step,
                )
        return True

    def log_train_stats(self, metrics, training_step, update_step, actor_lr, critic_lr, entropy_coef, step):
        """Log training statistics after each policy update."""
        self.writer.add_scalar("train/critic_loss", np.mean(metrics.critic_losses), step)
        self.writer.add_scalar("train/actor_loss", np.mean(metrics.actor_losses), step)
        self.writer.add_scalar("train/entropy", np.mean(metrics.entropy_bonuses), step)
        self.writer.add_scalar("train/kl_divergence", np.mean(metrics.kl_divergences), step)
        self.writer.add_scalar("train/clipped_ratios", np.mean(metrics.clipped_ratios), step)
        self.writer.add_scalar("train/actor_gradients", np.mean(metrics.actor_gradients), step)
        self.writer.add_scalar("train/critic_gradients", np.mean(metrics.critic_gradients), step)
        self.writer.add_scalar("train/num_gradient_steps", training_step, step)
        self.writer.add_scalar("train/num_updates", update_step, step)
        self.writer.add_scalar("train/current_actor_lr", actor_lr, step)
        self.writer.add_scalar("train/current_critic_lr", critic_lr, step)
        self.writer.add_scalar("train/current_entropy_coef", entropy_coef, step)

    def save_checkpoint(self, learner, step: int):
        import os
        import torch
        path = os.path.join(self.log_dir, f"checkpoint.pt")
        value_norm_state = (
            learner.value_normalizer.state_dict()
            if learner.value_normalizer is not None else None
        )
        torch.save(
            {
                "step": step,
                "actors_state_dict": learner.actors.state_dict(),
                "critic_state_dict": learner.critic.state_dict(),
                "actor_optimizer_state_dict": learner.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": learner.critic_optimizer.state_dict(),
                "value_normalizer_state_dict": value_norm_state,
            },
            path,
        )

    def close(self):
        self.writer.close()

class CloudpickleWrapper:
    def __init__(self, env):
        self.env = env

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.env)

    def __setstate__(self, env):
        import pickle

        self.env = pickle.loads(env)

def _env_worker(
    worker,
    task,
    env_kwargs,
    use_avail=True,
    centralized_critic=True,
):
    env = AircombatMARL(task=task, **env_kwargs)
    agent_keys = env.agent_keys

    def _obs_to_array(obs_dict):
        return np.array([obs_dict[k] for k in agent_keys], dtype=np.float32)

    def _state(obs):
        # Centralized critic: one state per env (state_dim,). Per-agent critic: the own obs of each agent (A, state_dim).
        return env.get_state() if centralized_critic else obs

    def _avail_to_array(avail_dict):
        return np.array([avail_dict[k] for k in agent_keys], dtype=np.bool_)

    def _reward_to_array(reward_dict):
        return np.array([reward_dict[k] for k in agent_keys], dtype=np.float32)

    action_space = env.get_action_space()
    space_type = action_space[0]

    def _actions_to_dict(actions_arr):
        arr = np.asarray(actions_arr)
        # arr is (A, K). For discrete K=1 pass scalar ints; otherwise pass per-agent arrays.
        if space_type == "discrete":
            flat = arr.reshape(len(agent_keys), -1)[:, 0]
            return {k: int(flat[i]) for i, k in enumerate(agent_keys)}
        if space_type == "multi_discrete":
            return {k: arr[i].astype(np.int64).reshape(-1) for i, k in enumerate(agent_keys)}
        # continuous
        return {k: arr[i].astype(np.float32).reshape(-1) for i, k in enumerate(agent_keys)}

    while True:
        command, content = worker.recv()
        if command == "reset":
            obs_dict, _ = env.reset()
            obs = _obs_to_array(obs_dict)
            worker.send({
                "obs": obs,
                "avail_actions": _avail_to_array(env.get_avail_actions()) if use_avail else None,
                "state": _state(obs),
            })
        elif command == "get_env_info":
            worker.send({
                "obs_size": env.get_obs_size(),
                "action_size": env.get_action_size(),
                "action_space": env.get_action_space(),
                "n_agents": env.n_agents,
                "state_size": env.get_state_size(),
            })
        elif command == "step":
            actions_dict = _actions_to_dict(content)
            obs_dict, reward_dict, done_dict, truncated_dict, infos = env.step(actions_dict)
            done = done_dict["all"]
            truncated = truncated_dict["all"]
            agent_dones = np.array([done_dict[k] for k in agent_keys], dtype=np.bool_)
            next_obs = _obs_to_array(obs_dict)

            # next_state: skip only for true terminal (done without truncation) — env resets
            # anyway and the value is never used for GAE bootstrap or current-state update.
            next_state = None if (done and not truncated) else _state(next_obs)
            avail_actions = (
                None if (done or truncated or not use_avail)
                else _avail_to_array(env.get_avail_actions())
            )
            worker.send({
                "next_obs": next_obs,
                "reward": _reward_to_array(reward_dict),
                "done": done,
                "truncated": truncated,
                "agent_dones": agent_dones,
                "infos": infos,
                "avail_actions": avail_actions,
                "next_state": next_state,
            })
        elif command == "close":
            env.close()
            worker.close()
            break

class Workers:
    """Manages parallel environment worker processes."""

    def __init__(self, args, kwargs):
        
        pipes = [Pipe() for _ in range(args.num_envs)]
        self._main_ends, env_ends = zip(*pipes)
        self._processes = [
            Process(
                target=_env_worker,
                args=(
                    env_ends[i],
                    args.task,
                    {
                        **kwargs,
                        "seed": args.seed + i * 1000,
                    },
                    getattr(args, "avail_actions", True),
                    args.centralized_critic,
                ),
            )
            for i in range(args.num_envs)
        ]
        for process in self._processes:
            process.daemon = True
            process.start()

    def __len__(self):
        return len(self._main_ends)

    def __getitem__(self, idx):
        return self._main_ends[idx]

    def __iter__(self):
        return iter(self._main_ends)

    def close(self):
        for worker in self._main_ends:
            worker.send(("close", None))
        for process in self._processes:
            process.join()
