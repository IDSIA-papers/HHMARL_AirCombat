from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from algorithm.networks import build_actor_critic, act_with_policies
from algorithm.valuenorm import ValueNorm


class Batch(NamedTuple):
    batch_obs: torch.Tensor
    batch_actions: torch.Tensor
    batch_log_probs: torch.Tensor
    batch_reward: torch.Tensor             # (T, E, A) per-agent rewards
    batch_states: torch.Tensor
    batch_avail_actions: torch.Tensor
    batch_done: torch.Tensor        # (T, E) float — 1 only for TRUE terminal (not time-limit)
    batch_truncated: torch.Tensor   # (T, E) float — 1 for time-limit truncation
    batch_values: torch.Tensor               # (T, E, A) always
    batch_terminal_values: torch.Tensor      # (T, E, A) — V(next_state) at truncated steps (0 elsewhere)
    batch_time_mask: torch.Tensor           # (T, E) bool
    batch_active_mask: torch.Tensor         # (T, E, A) float
    rnn_states_actor: Optional[torch.Tensor]   # (T, E, A, L, H) or None
    rnn_states_critic: Optional[torch.Tensor]  # (T, E, A, L, H) — per-agent for both centralized and IPPO
    rnn_masks: Optional[torch.Tensor]          # (T, E) float or None


class RolloutBuffer:
    def __init__(
        self,
        num_steps,
        num_envs,
        num_agents,
        obs_space,
        state_space,
        action_space,
        centralized_critic=True,
        device="cpu",
        rnn=False,
        actor_hidden_dim=128,
        critic_hidden_dim=128,
        rnn_num_layers=1,
        use_avail=None,
    ):
        """
        action_space: tuple of the form
          ("discrete", n)               -> action_width K=1, dtype long, avail mask stored
          ("multi_discrete", [n1,...])  -> action_width K=len(nvec), dtype long, no mask
          ("continuous", dim)           -> action_width K=dim, dtype float32, no mask
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.centralized_critic = centralized_critic
        self.device = device
        self.rnn = rnn

        space_type = action_space[0]
        if space_type == "discrete":
            K = 1
            action_dtype = torch.long
        elif space_type == "multi_discrete":
            K = len(action_space[1])
            action_dtype = torch.long
        elif space_type == "continuous":
            K = int(action_space[1])
            action_dtype = torch.float32
        else:
            raise ValueError(f"Unknown action space type: {space_type}")
        # use_avail: caller can override; default is True only for discrete spaces
        self.use_avail = use_avail if use_avail is not None else (space_type == "discrete")
        mask_width = int(action_space[1]) if self.use_avail else 0
        self.action_width = K
        self.action_dtype = action_dtype

        self.obs = torch.zeros(
            (num_steps, num_envs, num_agents, obs_space),
            dtype=torch.float32,
            device=device,
        )
        self.actions = torch.zeros(
            (num_steps, num_envs, num_agents, K), dtype=action_dtype, device=device
        )
        self.log_probs = torch.zeros(
            (num_steps, num_envs, num_agents), dtype=torch.float32, device=device
        )
        self.reward = torch.zeros(
            (num_steps, num_envs, num_agents), dtype=torch.float32, device=device
        )
        # Centralized: one shared state per env (T, E, state_D).
        # IPPO with state critic: one per-agent state per env (T, E, A, state_D).
        states_shape = (
            (num_steps, num_envs, state_space)
            if centralized_critic
            else (num_steps, num_envs, num_agents, state_space)
        )
        self.states = torch.zeros(states_shape, dtype=torch.float32, device=device)
        # Avail-actions buffer is a placeholder when use_avail=False (mask_width=0).
        self.avail_actions = torch.zeros(
            (num_steps, num_envs, num_agents, mask_width),
            dtype=torch.bool,
            device=device,
        )
        self.done = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.truncated = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros(
            (num_steps, num_envs, num_agents), dtype=torch.float32, device=device
        )
        # Raw (pre-denorm) critic output V(next_state) at truncated transitions;
        # used as GAE bootstrap so the time-limit's final state is not replaced by V(reset).
        self.terminal_values = torch.zeros(
            (num_steps, num_envs, num_agents), dtype=torch.float32, device=device
        )
        self.time_mask = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.active_masks = torch.ones((num_steps, num_envs, num_agents), dtype=torch.float32, device=device)
        if rnn:
            L, Ha, Hc = rnn_num_layers, actor_hidden_dim, critic_hidden_dim
            self.rnn_states_actor = torch.zeros(
                (num_steps, num_envs, num_agents, L, Ha), dtype=torch.float32, device=device
            )
            self.rnn_states_critic = torch.zeros(
                (num_steps, num_envs, num_agents, L, Hc), dtype=torch.float32, device=device
            )
            self.rnn_masks = torch.ones((num_steps, num_envs), dtype=torch.float32, device=device)

    def reset(self):
        self.time_mask.zero_()
        # Zero (not fill_(1.0)): positions never written by add_step stay masked out of
        # actor/critic loss. add_step writes the correct alive/dead mask for valid positions.
        self.active_masks.zero_()
        # terminal_values is sparse (only truncated steps get written); zero so the
        # torch.where override in GAE only kicks in at genuinely truncated positions.
        self.terminal_values.zero_()
        if self.rnn:
            self.rnn_masks.fill_(1.0)

    def add_step(
        self,
        step_idx: int,
        active_envs: int,
        obs: torch.Tensor,              # (E, A, D_obs)  GPU
        actions: torch.Tensor,          # (E, A, K)       GPU
        log_probs: torch.Tensor,        # (E, A)          GPU
        rewards: np.ndarray,            # (E, A) float32  CPU
        states: torch.Tensor,           # (E, D_state)    GPU
        avail_actions,                  # (E, A, act_D)   GPU or None
        dones: np.ndarray,              # (E,)  float32   CPU
        truncated: np.ndarray,          # (E,)  float32   CPU
        values: torch.Tensor,           # (E, A)          GPU
        active_masks: np.ndarray,       # (E, A) float32  CPU
        rnn_states_actor=None,          # (E, A, L, H)    GPU or None
        rnn_states_critic=None,         # (E, A, L, H)    GPU or None
        rnn_masks=None,                 # (E,)            GPU or None
    ):
        E = active_envs
        # GPU tensors: one assignment each = one kernel each
        self.obs[step_idx, :E]       = obs
        self.actions[step_idx, :E]   = actions
        self.log_probs[step_idx, :E] = log_probs
        self.states[step_idx, :E] = states
        self.values[step_idx, :E]    = values
        self.time_mask[step_idx, :E] = True
        if self.use_avail and avail_actions is not None:
            self.avail_actions[step_idx, :E] = avail_actions
        # CPU → GPU: one batched transfer per field
        self.reward[step_idx, :E]       = torch.as_tensor(rewards,      dtype=torch.float32, device=self.device)
        self.done[step_idx, :E]         = torch.as_tensor(dones,        dtype=torch.float32, device=self.device)
        self.truncated[step_idx, :E]    = torch.as_tensor(truncated,    dtype=torch.float32, device=self.device)
        self.active_masks[step_idx, :E] = torch.as_tensor(active_masks, dtype=torch.float32, device=self.device)
        if self.rnn and rnn_states_actor is not None:
            self.rnn_states_actor[step_idx, :E]  = rnn_states_actor
            self.rnn_states_critic[step_idx, :E] = rnn_states_critic
            self.rnn_masks[step_idx, :E]         = rnn_masks

    def get_batch(self):
        return Batch(
            batch_obs=self.obs,
            batch_actions=self.actions,
            batch_log_probs=self.log_probs,
            batch_reward=self.reward,
            batch_states=self.states,
            batch_avail_actions=self.avail_actions,
            batch_done=self.done,
            batch_truncated=self.truncated,
            batch_values=self.values,
            batch_terminal_values=self.terminal_values,
            batch_time_mask=self.time_mask,
            batch_active_mask=self.active_masks,
            rnn_states_actor=self.rnn_states_actor if self.rnn else None,
            rnn_states_critic=self.rnn_states_critic if self.rnn else None,
            rnn_masks=self.rnn_masks if self.rnn else None,
        )


@dataclass
class RolloutResult:
    steps_collected: int
    iteration_rewards: List[float]
    ep_rewards: List[float]
    ep_lengths: List[int]
    ep_stats: List[dict]
    num_new_episodes: int


@dataclass
class FlatBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    states: torch.Tensor
    avail_actions: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor
    active_masks: torch.Tensor  # (N, A) float32


@dataclass
class RNNBatch:
    """Structured (T, E, ...) batch for chunked RNN training."""
    obs: torch.Tensor            # (T, E, A, obs_dim)
    actions: torch.Tensor        # (T, E, A)
    log_probs: torch.Tensor      # (T, E, A)
    states: torch.Tensor         # (T, E, state_dim)
    avail_actions: torch.Tensor  # (T, E, A, act_dim)
    advantages: torch.Tensor     # (T, E, A) always — per-agent (identical across A for centralized)
    returns: torch.Tensor        # (T, E, A) always
    old_values: torch.Tensor     # (T, E, A) always
    active_masks: torch.Tensor   # (T, E, A)
    time_mask: torch.Tensor      # (T, E) bool
    rnn_masks: torch.Tensor      # (T, E) float — 1=continue, 0=episode-start
    rnn_states_actor: torch.Tensor    # (T, E, A, L, H)
    rnn_states_critic: torch.Tensor   # (T, E, A, L, H) — per-agent for both centralized and IPPO


@dataclass
class TrainMetrics:
    actor_losses: List[float] = field(default_factory=list)
    critic_losses: List[float] = field(default_factory=list)
    entropy_bonuses: List[float] = field(default_factory=list)
    kl_divergences: List[float] = field(default_factory=list)
    actor_gradients: List[float] = field(default_factory=list)
    critic_gradients: List[float] = field(default_factory=list)
    clipped_ratios: List[float] = field(default_factory=list)


class Learner:
    def __init__(self, args, device, workers):
        self.args = args
        self.device = device
        self.workers = workers
        self.training_step = 0

        workers[0].send(("get_env_info", None))
        env_info = workers[0].recv()
        obs_dim = env_info["obs_size"]
        state_dim = env_info["state_size"]
        action_dim = env_info["action_size"]
        num_agents = env_info["n_agents"]
        action_space = env_info["action_space"]
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_space = action_space
        self.space_type = action_space[0]
        if self.space_type == "discrete":
            self.action_width = 1
        elif self.space_type == "multi_discrete":
            self.action_width = len(action_space[1])
        else:  # continuous
            self.action_width = int(action_space[1])
        self.use_avail = (self.space_type == "discrete") and args.avail_actions

        self.actors, self.critic = build_actor_critic(
            args, obs_dim, action_space, state_dim, num_agents, device
        )

        # dynamic=True is conservative; if shapes are actually fully static after the first batch you could try dynamic=False for slightly better optimization,
        # but expect recompiles when active_envs < num_envs on the final partial batch.
        if torch.cuda.is_available() and getattr(args, "torch_compile", True):
            self.critic = torch.compile(self.critic, dynamic=True)
            for key in list(self.actors.keys()):
                self.actors[key].act = torch.compile(self.actors[key].act, dynamic=True)

        Optimizer = getattr(optim, args.optimizer)
        self.actor_optimizer = Optimizer(
            self.actors.parameters(), lr=args.learning_rate_actor, eps=1e-5
        )
        self.critic_optimizer = Optimizer(
            self.critic.parameters(), lr=args.learning_rate_critic, eps=1e-5
        )

        rollout_steps_per_env = args.batch_size // args.num_envs
        self.rb = RolloutBuffer(
            num_steps=rollout_steps_per_env,
            num_envs=args.num_envs,
            num_agents=num_agents,
            obs_space=obs_dim,
            state_space=state_dim,
            action_space=action_space,
            centralized_critic=args.centralized_critic,
            device=device,
            rnn=args.rnn,
            actor_hidden_dim=args.actor_hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
            rnn_num_layers=args.rnn_num_layers,
            use_avail=self.use_avail,
        )

        assert not (args.normalize_value and args.normalize_popart), \
            "normalize_value and normalize_popart are mutually exclusive"
        if args.normalize_popart:
            self.value_normalizer = self.critic.v_out
        elif args.normalize_value:
            self.value_normalizer = ValueNorm(1, device=device)
        else:
            self.value_normalizer = None

        # Pre-allocated numpy buffers reused each rollout step to avoid np.stack allocations
        mask_width = int(action_space[1]) if self.use_avail else 0
        self._obs_buf = np.zeros((args.num_envs, num_agents, obs_dim), dtype=np.float32)
        # State storage shape mirrors RolloutBuffer.states: shared per env for centralized,
        # per-agent for IPPO with state critic.
        if args.centralized_critic:
            state_shape = (args.num_envs, state_dim)
        else:
            state_shape = (args.num_envs, num_agents, state_dim)
        self._state_buf = np.zeros(state_shape, dtype=np.float32)
        self._avail_buf = np.zeros((args.num_envs, num_agents, mask_width), dtype=np.bool_)

        # Pre-allocated per-step scratch buffers populated from worker contents (avoids
        # repeated np.array(list-comp) allocations inside the rollout inner loop).
        self._rewards_buf = np.zeros((args.num_envs, num_agents), dtype=np.float32)
        self._episode_reward_buf = np.zeros(args.num_envs, dtype=np.float32)
        self._dones_buf = np.zeros(args.num_envs, dtype=np.float32)
        self._truncated_buf = np.zeros(args.num_envs, dtype=np.float32)
        self._agent_dones_buf = np.zeros((args.num_envs, num_agents), dtype=np.bool_)
        self._active_masks_buf = np.zeros((args.num_envs, num_agents), dtype=np.float32)
        # Pre-step alive status carried across rollout steps (all ones = fully alive).
        # active_masks[t] is read from here so dying agents' last transition is included.
        self._current_alive_masks = np.ones((args.num_envs, num_agents), dtype=np.float32)
        self._next_obs_buf = np.zeros((args.num_envs, num_agents, obs_dim), dtype=np.float32)
        self._next_state_buf = np.zeros(state_shape, dtype=np.float32)
        self._next_avail_buf = (
            np.zeros((args.num_envs, num_agents, mask_width), dtype=np.bool_)
            if self.use_avail else None
        )

        # Pre-allocated current-state arrays; rows updated in-place after each env step/reset
        self.current_obs = np.zeros((args.num_envs, num_agents, obs_dim), dtype=np.float32)
        self.current_states = np.zeros(state_shape, dtype=np.float32)
        self.current_avail_actions = (
            np.zeros((args.num_envs, num_agents, mask_width), dtype=np.bool_)
            if self.use_avail else None
        )
        self._current_ep_rewards = None
        self._current_ep_lengths = None

        # Cached chunk index tensors for _rnn_minibatch_iter (built lazily, reused across epochs).
        self._idx_cache_key = None
        self._E_all = None
        self._A_all = None
        self._T_all = None
        self._t_range = None
        # Number of rollout steps actually written in the most recent collect_rollout.
        # Initialized to the buffer length; collect_rollout updates it each call.
        self._rollout_steps_written = self.rb.num_steps

        # RNN hidden states carried across rollouts; None when rnn=False
        if args.rnn:
            L, Ha = args.rnn_num_layers, args.actor_hidden_dim
            Hc = args.critic_hidden_dim
            self.current_rnn_actor = torch.zeros(
                args.num_envs, num_agents, L, Ha, device=device
            )
            self.current_rnn_critic = torch.zeros(args.num_envs, num_agents, L, Hc, device=device)
            # 1=continue, 0=fresh start (reset hidden at next step)
            self.current_rnn_mask = torch.ones(args.num_envs, device=device)
            # Pre-allocated arange for scatter_ in centralized dedup inside _train_rnn.
            _max_rnn_mb = max(1, (args.minibatch_size * num_agents) // args.data_chunk_length)
            self._rnn_mb_arange = torch.arange(_max_rnn_mb, device=device)
        else:
            self.current_rnn_actor = None
            self.current_rnn_critic = None
            self.current_rnn_mask = None
            self._rnn_mb_arange = None

        self.initialize_env_state()

    def set_learning_rates(self, actor_lr: float, critic_lr: float):
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = critic_lr

    def initialize_env_state(self):
        for worker in self.workers:
            worker.send(("reset", None))
        contents = [worker.recv() for worker in self.workers]
        for i, c in enumerate(contents):
            self.current_obs[i] = c["obs"]
            self.current_states[i] = c["state"]
            if self.use_avail:
                self.current_avail_actions[i] = c["avail_actions"]
        self._current_ep_rewards = np.zeros(self.args.num_envs, dtype=np.float64)
        self._current_ep_lengths = np.zeros(self.args.num_envs, dtype=np.int64)
        self._current_alive_masks[:] = 1.0

    def _compute_terminal_values(self, contents, truncated_envs, new_critic_hxs):
        """Batch-compute V(next_state) for truncated transitions using the end-of-step
        critic hidden states. Returns a (K, A) tensor in raw (pre-denorm) critic space,
        matching how batch_values is stored."""
        args = self.args
        device = self.device
        A = self.num_agents
        K = len(truncated_envs)

        if args.centralized_critic:
            next_states_np = np.stack([contents[e]["next_state"] for e in truncated_envs])
            ns_t = torch.as_tensor(next_states_np, dtype=torch.float32, device=device)  # (K, state_D)
            if args.rnn:
                # Per-agent streams are identical for centralized — run once per env.
                sel = torch.tensor(truncated_envs, device=device, dtype=torch.long)
                hxs_E = new_critic_hxs[sel, 0]              # (K, L, H) — agent 0's stream
                masks = torch.ones(K, 1, device=device)
                v_E, _ = self.critic(ns_t, rnn_hxs=hxs_E, masks=masks)
                return v_E.squeeze(-1).unsqueeze(-1).expand(K, A).contiguous()
            v, _ = self.critic(ns_t)
            return v.squeeze(-1).unsqueeze(-1).expand(K, A).clone()

        # IPPO with state critic: per-agent next_state from env.get_state().
        next_state_np = np.stack([contents[e]["next_state"] for e in truncated_envs])  # (K, A, state_D)
        ns_t = torch.as_tensor(next_state_np, dtype=torch.float32, device=device)
        if getattr(self.critic, "use_attention", False):
            v, _ = self.critic(ns_t)
            return v.squeeze(-1)   # (K, A)
        flat_ns = ns_t.reshape(K * A, self.state_dim)
        if args.rnn:
            sel = torch.tensor(truncated_envs, device=device, dtype=torch.long)
            hxs = new_critic_hxs[sel].reshape(K * A, *new_critic_hxs.shape[2:])
            masks = torch.ones(K * A, 1, device=device)
            v_flat, _ = self.critic(flat_ns, rnn_hxs=hxs, masks=masks)
        else:
            v_flat, _ = self.critic(flat_ns)
        return v_flat.squeeze(-1).reshape(K, A)

    def collect_rollout(self, step) -> RolloutResult:
        args = self.args
        num_parallel_envs = args.num_envs
        device = self.device
        transitions_remaining = min(args.batch_size, args.total_timesteps - step)

        self.rb.reset()
        iteration_rewards = []
        ep_rewards = []
        ep_lengths = []
        ep_stats = []
        num_new_episodes = 0

        rollout_steps = (transitions_remaining + num_parallel_envs - 1) // num_parallel_envs

        collected_steps = 0
        for rollout_t in range(rollout_steps):
            active_envs = min(num_parallel_envs, transitions_remaining - collected_steps)
            self._obs_buf[:active_envs] = self.current_obs[:active_envs]
            self._state_buf[:active_envs] = self.current_states[:active_envs]
            if self.use_avail:
                self._avail_buf[:active_envs] = self.current_avail_actions[:active_envs]
            obs_tensor = torch.as_tensor(self._obs_buf[:active_envs], dtype=torch.float32, device=device)
            state_tensor = torch.as_tensor(self._state_buf[:active_envs], dtype=torch.float32, device=device)
            if self.use_avail:
                avail_action_tensor = torch.as_tensor(self._avail_buf[:active_envs], dtype=torch.bool, device=device)
            else:
                avail_action_tensor = None

            rnn_hxs_a = self.current_rnn_actor[:active_envs] if args.rnn else None
            rnn_hxs_c = self.current_rnn_critic[:active_envs] if args.rnn else None
            rnn_masks_e = self.current_rnn_mask[:active_envs] if args.rnn else None

            with torch.no_grad():
                actions, log_probs, _, new_actor_hxs = act_with_policies(
                    args, self.actors, obs_tensor, avail_actions=avail_action_tensor,
                    rnn_hxs=rnn_hxs_a, masks=rnn_masks_e,
                )
                A = self.num_agents
                if args.centralized_critic:
                    if args.rnn:
                        # All A per-agent RNN streams stay identical (same state input, same
                        # zero-init hidden, same zero-reset on termination). Run once per env.
                        agent0_hxs = rnn_hxs_c[:, 0]              # (E, L, H)
                        c_masks = rnn_masks_e.unsqueeze(1)         # (E, 1)
                        vals_E, new_hxs_E = self.critic(state_tensor, rnn_hxs=agent0_hxs, masks=c_masks)
                        values = vals_E.squeeze(-1).unsqueeze(-1).expand(active_envs, A).contiguous()
                        new_critic_hxs = new_hxs_E.unsqueeze(1).expand(
                            active_envs, A, *new_hxs_E.shape[1:]
                        ).contiguous()
                    else:
                        raw, _ = self.critic(state_tensor)
                        values = raw.squeeze(-1).unsqueeze(-1).expand(active_envs, A).clone()  # (E, A)
                else:
                    # IPPO with state critic: state_tensor is (E, A, state_D) per-agent.
                    if getattr(self.critic, "use_attention", False):
                        # Cross-agent attention: keep the agent axis intact.
                        values, _ = self.critic(state_tensor)
                        values = values.squeeze(-1)   # (E, A)
                    else:
                        flat_state = state_tensor.reshape(active_envs * A, self.state_dim)
                        if args.rnn:
                            flat_c_hxs = rnn_hxs_c.reshape(active_envs * A, *rnn_hxs_c.shape[2:])
                            flat_c_masks = rnn_masks_e.unsqueeze(1).expand(active_envs, A).reshape(active_envs * A, 1)
                            values, new_critic_hxs_flat = self.critic(flat_state, rnn_hxs=flat_c_hxs, masks=flat_c_masks)
                            new_critic_hxs = new_critic_hxs_flat.reshape(active_envs, A, *new_critic_hxs_flat.shape[1:])
                        else:
                            values, _ = self.critic(flat_state)
                        values = values.squeeze(-1).reshape(active_envs, A)

            actions_np = actions.cpu().numpy()
            for env_idx in range(active_envs):
                self.workers[env_idx].send(("step", actions_np[env_idx]))
            contents = [self.workers[env_idx].recv() for env_idx in range(active_envs)]

            # Compute GAE bootstrap V(next_state) for truncated transitions *before* the
            # env loop overwrites current_state/obs/hxs via reset. Stored raw (pre-denorm)
            # to match batch_values; denormalized in compute_advantages.
            truncated_envs = [
                i for i in range(active_envs) if bool(contents[i]["truncated"])
            ]
            if truncated_envs:
                with torch.no_grad():
                    tv = self._compute_terminal_values(
                        contents,
                        truncated_envs,
                        new_critic_hxs if args.rnn else None,
                    )
                trunc_t = torch.as_tensor(truncated_envs, device=device, dtype=torch.long)
                self.rb.terminal_values[rollout_t, trunc_t] = tv

            # ── Phase 1: build batch CPU arrays from worker results (vectorized) ─────
            # Fill pre-allocated scratch buffers row-by-row instead of allocating new
            # arrays via np.array(list-comp) / np.stack each step.
            rewards_buf      = self._rewards_buf[:active_envs]
            dones_buf        = self._dones_buf[:active_envs]
            truncated_buf    = self._truncated_buf[:active_envs]
            agent_dones_view = self._agent_dones_buf[:active_envs]
            active_masks_buf = self._active_masks_buf[:active_envs]
            for i in range(active_envs):
                c = contents[i]
                rewards_buf[i]   = c["reward"]
                dones_buf[i]     = c["done"]
                truncated_buf[i] = c["truncated"]
                agent_dones_view[i] = c["agent_dones"]
            terminal_mask = (dones_buf > 0) | (truncated_buf > 0)
            # Use pre-step alive status: a dying agent's last transition is not masked out,
            # and already-dead agents' no-op terminal transitions remain masked.
            active_masks_buf[...] = self._current_alive_masks[:active_envs]
            episode_rewards_buf = self._episode_reward_buf[:active_envs]
            episode_rewards_buf[...] = rewards_buf.mean(axis=1)

            # ── Phase 2: single vectorized buffer write ─
            self.rb.add_step(
                step_idx=rollout_t,
                active_envs=active_envs,
                obs=obs_tensor,
                actions=actions,
                log_probs=log_probs,
                rewards=rewards_buf,
                states=state_tensor,
                avail_actions=avail_action_tensor,
                dones=dones_buf,
                truncated=truncated_buf,
                values=values,
                active_masks=active_masks_buf,
                rnn_states_actor=rnn_hxs_a,  # view of h_t — copied inside add_step before overwrite below
                rnn_states_critic=rnn_hxs_c if args.rnn else None,
                rnn_masks=rnn_masks_e if args.rnn else None,
            )

            # Now safe to overwrite: add_step has already copied h_t into the buffer on the same stream.
            if args.rnn:
                self.current_rnn_actor[:active_envs] = new_actor_hxs.detach()

            # ── Phase 3: episode tracking + per-env state updates (vectorized) ──
            # Vectorized episode reward/length update
            self._current_ep_rewards[:active_envs] += episode_rewards_buf
            self._current_ep_lengths[:active_envs] += 1

            done_idx_np = np.where(terminal_mask)[0]
            not_done_idx_np = np.where(~terminal_mask)[0]
            done_env_indices = done_idx_np.tolist()

            # Advance alive masks for the next step: post-step dead agents → 0;
            # done envs reset to 1 since a new episode (all agents alive) begins.
            self._current_alive_masks[:active_envs] = ~agent_dones_view
            if done_idx_np.size > 0:
                self._current_alive_masks[done_idx_np] = 1.0

            # Episode-completion bookkeeping (small loop only over done envs)
            for env_idx in done_env_indices:
                ep_rewards.append(float(self._current_ep_rewards[env_idx]))
                ep_lengths.append(int(self._current_ep_lengths[env_idx]))
                iteration_rewards.append(float(self._current_ep_rewards[env_idx]))
                ep_stats.append(contents[env_idx]["infos"])
            num_new_episodes += len(done_env_indices)
            if done_env_indices:
                self._current_ep_rewards[done_idx_np] = 0.0
                self._current_ep_lengths[done_idx_np] = 0

            # State updates for non-terminal envs: write each env's next_* into a
            # pre-allocated row of the scratch buffer, then a single fancy-indexed
            # assign into current_*.
            n_not_done = len(not_done_idx_np)
            if n_not_done > 0:
                next_obs_view   = self._next_obs_buf[:n_not_done]
                next_state_view = self._next_state_buf[:n_not_done]
                next_avail_view = self._next_avail_buf[:n_not_done] if self.use_avail else None
                for j, i in enumerate(not_done_idx_np):
                    c = contents[i]
                    next_obs_view[j]   = c["next_obs"]
                    next_state_view[j] = c["next_state"]
                    if next_avail_view is not None:
                        next_avail_view[j] = c["avail_actions"]
                self.current_obs[not_done_idx_np]    = next_obs_view
                self.current_states[not_done_idx_np] = next_state_view
                if next_avail_view is not None:
                    self.current_avail_actions[not_done_idx_np] = next_avail_view

            # Vectorized RNN state updates (single batched indexed write each)
            if args.rnn:
                if len(not_done_idx_np) > 0:
                    nd_t = torch.as_tensor(not_done_idx_np, device=device, dtype=torch.long)
                    self.current_rnn_critic[nd_t] = new_critic_hxs[nd_t].detach()
                    self.current_rnn_mask[nd_t]   = 1.0
                if done_env_indices:
                    d_t = torch.as_tensor(done_idx_np, device=device, dtype=torch.long)
                    self.current_rnn_actor[d_t] = 0
                    self.current_rnn_critic[d_t] = 0
                    self.current_rnn_mask[d_t]   = 0.0

            # ── Phase 4: fan-out resets (all done envs reset in parallel) ─────
            for env_idx in done_env_indices:
                self.workers[env_idx].send(("reset", None))
            for env_idx in done_env_indices:
                reset_content = self.workers[env_idx].recv()
                self.current_obs[env_idx] = reset_content["obs"]
                self.current_states[env_idx] = reset_content["state"]
                if self.use_avail:
                    self.current_avail_actions[env_idx] = reset_content["avail_actions"]

            collected_steps += active_envs

        self._rollout_steps_written = rollout_steps

        return RolloutResult(
            steps_collected=collected_steps,
            iteration_rewards=iteration_rewards,
            ep_rewards=ep_rewards,
            ep_lengths=ep_lengths,
            ep_stats=ep_stats,
            num_new_episodes=num_new_episodes,
        )

    def compute_advantages(self):
        args = self.args
        device = self.device
        A = self.num_agents

        batch = self.rb.get_batch()

        with torch.no_grad():
            T = batch.batch_reward.shape[0]
            E = args.num_envs
            lam = args.td_lambda if args.use_gae else 1.0

            # --- Compute per-agent next_values (E, A) for bootstrap ---
            E_active = int(batch.batch_time_mask.any(dim=0).sum())
            next_values = torch.zeros(E, A, device=device)
            if args.centralized_critic:
                next_state_tensor = torch.as_tensor(self.current_states[:E_active], dtype=torch.float32, device=device)
                if args.rnn:
                    # All A streams are identical for centralized — use stream 0.
                    agent0_hxs = self.current_rnn_critic[:E_active, 0]        # (E_active, L, H)
                    c_masks = self.current_rnn_mask[:E_active].unsqueeze(1)   # (E_active, 1)
                    nv, _ = self.critic(next_state_tensor, rnn_hxs=agent0_hxs, masks=c_masks)
                else:
                    nv, _ = self.critic(next_state_tensor)
                next_values[:E_active] = nv.squeeze(-1).unsqueeze(-1).expand(E_active, A)
            else:
                # IPPO with state critic: per-agent state, current_states shape (E, A, state_D).
                next_state_tensor = torch.as_tensor(self.current_states[:E_active], dtype=torch.float32, device=device)
                if getattr(self.critic, "use_attention", False):
                    nv, _ = self.critic(next_state_tensor)
                    next_values[:E_active] = nv.squeeze(-1)
                else:
                    flat_next_state = next_state_tensor.reshape(E_active * A, self.state_dim)
                    if args.rnn:
                        flat_c_hxs = self.current_rnn_critic[:E_active].reshape(E_active * A, *self.current_rnn_critic.shape[2:])
                        flat_c_masks = self.current_rnn_mask[:E_active].unsqueeze(1).expand(E_active, A).reshape(E_active * A, 1)
                        nv, _ = self.critic(flat_next_state, rnn_hxs=flat_c_hxs, masks=flat_c_masks)
                    else:
                        nv, _ = self.critic(flat_next_state)
                    next_values[:E_active] = nv.squeeze(-1).reshape(E_active, A)

            # --- Unified per-agent GAE: all shapes (T, E, A) ---
            raw_values = batch.batch_values                        # (T, E, A)
            if self.value_normalizer is not None:
                raw_values = self.value_normalizer.denormalize(raw_values)
                raw_next_values = self.value_normalizer.denormalize(next_values)
                raw_terminal_values = self.value_normalizer.denormalize(batch.batch_terminal_values)
            else:
                raw_next_values = next_values                      # (E, A)
                raw_terminal_values = batch.batch_terminal_values  # (T, E, A)

            # bootstrap_mask: allow V(s_{t+1}) for truncated, zero only for true terminal
            bootstrap_mask = (1.0 - batch.batch_done).unsqueeze(-1)            # (T, E, 1)
            # gae_continue: stop GAE accumulation at any episode boundary
            episode_end = (batch.batch_done.bool() | batch.batch_truncated.bool()).float()
            gae_continue = (1.0 - episode_end).unsqueeze(-1)                   # (T, E, 1)
            reward = batch.batch_reward                                        # (T, E, A)

            # next_val[t] = V(s_{t+1}); last valid step per env gets the critic bootstrap
            next_val = torch.zeros_like(raw_values)
            next_val[:-1] = raw_values[1:]
            ep_lens = batch.batch_time_mask.long().sum(dim=0)      # (E,)
            last_t = (ep_lens - 1).clamp(min=0)                    # (E,)
            idx = last_t.view(1, E, 1).expand(1, E, A)
            next_val.scatter_(0, idx, raw_next_values.unsqueeze(0))

            # Truncation bootstrap
            trunc_mask = batch.batch_truncated.bool().unsqueeze(-1).expand_as(next_val)  # (T, E, A)
            next_val = torch.where(trunc_mask, raw_terminal_values, next_val)

            delta = reward + args.gamma * next_val * bootstrap_mask - raw_values  # (T, E, A)

            advantages = torch.zeros_like(raw_values)              # (T, E, A)
            gae = torch.zeros(E, A, device=device)
            mask_float = batch.batch_time_mask.float().unsqueeze(-1)   # (T, E, 1)
            for t in reversed(range(T)):
                gae = (delta[t] + args.gamma * lam * gae_continue[t] * gae) * mask_float[t]
                advantages[t] = gae
            return_lambda = advantages + raw_values                # (T, E, A)

        # Normalize
        time_expanded = batch.batch_time_mask.unsqueeze(-1).expand_as(advantages)
        if args.normalize_advantage:
            active_valid = time_expanded & batch.batch_active_mask.bool()
            active_advantages = advantages[active_valid]
            if active_advantages.numel() > 0:
                adv_mu = active_advantages.mean()
                adv_std = active_advantages.std(unbiased=False)
                advantages[time_expanded] = (advantages[time_expanded] - adv_mu) / (adv_std + 1e-5)

        if args.rnn:
            return RNNBatch(
                obs=batch.batch_obs,
                actions=batch.batch_actions,
                log_probs=batch.batch_log_probs,
                states=batch.batch_states,
                avail_actions=batch.batch_avail_actions,
                advantages=advantages,
                returns=return_lambda,
                old_values=batch.batch_values,
                active_masks=batch.batch_active_mask,
                time_mask=batch.batch_time_mask,
                rnn_masks=batch.rnn_masks,
                rnn_states_actor=batch.rnn_states_actor,
                rnn_states_critic=batch.rnn_states_critic,
            )

        # FlatBatch (non-RNN)
        tm = batch.batch_time_mask
        return FlatBatch(
            obs=batch.batch_obs[tm],
            actions=batch.batch_actions[tm],
            log_probs=batch.batch_log_probs[tm],
            states=batch.batch_states[tm],
            avail_actions=batch.batch_avail_actions[tm],
            advantages=advantages[tm],
            returns=return_lambda[tm],
            old_values=batch.batch_values[tm],
            active_masks=batch.batch_active_mask[tm],
        )

    def _masked_mean(self, x, mask):
        return (x * mask).sum() / mask.sum().clamp(min=1)

    def _policy_loss(self, ratio, advantages, log_ratio):
        """Compute per-sample policy loss (before masking/averaging).

        PPO:  max(-r·Â, -clip(r,1-ε,1+ε)·Â).
        SPO:  r·Â - (|Â|/2ε)·(r-1)²  → negate for gradient descent.
        MARS: r·Â - αₜ·(r + 1/r - 2)  → negate for gradient descent.
              Multiplicatively symmetric: B_upper = 1+ε, B_lower = 1/B_upper.
              αₜ = Â / (1 - r_target⁻²), where r_target = B_upper if Â≥0 else B_lower.
        """
        algo = self.args.algorithm
        eps = self.args.trust_region_clip
        if algo == "spo":
            return -(ratio * advantages - (advantages.abs() / (2 * eps)) * (ratio - 1) ** 2)
        if algo == "mars":
            b_upper = 1.0 + eps
            b_lower = 1.0 / b_upper
            r_target = torch.where(advantages >= 0, torch.full_like(advantages, b_upper), torch.full_like(advantages, b_lower))
            alpha_t = advantages / (1.0 - r_target.pow(-2))
            ratio_safe = ratio.clamp_min(1e-8)
            penalty = ratio_safe + 1.0 / ratio_safe - 2.0
            return -(ratio * advantages - alpha_t * penalty)
        return torch.max(
            -advantages * ratio,
            -advantages * ratio.clamp(1 - eps, 1 + eps),
        )

    def _compute_actor_stats(
        self, mb_obs, mb_actions, mb_log_probs, mb_avail_actions, mb_advantages,
        mb_active_masks, entropy_coef,
        mb_rnn_hxs_actor=None, mb_rnn_masks=None,
    ):
        """
        Non-RNN: mb_obs (B, A, D), mb_rnn_hxs_actor=None.
        RNN:     mb_obs (L*N, A, D), mb_rnn_hxs_actor (N, A, L_rnn, H), mb_rnn_masks (L*N, 1).
                 L=data_chunk_length, N=env-chunks in mini-batch.
        Returns: (actor_loss, entropy_loss, kl_divergence, clipped_ratio)
        """
        args = self.args
        use_rnn = mb_rnn_hxs_actor is not None
        K = self.action_width
        avail_arg = mb_avail_actions if self.use_avail else None

        if args.shared_policies:
            LN, num_agents, obs_dim = mb_obs.shape  # LN = B or L*N
            if use_rnn:
                # Flatten (L*N, A, D) → (L*N*A, D); hxs (N, A, L_rnn, H) → (N*A, L_rnn, H)
                N = mb_rnn_hxs_actor.shape[0]
                L_rnn_dim, H = mb_rnn_hxs_actor.shape[2], mb_rnn_hxs_actor.shape[3]
                flat_obs = mb_obs.reshape(LN * num_agents, obs_dim)
                flat_avail = avail_arg.reshape(LN * num_agents, -1) if avail_arg is not None else None
                flat_action = mb_actions.reshape(LN * num_agents, K)
                flat_hxs = mb_rnn_hxs_actor.reshape(N * num_agents, L_rnn_dim, H)
                # masks (L*N, 1) → per-agent: (L*N, A, 1) → (L*N*A, 1)
                flat_masks = mb_rnn_masks.unsqueeze(1).expand(LN, num_agents, 1).reshape(LN * num_agents, 1)
                _, current_logprob, entropy, _ = self.actors["actor_0"].act(
                    flat_obs, avail_action=flat_avail, action=flat_action,
                    rnn_hxs=flat_hxs, masks=flat_masks,
                )
            else:
                _, current_logprob, entropy, _ = act_with_policies(
                    args, self.actors, mb_obs, avail_actions=avail_arg, action=mb_actions,
                )
                current_logprob = current_logprob.reshape(-1)
                entropy = entropy.reshape(-1)

            log_ratio = current_logprob.reshape(-1) - mb_log_probs.reshape(-1)
            ratio = log_ratio.exp()
            if args.centralized_critic and mb_advantages.dim() == 1:
                expanded_advantages = mb_advantages.unsqueeze(-1).expand(-1, num_agents).reshape(-1)
            else:
                expanded_advantages = mb_advantages.reshape(-1)
            pg_loss = self._policy_loss(ratio, expanded_advantages, log_ratio)
            entropy_flat = entropy.reshape(-1)
            if args.policy_mask:
                mask_flat = mb_active_masks.reshape(-1)
                actor_pg = self._masked_mean(pg_loss, mask_flat)
                entropy_loss = self._masked_mean(entropy_flat, mask_flat)
            else:
                actor_pg = pg_loss.mean()
                entropy_loss = entropy_flat.mean()
            kl_divergence = ((ratio - 1) - log_ratio).mean()
            clipped_ratio = ((ratio - 1.0).abs() > args.trust_region_clip).float().mean()
            actor_loss = actor_pg - entropy_coef * entropy_loss
            return actor_loss, entropy_loss, kl_divergence, clipped_ratio

        actor_loss = torch.tensor(0.0, device=mb_obs.device)
        entropy_loss = torch.tensor(0.0, device=mb_obs.device)
        kl_divergence = torch.tensor(0.0, device=mb_obs.device)
        clipped_ratio = torch.tensor(0.0, device=mb_obs.device)
        LN = mb_obs.shape[0]
        for agent_idx, actor in enumerate(self.actors.values()):
            avail_i = avail_arg[:, agent_idx] if avail_arg is not None else None
            if use_rnn:
                N = mb_rnn_hxs_actor.shape[0]
                hxs_i = mb_rnn_hxs_actor[:, agent_idx]  # (N, L_rnn, H)
                _, current_logprob, entropy, _ = actor.act(
                    mb_obs[:, agent_idx],
                    avail_action=avail_i,
                    action=mb_actions[:, agent_idx],
                    rnn_hxs=hxs_i, masks=mb_rnn_masks,
                )
            else:
                _, current_logprob, entropy, _ = actor.act(
                    mb_obs[:, agent_idx],
                    avail_action=avail_i,
                    action=mb_actions[:, agent_idx],
                )
            log_ratio = current_logprob - mb_log_probs[:, agent_idx]
            ratio = log_ratio.exp()
            if args.centralized_critic and mb_advantages.dim() == 1:
                adv = mb_advantages
            else:
                adv = mb_advantages[:, agent_idx]
            pg_loss = self._policy_loss(ratio, adv, log_ratio)
            if args.policy_mask:
                mask_i = mb_active_masks[:, agent_idx]
                actor_loss = actor_loss + self._masked_mean(pg_loss, mask_i)
                entropy_loss = entropy_loss + self._masked_mean(entropy, mask_i)
            else:
                actor_loss = actor_loss + pg_loss.mean()
                entropy_loss = entropy_loss + entropy.mean()
            kl_divergence = kl_divergence + ((ratio - 1) - log_ratio).mean()
            clipped_ratio = clipped_ratio + ((ratio - 1.0).abs() > args.trust_region_clip).float().mean()
        n = len(self.actors)
        actor_loss = actor_loss / n - entropy_coef * (entropy_loss / n)
        return actor_loss, entropy_loss / n, kl_divergence / n, clipped_ratio / n

    def _critic_value(self, mb_obs, mb_states, mb_rnn_hxs_critic=None, mb_rnn_masks=None):
        """Return critic values for a mini-batch. Handles both RNN and non-RNN, centralized and IPPO.

        Centralized: mb_states (B, state_D) -> values (B,).
        IPPO with state critic: mb_states (B, A, state_D) -> values (B, A).
        """
        args = self.args
        use_rnn = mb_rnn_hxs_critic is not None
        if args.centralized_critic:
            if use_rnn:
                values, _ = self.critic(mb_states, rnn_hxs=mb_rnn_hxs_critic, masks=mb_rnn_masks)
            else:
                values, _ = self.critic(mb_states)
            return values.squeeze(-1)
        else:
            if getattr(self.critic, "use_attention", False):
                # Cross-agent MHA needs the agent axis preserved.
                values, _ = self.critic(mb_states)
                return values.squeeze(-1)   # (B, A)
            _B, _A, _D = mb_states.shape
            flat = mb_states.reshape(_B * _A, _D)
            if use_rnn:
                flat_hxs = mb_rnn_hxs_critic.reshape(_B * _A, *mb_rnn_hxs_critic.shape[2:])
                flat_m = mb_rnn_masks.unsqueeze(1).expand(_B, _A, 1).reshape(_B * _A, 1)
                values, _ = self.critic(flat, rnn_hxs=flat_hxs, masks=flat_m)
            else:
                values, _ = self.critic(flat)
            return values.squeeze(-1).reshape(_B, _A)

    def _gradient_step(self, actor_loss, critic_loss, metrics):
        args = self.args
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        critic_loss.backward()
        max_norm = args.clip_gradients if args.clip_gradients > 0 else float("inf")
        ag = torch.nn.utils.clip_grad_norm_(self.actors.parameters(), max_norm=max_norm)
        cg = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=max_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.training_step += 1
        metrics.actor_gradients.append(ag.item())
        metrics.critic_gradients.append(cg.item())

    def _value_norm_samples(self, mb_returns, mb_active_masks):
        """Flatten value targets into scalar samples for the global ValueNorm."""
        if self.args.value_mask:
            if mb_returns.dim() < mb_active_masks.dim():
                valid = mb_active_masks.bool().any(dim=-1)
                samples = mb_returns[valid]
            elif mb_returns.shape == mb_active_masks.shape:
                samples = mb_returns[mb_active_masks.bool()]
            else:
                samples = mb_returns.reshape(-1)
        else:
            samples = mb_returns.reshape(-1)
        return samples.reshape(-1, 1)

    def _critic_loss(self, current_values, mb_returns, mb_old_values, mb_active_masks):
        args = self.args
        if self.value_normalizer is not None:
            norm_samples = self._value_norm_samples(mb_returns, mb_active_masks)
            if norm_samples.numel() > 0:
                self.value_normalizer.update(norm_samples)
            target = self.value_normalizer.normalize(mb_returns)
        else:
            target = mb_returns

        if current_values.shape != target.shape:
            if current_values.dim() < target.dim():
                current_values = current_values.unsqueeze(-1)
            current_values = current_values.expand_as(target)
        if mb_old_values.shape != target.shape:
            if mb_old_values.dim() < target.dim():
                mb_old_values = mb_old_values.unsqueeze(-1)
            mb_old_values = mb_old_values.expand_as(target)

        if args.clip_value_loss > 0:
            clipped = mb_old_values + (current_values - mb_old_values).clamp(
                -args.clip_value_loss, args.clip_value_loss
            )
            if args.huber_coef > 0:
                raw = torch.max(
                    F.huber_loss(current_values, target, reduction="none", delta=args.huber_coef),
                    F.huber_loss(clipped, target, reduction="none", delta=args.huber_coef),
                )
            else:
                raw = torch.max(
                    F.mse_loss(current_values, target, reduction="none"),
                    F.mse_loss(clipped, target, reduction="none"),
                )
        elif args.huber_coef > 0:
            raw = F.huber_loss(current_values, target, reduction="none", delta=args.huber_coef)
        else:
            raw = F.mse_loss(current_values, target, reduction="none")

        if args.value_mask:
            if raw.dim() < mb_active_masks.dim():
                raw = raw.unsqueeze(-1)
            if raw.shape != mb_active_masks.shape:
                # V(s) can be shared across agents; align it before applying masks.
                raw = raw.expand_as(mb_active_masks)
            loss = self._masked_mean(raw, mb_active_masks)
        else:
            loss = raw.mean()
        return args.value_loss_coef * loss

    # ------------------------------------------------------------------
    # RNN training helpers
    # ------------------------------------------------------------------

    def _rnn_minibatch_iter(self, rnn_data: "RNNBatch"):
        """
        Yield mini-batches for chunked RNN training, matching onpolicy's per-(env, agent) chunking.

        Each chunk is a single (env, agent) stream of length L. Mini-batch shapes:
          obs        (L, N, D_obs)     actions     (L, N)
          log_probs  (L, N)            avail       (L, N, act_D) or None
          advantages (L, N)            active_masks(L, N)
          rnn_masks  (L, N, 1)         hxs_actor   (N, L_rnn, H_a)
          states     (L, N, state_D)   returns     (L, N)
          old_values (L, N)            hxs_critic  (N, L_rnn, H_c)
          agent_indices (N,) GPU tensor — which agent each chunk belongs to
        where N = chunks per mini-batch, L = data_chunk_length.
        """
        args = self.args
        _T_buf, E = rnn_data.time_mask.shape
        A = self.num_agents
        L = args.data_chunk_length
        # Use the number of rows actually written this rollout, not the buffer
        # capacity — on the final partial batch the tail rows hold stale data
        # from the prior rollout and must not be sampled.
        T = min(_T_buf, self._rollout_steps_written)
        n_per_env = T // L
        if n_per_env == 0:
            return

        n_triples = E * A * n_per_env
        mb_size = max(1, (args.minibatch_size * A) // L)

        # Index tensors depend only on (E, A, n_per_env, L) — cache across epochs.
        cache_key = (E, A, n_per_env, L)
        if self._idx_cache_key != cache_key:
            self._idx_cache_key = cache_key
            self._E_all = torch.arange(E, device=self.device).repeat_interleave(A * n_per_env)
            self._A_all = torch.arange(A, device=self.device).repeat(E).repeat_interleave(n_per_env)
            self._T_all = torch.arange(n_per_env, device=self.device).repeat(E * A) * L
            self._t_range = torch.arange(L, device=self.device)
        E_all, A_all, T_all, t_range = self._E_all, self._A_all, self._T_all, self._t_range

        perm = torch.randperm(n_triples, device=self.device)

        for start in range(0, n_triples, mb_size):
            idx    = perm[start: start + mb_size]   # (N,) stays on GPU — no .tolist() sync
            E_idx  = E_all[idx]                      # (N,)
            A_idx  = A_all[idx]                      # (N,)
            T_idx  = T_all[idx]                      # (N,)
            N = idx.shape[0]

            # Time offset grid: t_off[l, n] = T_idx[n] + l  →  (L, N)
            t_off = T_idx.unsqueeze(0) + t_range.unsqueeze(1)
            E_exp = E_idx.unsqueeze(0).expand(L, N)
            A_exp = A_idx.unsqueeze(0).expand(L, N)

            # Single advanced-index gather per field — replaces list comprehension + torch.stack
            obs_mb       = rnn_data.obs[t_off, E_exp, A_exp]           # (L, N, D_obs)
            actions_mb   = rnn_data.actions[t_off, E_exp, A_exp]       # (L, N, K)
            log_probs_mb = rnn_data.log_probs[t_off, E_exp, A_exp]     # (L, N)
            active_mb    = rnn_data.active_masks[t_off, E_exp, A_exp]  # (L, N)
            # Centralized states are shared across agents (T, E, state_D); IPPO states are
            # per-agent (T, E, A, state_D) so each chunk gathers its own agent's stream.
            if args.centralized_critic:
                states_mb = rnn_data.states[t_off, E_exp]              # (L, N, state_D)
            else:
                states_mb = rnn_data.states[t_off, E_exp, A_exp]       # (L, N, state_D)
            masks_mb     = rnn_data.rnn_masks[t_off, E_exp].unsqueeze(-1)  # (L, N, 1)
            adv_mb       = rnn_data.advantages[t_off, E_exp, A_exp]    # (L, N)
            ret_mb       = rnn_data.returns[t_off, E_exp, A_exp]       # (L, N)
            oldv_mb      = rnn_data.old_values[t_off, E_exp, A_exp]    # (L, N)

            avail_mb = rnn_data.avail_actions[t_off, E_exp, A_exp] if self.use_avail else None

            # Initial hidden states: one per chunk (point-lookup, no slice)
            hxs_actor_mb  = rnn_data.rnn_states_actor[T_idx, E_idx, A_idx]   # (N, L_rnn, H_a)
            hxs_critic_mb = rnn_data.rnn_states_critic[T_idx, E_idx, A_idx]  # (N, L_rnn, H_c)

            yield {
                "obs": obs_mb,
                "actions": actions_mb,
                "log_probs": log_probs_mb,
                "avail": avail_mb,
                "advantages": adv_mb,
                "active_masks": active_mb,
                "rnn_masks": masks_mb,
                "hxs_actor": hxs_actor_mb,
                "states": states_mb,
                "returns": ret_mb,
                "old_values": oldv_mb,
                "hxs_critic": hxs_critic_mb,
                "agent_indices": A_idx,  # (N,) GPU tensor
                "env_indices": E_idx,    # (N,) GPU tensor — chunk's env id
                "time_indices": T_idx,   # (N,) GPU tensor — chunk start step (multiple of L)
            }

    def train(self, data, entropy_coef: float) -> TrainMetrics:
        args = self.args
        metrics = TrainMetrics()

        if args.rnn:
            return self._train_rnn(data, entropy_coef, metrics)

        flat_data: FlatBatch = data
        # flat_data is indexed by env-transition rows; each row still carries the
        # agent dimension, so minibatch_size already selects the intended rows.
        mb_step = args.minibatch_size
        for _ in range(args.train_epochs):
            b_inds = torch.randperm(flat_data.advantages.size(0), device=self.device)
            for start in range(0, flat_data.advantages.size(0), mb_step):
                end = start + mb_step
                mb_inds = b_inds[start:end]

                mb_obs           = flat_data.obs[mb_inds]
                mb_actions       = flat_data.actions[mb_inds]
                mb_log_probs     = flat_data.log_probs[mb_inds]
                mb_avail_actions = flat_data.avail_actions[mb_inds]
                mb_advantages    = flat_data.advantages[mb_inds]
                mb_states        = flat_data.states[mb_inds]
                mb_returns       = flat_data.returns[mb_inds]
                mb_old_values    = flat_data.old_values[mb_inds]
                mb_active_masks  = flat_data.active_masks[mb_inds]

                actor_loss, entropy_loss, kl_divergence, clipped_ratio = (
                    self._compute_actor_stats(
                        mb_obs, mb_actions, mb_log_probs, mb_avail_actions,
                        mb_advantages, mb_active_masks, entropy_coef,
                    )
                )
                current_values = self._critic_value(mb_obs, mb_states)
                critic_loss = self._critic_loss(current_values, mb_returns, mb_old_values, mb_active_masks)

                self._gradient_step(actor_loss, critic_loss, metrics)
                metrics.actor_losses.append(actor_loss.item())
                metrics.critic_losses.append(critic_loss.item())
                metrics.entropy_bonuses.append(entropy_loss.item())
                metrics.kl_divergences.append(kl_divergence.item())
                metrics.clipped_ratios.append(clipped_ratio.item())

        return metrics

    def _train_rnn(self, rnn_data: "RNNBatch", entropy_coef: float, metrics: TrainMetrics) -> TrainMetrics:
        """
        Per-(env, agent) chunked RNN training matching onpolicy's recurrent_generator structure.
        Each mini-batch chunk is a single (env, agent) stream of length L — no agent dimension.
        """
        args = self.args

        K = self.action_width
        for _ in range(args.train_epochs):
            for mb in self._rnn_minibatch_iter(rnn_data):
                L_, N, obs_dim = mb["obs"].shape   # obs (L, N, D) — no agent dim

                mb_obs_flat   = mb["obs"].reshape(L_ * N, obs_dim)   # (L*N, D)
                mb_act_flat   = mb["actions"].reshape(L_ * N, K)      # (L*N, K)
                mb_lp_flat    = mb["log_probs"].reshape(L_ * N)       # (L*N,)
                if self.use_avail:
                    mb_avail_flat = mb["avail"].reshape(L_ * N, -1)   # (L*N, act_D)
                else:
                    mb_avail_flat = None
                mb_active_flat = mb["active_masks"].reshape(L_ * N)   # (L*N,)
                mb_adv_flat   = mb["advantages"].reshape(L_ * N)      # (L*N,)
                mb_rnn_masks  = mb["rnn_masks"].reshape(L_ * N, 1)    # (L*N, 1)

                # ---- actor ----
                # hxs_actor (N, L_rnn, H_a) — one stream per chunk (one agent per chunk)
                if args.shared_policies:
                    _, logprob, entropy, _ = self.actors["actor_0"].act(
                        mb_obs_flat,
                        avail_action=mb_avail_flat,
                        action=mb_act_flat,
                        rnn_hxs=mb["hxs_actor"],
                        masks=mb_rnn_masks,
                    )
                else:
                    # Dispatch each chunk to its agent's actor
                    agent_indices = mb["agent_indices"]  # (N,) GPU tensor
                    logprob = torch.zeros(L_ * N, device=self.device)
                    entropy = torch.zeros(L_ * N, device=self.device)
                    for a_idx, actor in enumerate(self.actors.values()):
                        sel_t = (agent_indices == a_idx).nonzero(as_tuple=False).squeeze(1)
                        if sel_t.numel() == 0:
                            continue
                        # rows for chunk i at times 0..L-1: t*N + i
                        rows = (sel_t.unsqueeze(0) + torch.arange(L_, device=self.device).unsqueeze(1) * N).reshape(-1)
                        avail_rows = mb_avail_flat[rows] if mb_avail_flat is not None else None
                        _, lp_i, ent_i, _ = actor.act(
                            mb_obs_flat[rows],
                            avail_action=avail_rows,
                            action=mb_act_flat[rows],
                            rnn_hxs=mb["hxs_actor"][sel_t],
                            masks=mb_rnn_masks[rows],
                        )
                        logprob[rows] = lp_i
                        entropy[rows] = ent_i

                log_ratio = logprob - mb_lp_flat
                ratio = log_ratio.exp()
                pg_loss = self._policy_loss(ratio, mb_adv_flat, log_ratio)
                if args.policy_mask:
                    actor_pg    = self._masked_mean(pg_loss, mb_active_flat)
                    entropy_loss = self._masked_mean(entropy, mb_active_flat)
                else:
                    actor_pg    = pg_loss.mean()
                    entropy_loss = entropy.mean()
                actor_loss     = actor_pg - entropy_coef * entropy_loss
                kl_divergence  = ((ratio - 1) - log_ratio).mean()
                clipped_ratio  = ((ratio - 1.0).abs() > args.trust_region_clip).float().mean()

                # ---- critic ----
                mb_states_flat = mb["states"].reshape(L_ * N, -1)    # (L*N, state_D)
                mb_ret_flat    = mb["returns"].reshape(L_ * N)        # (L*N,)
                mb_oldv_flat   = mb["old_values"].reshape(L_ * N)     # (L*N,)

                if args.centralized_critic:
                    # Chunks from the same (env, chunk_start) have identical state and hidden
                    # (different agents, identical centralized streams) — dedup before critic forward.
                    T_buf = rnn_data.time_mask.shape[0]
                    flat_key = mb["env_indices"].long() * T_buf + mb["time_indices"].long()
                    unique_keys, inverse = torch.unique(flat_key, return_inverse=True)
                    n_unique = unique_keys.shape[0]
                    representatives = torch.empty(n_unique, dtype=torch.long, device=self.device)
                    representatives.scatter_(0, inverse, self._rnn_mb_arange[:N])
                    states_unique = mb["states"][:, representatives]          # (L, n_unique, state_D)
                    hxs_unique    = mb["hxs_critic"][representatives]         # (n_unique, L_rnn, H)
                    masks_unique  = mb["rnn_masks"][:, representatives]       # (L, n_unique, 1)
                    vals_u, _ = self.critic(
                        states_unique.reshape(L_ * n_unique, -1),
                        rnn_hxs=hxs_unique,
                        masks=masks_unique.reshape(L_ * n_unique, 1),
                    )
                    values_grid = vals_u.squeeze(-1).reshape(L_, n_unique)[:, inverse]  # (L, N)
                    current_values = values_grid.reshape(L_ * N)
                else:
                    values, _ = self.critic(mb_states_flat, rnn_hxs=mb["hxs_critic"], masks=mb_rnn_masks)
                    current_values = values.squeeze(-1)

                # active_masks (L*N,) → unsqueeze for _critic_loss's any(dim=-1)
                critic_loss = self._critic_loss(
                    current_values, mb_ret_flat, mb_oldv_flat, mb_active_flat.unsqueeze(-1)
                )

                self._gradient_step(actor_loss, critic_loss, metrics)
                metrics.actor_losses.append(actor_loss.item())
                metrics.critic_losses.append(critic_loss.item())
                metrics.entropy_bonuses.append(entropy_loss.item())
                metrics.kl_divergences.append(kl_divergence.item())
                metrics.clipped_ratios.append(clipped_ratio.item())

        return metrics
