import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from algorithm.valuenorm import PopArt


def layer_init(layer, std=np.sqrt(2), bias_const=0.0, orthogonal=True):
    if orthogonal:
        torch.nn.init.orthogonal_(layer.weight, std)
    else:
        torch.nn.init.xavier_uniform_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AttentionLayer(nn.Module):
    """Cross-agent MultiHeadAttention with residual + LayerNorm.

    Inputs are (E, A, H) — applies self-attention across the agent axis A so each
    agent's features mix with the other agents' features at the same timestep.
    Returns a tensor of the same shape. Stateless (no per-step carry like rnn_hxs).
    """

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (E, A, H). Self-attention across agents.
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        return self.norm(x + attn_out)


class RNNLayer(nn.Module):
    """GRU with LayerNorm, adapted from onpolicy. hxs convention: (N, num_layers, H) externally."""

    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super().__init__()
        self._recurrent_N = recurrent_N
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_N)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        # hxs: (N, recurrent_N, H)  masks: (N, 1) single-step or (T*N, 1) sequence
        if x.size(0) == hxs.size(0):
            # Single-step inference: x (N, D)
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (hxs * masks.unsqueeze(-1)).transpose(0, 1).contiguous(),
            )
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # Sequence training: x (T*N, D), hxs (N, recurrent_N, H)
            N = hxs.size(0)
            T = x.size(0) // N
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)

            has_zeros = (
                (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            )
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (
                    hxs
                    * masks[start_idx]
                    .view(1, -1, 1)
                    .repeat(self._recurrent_N, 1, 1)
                ).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0).reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs


class Actor(nn.Module):
    """
    Flexible actor supporting three action-space types:
      - discrete:       action_space = ("discrete", n)            -> action width K=1
      - multi_discrete: action_space = ("multi_discrete", [n1,..]) -> action width K=len(nvec)
      - continuous:     action_space = ("continuous", dim)         -> action width K=dim

    act() returns (action, log_prob, entropy, rnn_hxs) where
      action:    (N, K) — long for discrete/multi_discrete, float for continuous
      log_prob:  (N,)   — summed across branches/dims
      entropy:   (N,)   — summed across branches/dims
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layer,
        action_space,
        use_relu=False,
        normalize_input=False,
        init_orthogonal=True,
        use_rnn=False,
        rnn_num_layers=1,
        use_attention=False,
        attention_heads=4,
    ) -> None:
        super().__init__()
        self.action_space = action_space
        self.space_type = action_space[0]
        self.use_rnn = use_rnn
        self.use_attention = use_attention
        assert not (use_rnn and use_attention), "Set at most one of rnn / attention."
        act_fn = nn.ReLU if use_relu else nn.Tanh
        gain = nn.init.calculate_gain("relu" if use_relu else "tanh")
        base_layers = []
        if normalize_input:
            base_layers.append(nn.LayerNorm(input_dim))
        base_layers += [
            layer_init(nn.Linear(input_dim, hidden_dim), std=gain, orthogonal=init_orthogonal),
            act_fn(),
            nn.LayerNorm(hidden_dim),
        ]
        for _ in range(num_layer):
            base_layers.extend([
                layer_init(nn.Linear(hidden_dim, hidden_dim), std=gain, orthogonal=init_orthogonal),
                act_fn(),
                nn.LayerNorm(hidden_dim),
            ])
        self.base = nn.Sequential(*base_layers)
        if use_rnn:
            self.rnn = RNNLayer(hidden_dim, hidden_dim, rnn_num_layers, init_orthogonal)
        if use_attention:
            self.attention = AttentionLayer(hidden_dim, attention_heads)

        if self.space_type == "discrete":
            n = int(action_space[1])
            self.action_width = 1
            self.output = layer_init(nn.Linear(hidden_dim, n), std=0.1, orthogonal=init_orthogonal)
        elif self.space_type == "multi_discrete":
            nvec = [int(n) for n in action_space[1]]
            self.action_width = len(nvec)
            self.output = nn.ModuleList([
                layer_init(nn.Linear(hidden_dim, n), std=0.1, orthogonal=init_orthogonal)
                for n in nvec
            ])
        elif self.space_type == "continuous":
            dim = int(action_space[1])
            self.action_width = dim
            self.output = layer_init(nn.Linear(hidden_dim, dim), std=0.1, orthogonal=init_orthogonal)
            # State-independent learned log_std, init 0 => std=1 (matches onpolicy DiagGaussian).
            self.log_std = nn.Parameter(torch.zeros(dim))
        else:
            raise ValueError(f"Unknown action space type: {self.space_type}")

    def act(self, x, avail_action=None, action=None, rnn_hxs=None, masks=None, deterministic=False):
        """Returns (action, log_prob, entropy, new_rnn_hxs).

        When use_attention is False:
          x: (N, D); outputs are (N, K) / (N,) / (N,).
        When use_attention is True:
          x: (N, A, D); base + MHA run on the (N, A) grid, then heads see (N*A, H).
          Outputs are reshaped back to (N, A, K) / (N, A) / (N, A).
        """
        if self.use_attention:
            # x: (N, A, D). Flatten avail/action over the agent axis so head logic stays
            # shape-agnostic; we reshape outputs back to (N, A, ...) before returning.
            N, A, D = x.shape
            x = self.base(x.reshape(N * A, D)).reshape(N, A, -1)
            x = self.attention(x)
            x = x.reshape(N * A, -1)
            if avail_action is not None:
                avail_action = avail_action.reshape(N * A, -1)
            if action is not None:
                action = action.reshape(N * A, -1) if action.dim() > 2 else action.reshape(N * A)
        else:
            x = self.base(x)
            if self.use_rnn:
                x, rnn_hxs = self.rnn(x, rnn_hxs, masks)

        def _unflatten(action_t, log_prob, entropy):
            """If use_attention, restore the agent axis on outputs before returning."""
            if not self.use_attention:
                return action_t, log_prob, entropy
            K = action_t.size(-1)
            return (
                action_t.reshape(N, A, K),
                log_prob.reshape(N, A),
                entropy.reshape(N, A),
            )

        if self.space_type == "discrete":
            logits = self.output(x)
            if avail_action is not None:
                logits = logits.masked_fill(~avail_action, -1e9)
            dist = Categorical(logits=logits)
            if action is None:
                sampled = logits.argmax(dim=-1) if deterministic else dist.sample()
            else:
                # action may be (N,) or (N, 1); squeeze trailing dim for Categorical.
                sampled = action.squeeze(-1) if action.dim() > 1 else action
            log_prob = dist.log_prob(sampled)
            entropy = dist.entropy()
            action_out, log_prob, entropy = _unflatten(sampled.unsqueeze(-1), log_prob, entropy)
            return action_out, log_prob, entropy, rnn_hxs

        if self.space_type == "multi_discrete":
            K = self.action_width
            if action is None:
                samples = []
                lps = []
                ents = []
                for head in self.output:
                    logits = head(x)
                    dist = Categorical(logits=logits)
                    s = logits.argmax(dim=-1) if deterministic else dist.sample()
                    samples.append(s)
                    lps.append(dist.log_prob(s))
                    ents.append(dist.entropy())
                action_out = torch.stack(samples, dim=-1)  # (N, K)
                log_prob = torch.stack(lps, dim=-1).sum(dim=-1)
                entropy = torch.stack(ents, dim=-1).sum(dim=-1)
            else:
                lps = []
                ents = []
                for k, head in enumerate(self.output):
                    logits = head(x)
                    dist = Categorical(logits=logits)
                    s_k = action[..., k].long()
                    lps.append(dist.log_prob(s_k))
                    ents.append(dist.entropy())
                action_out = action
                log_prob = torch.stack(lps, dim=-1).sum(dim=-1)
                entropy = torch.stack(ents, dim=-1).sum(dim=-1)
            action_out, log_prob, entropy = _unflatten(action_out, log_prob, entropy)
            return action_out, log_prob, entropy, rnn_hxs

        # continuous
        mean = self.output(x)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        action, log_prob, entropy = _unflatten(action, log_prob, entropy)
        return action, log_prob, entropy, rnn_hxs


class Critic(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layer,
        use_relu=False,
        normalize_input=False,
        init_orthogonal=True,
        normalize_popart=False,
        device=torch.device("cpu"),
        use_rnn=False,
        rnn_num_layers=1,
        use_attention=False,
        attention_heads=4,
    ) -> None:
        super().__init__()
        self.use_rnn = use_rnn
        self.use_attention = use_attention
        assert not (use_rnn and use_attention), "Set at most one of rnn / attention."
        act_fn = nn.ReLU if use_relu else nn.Tanh
        gain = nn.init.calculate_gain("relu" if use_relu else "tanh")
        base_layers = []
        if normalize_input:
            base_layers.append(nn.LayerNorm(input_dim))
        base_layers += [
            layer_init(nn.Linear(input_dim, hidden_dim), std=gain, orthogonal=init_orthogonal),
            act_fn(),
            nn.LayerNorm(hidden_dim),
        ]
        for _ in range(num_layer):
            base_layers.extend([
                layer_init(nn.Linear(hidden_dim, hidden_dim), std=gain, orthogonal=init_orthogonal),
                act_fn(),
                nn.LayerNorm(hidden_dim),
            ])
        self.base = nn.Sequential(*base_layers)
        if use_rnn:
            self.rnn = RNNLayer(hidden_dim, hidden_dim, rnn_num_layers, init_orthogonal)
        if use_attention:
            self.attention = AttentionLayer(hidden_dim, attention_heads)
        if normalize_popart:
            self.v_out = layer_init(PopArt(hidden_dim, 1, device=device), std=1.0, orthogonal=init_orthogonal)
        else:
            self.v_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0, orthogonal=init_orthogonal)

    def forward(self, x, rnn_hxs=None, masks=None):
        """Returns (values, new_rnn_hxs).

        When use_attention is True the caller must pass x as (N, A, D); output is (N, A, 1).
        Otherwise x is (N, D) and output is (N, 1). new_rnn_hxs is None unless use_rnn.
        """
        if self.use_attention:
            N, A, D = x.shape
            x = self.base(x.reshape(N * A, D)).reshape(N, A, -1)
            x = self.attention(x)
            return self.v_out(x), rnn_hxs
        x = self.base(x)
        if self.use_rnn:
            x, rnn_hxs = self.rnn(x, rnn_hxs, masks)
        return self.v_out(x), rnn_hxs


def build_actor_critic(args, obs_dim, action_space, state_dim, num_agents, device):
    # Critic always consumes a state vector. centralized_critic=True -> shared global state
    # per env; centralized_critic=False -> per-agent individual state. Input width is
    # state_dim either way (per-agent dim equals shared dim by construction).

    use_attention = getattr(args, "attention", False)
    if use_attention:
        assert not args.rnn, "attention and rnn are mutually exclusive."
        assert args.shared_policies, "attention requires shared_policies=True (one actor processes the agent set jointly)."

    # The critic only gets attention when its input has an agent axis (IPPO state critic).
    # With centralized_critic=True the critic consumes a single global state per env, so MHA
    # over agents has nothing to attend to — fall back to plain MLP for the critic.
    
    critic_use_attention = use_attention and not args.centralized_critic
    attention_heads = getattr(args, "attention_heads", 4)

    critic_input_dim = state_dim
    actor_keys = ["actor_0"] if args.shared_policies else [f"actor_{i}" for i in range(num_agents)]
    actors = nn.ModuleDict({
        key: Actor(
            input_dim=obs_dim,
            hidden_dim=args.actor_hidden_dim,
            num_layer=args.actor_num_layers,
            action_space=action_space,
            use_relu=args.use_ReLU,
            normalize_input=args.normalize_input,
            init_orthogonal=args.init_orthogonal,
            use_rnn=args.rnn,
            rnn_num_layers=args.rnn_num_layers,
            use_attention=use_attention,
            attention_heads=attention_heads,
        )
        for key in actor_keys
    }).to(device)
    critic = Critic(
        input_dim=critic_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        use_relu=args.use_ReLU,
        normalize_input=args.normalize_input,
        init_orthogonal=args.init_orthogonal,
        normalize_popart=args.normalize_popart,
        device=device,
        use_rnn=args.rnn,
        rnn_num_layers=args.rnn_num_layers,
        use_attention=critic_use_attention,
        attention_heads=attention_heads,
    ).to(device)
    return actors, critic


def act_with_policies(args, actors, obs, avail_actions=None, action=None, rnn_hxs=None, masks=None, deterministic=False):
    """
    Run actors for one rollout step.

    obs: (E, A, obs_dim)
    avail_actions: (E, A, act_dim) bool — discrete only; None for multi_discrete/continuous.
    action: (E, A, K) or None
    rnn_hxs: (E, A, rnn_num_layers, H) or None
    masks: (E,) float — 1=continue, 0=episode-start; broadcast per agent; or None
    deterministic: if True, use argmax/mean instead of sampling.

    Returns: (actions, log_probs, entropy, new_rnn_hxs)
      shapes (E, A, K), (E, A), (E, A), (E, A, L, H) or None
    """
    num_envs, num_agents, obs_dim = obs.shape

    if args.shared_policies:
        actor = actors["actor_0"]
        if getattr(actor, "use_attention", False):
            # Attention path: keep the agent axis intact so the actor can attend across it.
            out_action, log_prob, entropy, _ = actor.act(
                obs, avail_action=avail_actions, action=action,
                rnn_hxs=None, masks=None, deterministic=deterministic,
            )
            return out_action, log_prob, entropy, None

        N = num_envs * num_agents
        flat_obs = obs.reshape(N, obs_dim)
        flat_avail = None
        if avail_actions is not None:
            flat_avail = avail_actions.reshape(N, avail_actions.size(-1))
        flat_action = None
        if action is not None:
            K = action.size(-1)
            flat_action = action.reshape(N, K)

        flat_hxs, flat_masks = None, None
        if rnn_hxs is not None:
            L_rnn, H = rnn_hxs.shape[2], rnn_hxs.shape[3]
            flat_hxs = rnn_hxs.reshape(N, L_rnn, H)
            # masks (E,) → per-agent: (E, A) → (E*A, 1)
            flat_masks = masks.unsqueeze(1).expand(num_envs, num_agents).reshape(N, 1)

        out_action, log_prob, entropy, new_flat_hxs = actor.act(
            flat_obs, avail_action=flat_avail, action=flat_action,
            rnn_hxs=flat_hxs, masks=flat_masks, deterministic=deterministic,
        )
        K = out_action.size(-1)
        new_hxs = None
        if new_flat_hxs is not None:
            new_hxs = new_flat_hxs.reshape(num_envs, num_agents, L_rnn, H)
        return (
            out_action.reshape(num_envs, num_agents, K),
            log_prob.reshape(num_envs, num_agents),
            entropy.reshape(num_envs, num_agents),
            new_hxs,
        )

    act_list, lp_list, ent_list, hxs_list = [], [], [], []
    for agent_idx, actor in enumerate(actors.values()):
        action_i = None if action is None else action[:, agent_idx]
        avail_i = None if avail_actions is None else avail_actions[:, agent_idx]
        hxs_i, masks_i = None, None
        if rnn_hxs is not None:
            L_rnn, H = rnn_hxs.shape[2], rnn_hxs.shape[3]
            hxs_i = rnn_hxs[:, agent_idx]           # (E, L_rnn, H)
            masks_i = masks.unsqueeze(1)             # (E, 1)
        out_action, log_prob, entropy, new_hxs_i = actor.act(
            obs[:, agent_idx],
            avail_action=avail_i,
            action=action_i,
            rnn_hxs=hxs_i,
            masks=masks_i,
            deterministic=deterministic,
        )
        act_list.append(out_action)
        lp_list.append(log_prob)
        ent_list.append(entropy)
        hxs_list.append(new_hxs_i)

    new_hxs = None
    if hxs_list[0] is not None:
        new_hxs = torch.stack(hxs_list, dim=1)  # (E, A, L_rnn, H)
    return (
        torch.stack(act_list, dim=1),   # (E, A, K)
        torch.stack(lp_list, dim=1),    # (E, A)
        torch.stack(ent_list, dim=1),   # (E, A)
        new_hxs,
    )
