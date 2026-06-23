from dataclasses import dataclass, field
from datetime import datetime


def default_run_name():
    now = datetime.now()
    return f"_{now.year % 100}.{now.month}.{now.day}-{now.hour:02d}:{now.minute:02d}"


def schedule_field(default=None):
    if default is None:
        return field(default=None)
    return field(
        default_factory=lambda default=default: [list(pair) for pair in default]
    )


@dataclass
class Args:
    task: str = "track"
    """ Task of the environment"""
    run_name: str = field(default="") #field(default_factory=default_run_name)
    """ Run name used by training logs and progress output """
    level: int = 1
    """ level of task. """
    hierarchy_level: int = 1
    """ Hierarchical Level (1 = low-level control training, 2 = high-level commander training) """
    policy_mask: bool = True
    """ If True, mask dead agents out of the actor loss (active masks on policy) """
    value_mask: bool = True
    """ If True, mask dead agents out of the critic loss (active masks on value) """
    actor_hidden_dim: int = 128
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    shared_policies: bool = True
    """ Share one actor across all agents; otherwise one actor per agent"""
    centralized_critic: bool = True
    """ If True, critic takes global state as input (MAPPO); if False, takes each agent's local obs (IPPO) """
    critic_hidden_dim: int = 128
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 3e-4
    """ Learning rate for the actor"""
    learning_rate_critic: float = 3e-4
    """ Learning rate for the critic"""
    lr_decay: list[list[float]] | None = schedule_field()
    """ Optional decay schedule in the form [[t0, l0], [t1, l1]] """
    total_timesteps: int = 5000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.92
    """ TD(λ) discount factor"""
    normalize_advantage: bool = True
    """ Normalize the advantage if True"""
    normalize_value: bool = True
    """ If True, use running EMA normalization (ValueNorm) for value targets """
    normalize_popart: bool = False
    """ If True, use PopArt adaptive normalization (replaces critic output layer); mutually exclusive with normalize_value """
    num_envs: int = 20
    """ Number of parallel environment workers collecting each rollout"""
    minibatch_size: int = 5000
    """ Number of samples in each SGD minibatch"""
    batch_size: int = 10000
    """ Lower bound for the full rollout batch, rounded up for clean minibatch/env splits"""
    train_epochs: int = 10
    """ Number of training epochs"""
    algorithm: str = "ppo"
    """ Policy optimization algorithm: "ppo" (clipped surrogate), "spo" (soft quadratic penalty), or "mars" (multiplicatively symmetric geometric barrier) """
    trust_region_clip: float = 0.25
    """ Trust region parameter ε. For PPO: clipping factor. For SPO: penalty scale. For MARS: B_upper = 1+ε, B_lower = 1/B_upper. """
    clip_value_loss: float = 1.0
    """ If >0, clip value targets at ±this amount from old values; ≤0 disables clipping """
    value_loss_coef: float = 1.0
    """ Coefficient applied to the critic loss """
    huber_coef: float = 10.0
    """ If >0, use Huber loss with this delta for the critic; ≤0 uses plain MSE """
    use_ReLU: bool = True
    """ If True use ReLU activations in actor/critic, else Tanh """
    normalize_input: bool = True
    """ If True, prepend LayerNorm to actor and critic inputs """
    init_orthogonal: bool = True
    """ If True, use orthogonal weight init; if False, use Xavier uniform """
    use_gae: bool = True
    """ If True compute GAE advantages; False uses plain discounted returns (td_lambda=1) """
    entropy_coef: float = 0.01
    """ Entropy coefficient """
    entropy_decay: list[list[float]] | None = schedule_field()
    """ Optional decay schedule in the form [[t0, l0], [t1, l1]] """
    lr_decay_schedule: tuple | None = field(init=False, default=None)
    """ Validated decay schedule derived from lr_decay; set by custom_args """
    entropy_decay_schedule: tuple | None = field(init=False, default=None)
    """ Validated decay schedule derived from entropy_decay; set by custom_args """
    log_every_step: int = 100
    """ Logging episode steps """
    clip_gradients: float = 10
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    cuda_id: int = 0
    """ CUDA device index to use when cuda is enabled """
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic` is set for reproducibility."""
    seed: int = 1
    """ Random seed"""
    rnn: bool = False
    """ If True, insert a GRU layer in the actor and critic after the MLP base """
    rnn_num_layers: int = 1
    """ Number of recurrent layers in the GRU (recurrent_N in onpolicy) """
    data_chunk_length: int = 10
    """ Sequence chunk length for RNN training mini-batches """
    attention: bool = False
    """ If True, insert a cross-agent MultiHeadAttention layer in the actor (and IPPO critic) in place of the GRU. Mutually exclusive with rnn. Requires shared_policies=True. Centralized critic stays plain MLP. """
    attention_heads: int = 2
    """ Number of attention heads. Must divide actor_hidden_dim (and critic_hidden_dim when used). """
    avail_actions: bool = False
    """ If True, collect and mask with available actions; if False, skip entirely (e.g. GRF where all actions are always available) """
    torch_compile: bool = False
    """ If True, apply torch.compile to actors' .act and critic.forward (dynamic shapes). Disable if compilation regresses on your hardware. """
    restore_path: str | None = None
    """ Path to a checkpoint file or directory (loads the latest checkpoint) to resume training. None disables restore. """
