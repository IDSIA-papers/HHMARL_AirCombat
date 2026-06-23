import os
import time
import math
import torch
import tyro
import random
import numpy as np
from tqdm import tqdm
from args import Args
from algorithm.learner import Learner
from algorithm.utils import validate_decay_schedule, get_decayed_value, Logger, Workers
from env.utils import parse_config

"""
The main file for running experiments on AircraftEnv.
"""

def entropy_decay_schedule(args):
    if args.level >= 3 and args.task.startswith("track") or args.level >= 3 and args.task.startswith("cannon") or args.level >= 2 and args.task.startswith("escape") or args.hierarchy_level == 2:
        args.entropy_coef = 0
        return None
    else:
        return [[30000000 * (args.level-1), 0.07], [30000000 * (args.level-1) + 2e6, 0]]

def process_args(args):
    if args.hierarchy_level == 2:
        args.batch_size = 5000
        args.minibatch_size = 1000
        args.level = 1
        args.task = "hier"
        args.total_timesteps = 3000000
    else:
        args.batch_size = 10000
        args.minibatch_size = 5000
        args.hierarchy_level = 1
        args.task = f"{args.task}/L{args.level}"
        args.total_timesteps = 30000000 * args.level

    args.entropy_decay = entropy_decay_schedule(args)
    args.cuda_id = 0

    batch_alignment = math.lcm(args.num_envs, args.minibatch_size)
    args.batch_size = int(math.ceil(args.batch_size / batch_alignment) * batch_alignment)
    args.lr_decay_schedule = validate_decay_schedule(args.lr_decay, "lr_decay")
    args.entropy_decay_schedule = validate_decay_schedule(args.entropy_decay, "entropy_decay")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.set_device(args.cuda_id)
        device = torch.device(f"cuda:{args.cuda_id}")
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.benchmark = not args.torch_deterministic
    torch.backends.cudnn.deterministic = args.torch_deterministic

    kwargs = {"hierarchy_level":args.hierarchy_level, "training_level": args.level+int(args.task == "track" or args.task == "escape" or args.hierarchy_level == 2), "rendering": False}
    return args, kwargs, device


def restore_checkpoint(args, device, learner):
    """Load a checkpoint to resume training. Returns the step to resume from."""
    restore = args.restore_path
    if os.path.isdir(restore):
        restore = os.path.join(restore, "checkpoint.pt")
    assert os.path.isfile(restore), f"No checkpoint found at {restore}"
    ckpt = torch.load(restore, map_location=device)
    learner.actors.load_state_dict(ckpt["actors_state_dict"])
    learner.critic.load_state_dict(ckpt["critic_state_dict"])
    learner.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
    learner.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
    if ckpt["value_normalizer_state_dict"] is not None and learner.value_normalizer is not None:
        learner.value_normalizer.load_state_dict(ckpt["value_normalizer_state_dict"])
    print(f"Restored checkpoint from {restore} at step {ckpt['step']}")
    return ckpt["step"]


if __name__ == "__main__":
    args, kwargs, device = process_args(tyro.cli(Args))

    workers = Workers(args, kwargs)
    learner = Learner(args, device, workers)
    task_name = args.task.replace("/", "_")
    # Heterogeneous hierarchical combat yields model name "f16_a4", otherwise the model name is just the model of the aircraft in the task.
    models = list(dict.fromkeys(
        cfg.get("model", "f16")
        for cfg in parse_config(args.task).aircraft.values()
        if str(cfg.get("mission", "")).startswith("task")
    ))
    model = "_".join(models) if models else "f16"
    log_name = f"runs/{task_name}_{model}_{args.run_name}" if args.run_name else f"runs/{task_name}_{model}"
    logger = Logger(log_dir=log_name, args=args)

    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    num_episodes = 0
    update_step = 0
    step = restore_checkpoint(args, device, learner) if args.restore_path is not None else 0
    iteration_times = []

    progress_bar = tqdm(total=args.total_timesteps, initial=step, desc=f"train {task_name}_{args.run_name}", dynamic_ncols=True)

    while step < args.total_timesteps:
        iteration_start = time.time()

        # 1. Update scheduled hyperparameters
        current_lr = get_decayed_value(step, args.learning_rate_actor, args.lr_decay_schedule)
        current_critic_lr = get_decayed_value(step, args.learning_rate_critic, args.lr_decay_schedule)
        current_entropy_coef = get_decayed_value(step, args.entropy_coef, args.entropy_decay_schedule)
        learner.set_learning_rates(current_lr, current_critic_lr)

        # 2. Collect rollout
        rollout = learner.collect_rollout(step)
        step += rollout.steps_collected
        progress_bar.update(rollout.steps_collected)

        ep_rewards.extend(rollout.ep_rewards)
        ep_lengths.extend(rollout.ep_lengths)
        ep_stats.extend(rollout.ep_stats)
        num_episodes += rollout.num_new_episodes

        if logger.log_episode_stats(ep_rewards, ep_lengths, ep_stats, num_episodes, step):
            ep_rewards, ep_lengths, ep_stats = [], [], []
            logger.save_checkpoint(learner, step)

        # 3. Compute GAE advantages and flatten batch
        flat_data = learner.compute_advantages()

        # 4. Train actor and critic on the collected batch
        metrics = learner.train(flat_data, current_entropy_coef)
        update_step += 1

        logger.log_train_stats(
            metrics, learner.training_step, update_step,
            current_lr, current_critic_lr, current_entropy_coef, step
        )

        iteration_time = time.time() - iteration_start
        iteration_times.append(iteration_time)
        mean_reward = (
            float(np.mean(rollout.iteration_rewards))
            if rollout.iteration_rewards
            else float(np.mean(learner._current_ep_rewards))
        )
        progress_bar.set_postfix(
            training_step=learner.training_step,
            mean_reward=f"{mean_reward:.3f}",
            mean_iter_s=f"{float(np.mean(iteration_times)):.3f}",
        )

    progress_bar.close()
    workers.close()
    logger.close()
