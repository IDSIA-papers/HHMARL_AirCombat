import os
import glob
import torch
import tyro
import numpy as np
from args import Args
from algorithm.networks import build_actor_critic, act_with_policies
from env.env import AircombatMARL
from train import process_args

def load_checkpoint(restore_path, device):
    path = restore_path
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "checkpoint*.pt")))
        assert candidates, f"No checkpoints found in {path}"
        path = candidates[-1]
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {path}  (trained to step {ckpt['step']})")
    return ckpt

RENDER = True

def run_inference(num_episodes=500):
    args, _, device = process_args(tyro.cli(Args))
    
    if args.hierarchy_level == 2:
        args.level = 1
        args.task = "hier"
    else:
        args.hierarchy_level = 1
        args.task = f"{args.task}/L{args.level}"

    kwargs = {"hierarchy_level":args.hierarchy_level, "training_level": args.level+int(args.task == "track" or args.task == "escape" or args.hierarchy_level == 2), 
              "rendering": RENDER, "render_view": 0}

    assert args.restore_path is not None, "Set --restore_path to a checkpoint file or directory."

    env = AircombatMARL(task=args.task, **kwargs)
    actors, _ = build_actor_critic(
        args, env.get_obs_size(), env.get_action_space(),
        env.get_state_size(), env.n_agents, device,
    )
    actors.load_state_dict(load_checkpoint(args.restore_path, device)["actors_state_dict"])
    actors.eval()

    agent_keys = env.agent_keys
    A = env.n_agents
    L, H = args.rnn_num_layers, args.actor_hidden_dim

    ep_rewards, ep_lengths = [], []
    wins = losses = 0
    for ep in range(num_episodes):
        obs_dict, _ = env.reset(seed=args.seed + ep)
        obs = np.array([obs_dict[k] for k in agent_keys], dtype=np.float32)

        rnn_hxs = torch.zeros(1, A, L, H, device=device) if args.rnn else None
        masks = torch.ones(1, device=device) if args.rnn else None

        ep_reward, ep_len, done = 0.0, 0, False
        while not done:
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)  # (1, A, D)
            with torch.no_grad():
                actions, _, _, rnn_hxs = act_with_policies(
                    args, actors, obs_t, rnn_hxs=rnn_hxs, masks=masks, deterministic=True,
                )
            actions_np = actions[0].cpu().numpy()  # (A, K)
            actions_dict = {k: actions_np[i] for i, k in enumerate(agent_keys)}

            obs_dict, reward_dict, done_dict, trunc, info = env.step(actions_dict)
            obs = np.array([obs_dict[k] for k in agent_keys], dtype=np.float32)
            ep_reward += float(np.mean([reward_dict[k] for k in agent_keys]))
            ep_len += 1
            done = done_dict["all"] or trunc["all"]

            if RENDER and not args.hierarchy_level == 2:
                env.env.render()

        agents = env.env.agents
        n_killed_opp = sum(not agents[aid].alive for aid in env.env.opp_agent_ids)
        n_killed_agent = sum(not agents[aid].alive for aid in env.env.rl_agent_ids)

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)
        n_opp = len(env.env.opp_agent_ids)
        n_rl = len(env.env.rl_agent_ids)
        print(f"Episode {ep + 1:3d} | reward: {ep_reward:8.3f} | length: {ep_len:4d} | killed opp: {n_killed_opp} | killed agent: {n_killed_agent}")
        if n_killed_opp == n_opp:
            wins += 1
        if n_killed_agent == n_rl:
            losses += 1

    print(f"\nMean reward: {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}  "
          f"| Mean length: {np.mean(ep_lengths):.1f}")
    print(f"Mean Wins: {wins/num_episodes:.3f} | Mean Losses: {losses/num_episodes:.3f}")
    env.close()


if __name__ == "__main__":
    run_inference()
