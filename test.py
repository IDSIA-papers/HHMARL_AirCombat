import os
import glob
import torch
import tyro
import numpy as np
from args import Args
from algorithm.networks import build_actor_critic, act_with_policies
from env.env import AircombatMARL
from rendering.flightsim import FlightSimClosed
from train import process_args

# The policy is selected on the command line, as for training, e.g.
#   python test.py --task cannon --level 4 --restore_path runs/cannon_L4_f16
#   python test.py --hierarchy_level 2 --restore_path runs/hier_<run_name>
RENDER = True
RENDERER = "pygame"  # "pygame": simple rendering of this repository, "flightsim": FlightSim (see README)

# Initial states: None for the states of the config (random if rnd_init), or per aircraft relative to the
# battle field center: "north"/"east" in m (or "lat"/"lon" in degrees), "alt" in m, "speed" in m/s,
# "heading" in degrees, and optionally "roll"/"pitch" in degrees. Aircraft without an entry use the config.
# Example: B1 starts 4 km south of the center at 7500 m above sea level, heading north (0 deg) at 250 m/s;
# R2 starts 4 km north and 1 km east of the center at 7000 m above sea level, heading south (180 deg) at 220 m/s.
INIT_STATES = None
# INIT_STATES = {"B1": {"north": -4000, "east": 0, "alt": 7500, "heading": 0, "speed": 250},     # m, m, m, deg, m/s
#                "R2": {"north": 4000, "east": 1000, "alt": 7000, "heading": 180, "speed": 220}}  # m, m, m, deg, m/s

# FlightSim settings (RENDER = True and RENDERER = "flightsim")
JSBSIM_STATES = "python"  # "python": the JSBSim of this env simulates the aircraft, "c++": the JSBSim of FlightSim
MANUAL = False            # True: a human flies MANUAL_AGENT with the joysticks or the keyboard, as teammate of the blue RL agents
MANUAL_AGENT = "B2"       # added as copy of B1 if the config has no such aircraft (track/cannon/escape), replaces it otherwise (hier: B3)
LOCATION = (8.965318, 45.975305)  # (longitude, latitude) of the battle field center in degrees, here Lugano
MINIMAP = True            # map of the aircraft around the camera target in the bottom right corner (key M toggles it, N north-up/heading-up)
MINIMAP_RANGE = 20000     # m, radius of the minimap; aircraft farther away are shown at its edge with their distance
WEZ = True                # cockpit/follow view (F1/F2): green circle if the camera target can hit an opponent, red beam if an opponent can hit it (key Z)

def load_checkpoint(restore_path, device):
    path = restore_path
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "checkpoint*.pt")))
        assert candidates, f"No checkpoints found in {path}"
        path = candidates[-1]
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {path}  (trained to step {ckpt['step']})")
    return ckpt

def run_inference(num_episodes=500):
    # process_args sets args.task to "<task>/L<level>" (hierarchy_level 1) or "hier" (hierarchy_level 2)
    # and returns the env kwargs of training (hierarchy_level, training_level, render_samples)
    args, kwargs, device = process_args(tyro.cli(Args))
    kwargs.update({"rendering": RENDER, "render_view": 0})

    assert RENDERER in ("pygame", "flightsim"), f"Unknown RENDERER {RENDERER}"
    use_flightsim = RENDER and RENDERER == "flightsim"
    assert use_flightsim or (not MANUAL and JSBSIM_STATES == "python"), \
        "MANUAL and JSBSIM_STATES = 'c++' need RENDER = True and RENDERER = 'flightsim'"
    if use_flightsim:
        kwargs["flightsim"] = {"jsbsim_states": JSBSIM_STATES, "manual_agent": MANUAL_AGENT if MANUAL else None,
                               "minimap": MINIMAP, "minimap_range": MINIMAP_RANGE, "wez": WEZ}
        kwargs["config_overrides"] = {"battle_field_center": [LOCATION[0], LOCATION[1], 0.0]}

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
    reset_options = {"init_states": INIT_STATES} if INIT_STATES else None

    ep_rewards, ep_lengths = [], []
    wins = losses = 0
    try:
        for ep in range(num_episodes):
            obs_dict, _ = env.reset(seed=args.seed + ep, options=reset_options)
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
            won, lost = n_killed_opp == n_opp, n_killed_agent == n_rl
            if won and not lost:  # both: a draw, neither a win nor a loss
                wins += 1
            elif lost and not won:
                losses += 1
    except FlightSimClosed:
        print("FlightSim window closed.")

    if ep_rewards:
        n = len(ep_rewards)
        print(f"\nMean reward: {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}  "
              f"| Mean length: {np.mean(ep_lengths):.1f}")
        print(f"Mean Wins: {wins/n:.3f} | Mean Losses: {losses/n:.3f}")
    env.close()


if __name__ == "__main__":
    run_inference()
