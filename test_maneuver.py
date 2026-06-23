import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.env import AircombatMARL

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")


def main():
    kwargs = {"hierarchy_level": 0, "training_level": 2}
    env = AircombatMARL(task="track/L1", **kwargs)
    act = {"B1": [10, 5, 5, 5, 1]}

    env.reset()
    done = False

    while not done:
        env.env.render(viewpoint=0)
        _, _, terminated, truncated, _ = env.step(act)
        done = terminated["all"] or truncated["all"]

    env.close()


if __name__ == "__main__":
    main()
