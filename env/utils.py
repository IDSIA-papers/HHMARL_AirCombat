import glob
import os
import yaml
import pymap3d
import numpy as np
import torch


def load_pretrained_actors(policy_name, device=torch.device('cpu')):
    """Load a checkpoint from runs/<policy_name>/ and return the full saved dict.

    The dict matches what algorithm/utils.py:Logger.save_checkpoint writes:
    "step", "actors_state_dict", "critic_state_dict", optimizer states, and an
    optional value-normalizer state. If policy_name resolves to a directory we
    pick the most recent checkpoint*.pt inside it.
    """
    path = os.path.join(get_root_dir(), 'runs', policy_name)
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "checkpoint*.pt")))
        assert candidates, f"No checkpoints found in {path}"
        path = candidates[-1]
    assert os.path.isfile(path), f"No checkpoint at {path}"
    return torch.load(path, map_location=device, weights_only=False)


def load_low_level_policies(pol_dirs, device=torch.device('cpu')):
    """Load a set of low-level policy checkpoints from runs/ keyed by model & task.

    pol_dirs entries are like 'L4_cannon_f16'. The middle token names the task
    (cannon|track|escape), the trailing token names the aircraft model (f16|A4).
    Returns {model_type: {task_type: ckpt_dict}}.
    """
    policies = {'f16': {}, 'A4': {}}
    for policy_name in pol_dirs:
        ckpt = load_pretrained_actors(policy_name, device=device)
        task_type = policy_name.split('_')[1]
        model_type = 'A4' if 'A4' in policy_name else 'f16'
        policies[model_type][task_type] = ckpt
    return policies


def _deep_merge(base, override):
    """Recursively merge override into base. Override leaves replace base leaves; dict branches merge."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    result = dict(base)
    for key, value in override.items():
        result[key] = _deep_merge(result.get(key), value) if isinstance(value, dict) else value
    return result


def _load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def parse_config(name):
    """Load a config by name with a deep merge of base.yaml and the level file.

    Accepted forms (all relative to env/configs/):
      "cannon/L1"             -> base.yaml <- levels/cannon_L1.yaml
      "cannon/L1_cannon"      -> same as cannon/L1 (back-compat with the old "L<N>_<task>" form)
      "hier"                  -> base.yaml <- hier(_7vs7).yaml
    """
    cfg_root = os.path.join(get_root_dir(), 'env', 'configs')
    base_path = os.path.join(cfg_root, 'base.yaml')
    assert os.path.exists(base_path), f"Missing base config at {base_path}"
    merged = _load_yaml(base_path)

    if '/' in name:
        task, level = name.split('/', 1)
        # Accept both "L1" and the legacy "L1_<task>" form.
        level_token = level.split('_', 1)[0]
        level_path = os.path.join(cfg_root, 'levels', f'{task}_{level_token}.yaml')
        assert os.path.exists(level_path), f"Missing level config at {level_path}"
        merged = _deep_merge(merged, _load_yaml(level_path))
    elif name != 'base':
        # Standalone configs (e.g. hier, hier_7vs7) live at the top of configs/.
        cfg_path = os.path.join(cfg_root, f'{name}.yaml')
        assert os.path.exists(cfg_path), f"Missing config at {cfg_path}"
        merged = _deep_merge(merged, _load_yaml(cfg_path))

    # A config may remove an inherited aircraft (e.g. base.yaml's R2 when a hier
    # roster uses different ids) by setting it to null. Drop those entries so the
    # env never sees a roster slot without a mission.
    aircraft = merged.get('aircraft')
    if isinstance(aircraft, dict):
        merged['aircraft'] = {aid: cfg for aid, cfg in aircraft.items() if cfg is not None}

    return type('EnvConfig', (object,), merged)


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def LLA2NEU(lon, lat, alt, lon0=0.0, lat0=0.0, alt0=0):
    """Geodetic (lon, lat, alt) -> NEU (north, east, up) in metres."""
    n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.array([n, e, -d])


def NEU2LLA(n, e, u, lon0=0.0, lat0=0.0, alt0=0):
    """NEU (north, east, up) -> geodetic (lon, lat, alt)."""
    lat, lon, h = pymap3d.ned2geodetic(n, e, -u, lat0, lon0, alt0)
    return np.array([lon, lat, h])


def get_AO_TA_R(ego_feature, enm_feature, return_side=False):
    """3D AO (antenna train angle), TA (aspect angle), R (range) between two agents.

    Features are (north, east, up, vn, ve, vu).
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y, delta_z])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
    return ego_AO, ego_TA, R, side_flag


def get2d_AO_TA_R(ego_feature, enm_feature, return_side=False, return_delta_z=False):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return (ego_AO, ego_TA, R, delta_z) if return_delta_z else (ego_AO, ego_TA, R)
    side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
    return (ego_AO, ego_TA, R, side_flag, delta_z) if return_delta_z else (ego_AO, ego_TA, R, side_flag)


def get_separate_deltas(ego_feature, enm_feature):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    delta_vx, delta_vy, delta_vz = enm_vx - ego_vx, enm_vy - ego_vy, enm_vz - ego_vz
    return delta_x, delta_y, delta_z, delta_vx, delta_vy, delta_vz


def in_range_deg(angle):
    """Normalise angle in degrees to (-180, 180]."""
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def in_range_rad(angle):
    """Normalise angle in radians to (-pi, pi]."""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle
