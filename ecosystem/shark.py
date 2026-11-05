"""Shark entity, stamina state, reward ledger and the observation function."""
import math
import numpy as np
from agent import Agent

COMPS = ("eat", "energy", "time", "shape", "low", "death")
OBS_DIM = 15
OBS_NAMES = [f"f{k}.{n}" for k in (1, 2, 3) for n in ("x", "y", "vx", "vy")] + ["density", "stamina", "speed"]


class Shark(Agent):
    def __init__(self, id, x, y, heading, cfg):
        super().__init__(id, "shark", x, y, heading, 0.0, cfg.shark_radius)
        self.speed = 0.5 * cfg.shark_max_speed
        self.stamina = cfg.stamina_max
        self.cmd_turn = self.cmd_speed = 0.0
        self.alive, self.life = True, 1
        self.eaten = self.eaten_total = 0
        self.interval_eaten, self.interval_energy = 0, 0.0
        self.points = {k: 0.0 for k in COMPS}       # cumulative points by component
        self.life_points = self.total_points = self.last_reward = 0.0
        self.view = None                             # last NN activations for the GUI


def observe(env, s, Fpos, Fvel):
    """15-D egocentric observation (shark frame: +x = heading)."""
    c = env.cfg
    ch, sh = math.cos(s.heading), math.sin(s.heading)
    obs = np.zeros(OBS_DIM)
    d_near = c.shape_dmax
    if len(Fpos):
        rel = Fpos - np.array([s.x, s.y])
        dist = np.hypot(rel[:, 0], rel[:, 1])
        d_near = float(dist.min())
        order = np.argsort(dist)[:c.obs_k_fish]
        svx, svy = s.speed * ch, s.speed * sh
        for k, j in enumerate(order):
            dx, dy = rel[j]
            dvx, dvy = Fvel[j, 0] - svx, Fvel[j, 1] - svy
            obs[4 * k:4 * k + 4] = [(dx * ch + dy * sh) / c.obs_pos_scale, (-dx * sh + dy * ch) / c.obs_pos_scale,
                                    (dvx * ch + dvy * sh) / c.obs_vel_scale, (-dvx * sh + dvy * ch) / c.obs_vel_scale]
        obs[12] = (dist < c.obs_density_radius).sum() / 10.0
    obs[:12] = np.clip(obs[:12], -1, 1)
    obs[12] = min(obs[12], 1.0)
    obs[13] = s.stamina / c.stamina_max
    obs[14] = s.speed / c.shark_max_speed
    return obs, d_near
