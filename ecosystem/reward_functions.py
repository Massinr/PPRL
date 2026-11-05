"""Shark reward. r = r_eat + r_energy + r_time + r_shape + r_low + r_death (see PDF for derivation)."""


def potential(d, cfg):
    """Phi(s) = -min(d, dmax)/dmax  in [-1, 0]; 0 is best (touching prey)."""
    return -min(d, cfg.shape_dmax) / cfg.shape_dmax


def shark_reward(cfg, n_eaten, energy_spent, stamina_frac, d_prev, d_now, died, dt_decision):
    phi_prev = potential(d_prev, cfg)
    phi_now = 0.0 if n_eaten > 0 else potential(d_now, cfg)       # catching = reaching the goal potential
    low = max(0.0, cfg.low_stamina_frac - stamina_frac)
    comps = {
        "eat": cfg.r_eat * n_eaten,
        "energy": -cfg.lam_energy * energy_spent,
        "time": -cfg.lam_time * dt_decision,
        "shape": cfg.lam_shape * (cfg.gamma * phi_now - phi_prev),   # Ng, Harada & Russell (1999)
        "low": -cfg.lam_low * low * low,
        "death": -cfg.r_death if died else 0.0,
    }
    return comps, sum(comps.values())
