"""Central configuration. Every coefficient lives here so nothing is hidden."""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- World ----
    width: int = 1000
    height: int = 700
    dt: float = 1 / 60                 # fixed physics step (s)
    seed: int = 42
    n_fish: int = 50
    n_sharks: int = 3

    # ---- Geometry (px) ----
    fish_radius: float = 5.0
    shark_radius: float = 10.0
    shark_attack_range: float = 25.0   # a fish closer than range + fish_radius is caught

    # ---- Fish (random wanderers for now; fish learning comes in a later phase) ----
    fish_cruise: float = 60.0
    speed_sigma: float = 6.0
    speed_theta: float = 1.0
    ou_theta: float = 1.5              # OU mean reversion kappa
    ou_sigma: float = 2.5              # stationary std of turn rate = sigma/sqrt(2*kappa)
    max_turn: float = 3.0              # rad/s

    # ---- Shark body & stamina ----
    shark_max_speed: float = 140.0     # px/s at speed command +1
    shark_accel: float = 4.0           # 1/s, first-order speed response
    stamina_max: float = 100.0
    shark_e0: float = 1.0              # idle stamina drain  (units/s)
    shark_e1: float = 4.0              # extra drain at full speed, scales with (v/vmax)^2
    shark_rest_thresh: float = 0.25    # v/vmax below this -> recovering
    shark_rest_rate: float = 2.0       # recovery (units/s)
    shark_eat_gain: float = 25.0       # stamina gained per fish (adjustable live)

    # ---- Observation scales ----
    obs_k_fish: int = 3
    obs_pos_scale: float = 300.0
    obs_vel_scale: float = 150.0
    obs_density_radius: float = 150.0

    # ---- Reward coefficients (explained in the PDF, section "Shark reward") ----
    r_eat: float = 10.0                # +R per fish caught
    lam_energy: float = 0.05           # penalty per stamina unit spent
    lam_time: float = 0.1              # penalty per second alive without a catch
    lam_shape: float = 1.0             # potential-based approach shaping weight
    shape_dmax: float = 400.0          # distance scale for potential
    lam_low: float = 2.0               # low-stamina penalty weight
    low_stamina_frac: float = 0.3      # threshold below which penalty applies
    r_death: float = 5.0               # penalty on starvation

    # ---- PPO / network ----
    decision_steps: int = 6            # physics steps per decision (10 Hz)
    hidden: int = 24
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 3e-3
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_batch: int = 1024
    ppo_epochs: int = 4
    ppo_minibatch: int = 256
    init_log_std: float = -0.5

    # ---- UI / logging ----
    speed_levels: tuple = (0.25, 0.5, 1, 2, 4, 8, 16, 32)
    default_speed_index: int = 2
    audit_dir: str = "logs"
    model_dir: str = "models"
    summary_interval_s: float = 5.0
    console_interval_s: float = 10.0   # sim seconds between console tables
    console_events: bool = True        # echo EAT / DEATH / UPDATE events to console
