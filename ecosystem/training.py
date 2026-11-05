"""Glue: decision loop, trajectories, PPO updates, point ledger, console reports."""
import numpy as np
from shark import observe, OBS_DIM, COMPS
from reinforcement_learning import ActorCritic
from reward_functions import shark_reward


class Trainer:
    def __init__(self, cfg, env, audit):
        self.cfg, self.env, self.audit = cfg, env, audit
        self.rng = np.random.default_rng(cfg.seed + 1)
        self.policy = ActorCritic(OBS_DIM, 2, cfg, self.rng)
        self.training = True
        self.traj, self.updates = {}, 0
        self.hist = {k: [] for k in ("mean_reward", "eats_per_min", "stamina", "kl", "entropy", "value_loss", "std")}
        self._mark = (0, 0.0); self._next_console = cfg.console_interval_s
        self.last_stats = {}

    # ---------- housekeeping ----------
    def reset_brain(self):
        self.policy = ActorCritic(OBS_DIM, 2, self.cfg, self.rng)
        self.traj.clear(); self.updates = 0
        for k in self.hist: self.hist[k].clear()
        self.audit.log(self.env.step_count, self.env.time, "ACTION", "USER", "brain re-initialised (random weights)")

    def on_world_reset(self):
        self.traj.clear(); self._mark = (0, 0.0); self._next_console = self.cfg.console_interval_s

    def drop(self, sharks):
        for s in sharks:
            if s is not None: self.traj.pop(s.id, None)

    def save(self, path):
        np.savez(path, **self.policy.state())
        self.audit.log(self.env.step_count, self.env.time, "INFO", "MODEL", f"saved {path}")

    def load(self, path):
        self.policy.load(dict(np.load(path)))
        self.audit.log(self.env.step_count, self.env.time, "INFO", "MODEL", f"loaded {path}")

    # ---------- main loop ----------
    def _fish_arrays(self):
        env = self.env
        P = np.array([(f.x, f.y) for f in env.fish]).reshape(-1, 2)
        V = np.array([(f.speed * np.cos(f.heading), f.speed * np.sin(f.heading)) for f in env.fish]).reshape(-1, 2)
        return P, V

    def _nearest(self, s, P):
        if not len(P): return self.cfg.shape_dmax
        return float(np.hypot(P[:, 0] - s.x, P[:, 1] - s.y).min())

    def tick(self):
        """One decision interval (decision_steps physics steps)."""
        cfg, env = self.cfg, self.env
        sharks = [s for s in env.sharks if s.alive]
        P, V = self._fish_arrays()
        obs = {s.id: observe(env, s, P, V) for s in sharks}
        if self.training and sum(len(t["r"]) for t in self.traj.values()) >= cfg.ppo_batch:
            self._update(obs)
        pend = {}
        for s in sharks:
            o, d = obs[s.id]
            a, logp, v, mu, acts = self.policy.act(o, deterministic=not self.training)
            s.cmd_turn, s.cmd_speed = float(np.clip(a[0], -1, 1)), float(np.clip(a[1], -1, 1))
            s.view = dict(acts=acts, mu=mu, a=a, v=v)
            s.interval_energy, s.interval_eaten = 0.0, 0
            pend[s.id] = (o, a, logp, v, d)
        for _ in range(cfg.decision_steps): env.step()
        P, _ = self._fish_arrays()
        dt_dec = cfg.decision_steps * cfg.dt
        for s in sharks:
            if s.id not in pend or s not in env.sharks: continue
            o, a, logp, v, d_prev = pend[s.id]
            died = not s.alive
            comps, r = shark_reward(cfg, s.interval_eaten, s.interval_energy, s.stamina / cfg.stamina_max,
                                    d_prev, self._nearest(s, P), died, dt_dec)
            for k in COMPS: s.points[k] += comps[k]
            s.life_points += r; s.total_points += r; s.last_reward = r
            if self.training:
                t = self.traj.setdefault(s.id, {k: [] for k in ("o", "a", "logp", "v", "r", "done")})
                for k, x in zip(("o", "a", "logp", "v", "r", "done"), (o, a, logp, v, r, float(died))): t[k].append(x)
            if died:
                env.respawn_shark(s)
        self._console()

    def _update(self, obs):
        cfg, env, pol = self.cfg, self.env, self.policy
        O, A, LP, AD, RT, R = [], [], [], [], [], []
        for sid, t in self.traj.items():
            if not t["r"]: continue
            last_v = 0.0 if t["done"][-1] else (pol.value(obs[sid][0]) if sid in obs else t["v"][-1])
            adv, ret = pol.gae(t["r"], t["v"], t["done"], last_v)
            O += t["o"]; A += t["a"]; LP += t["logp"]; AD.append(adv); RT.append(ret); R += t["r"]
        self.traj.clear()
        st = pol.update(np.array(O), np.array(A), np.array(LP), np.concatenate(AD), np.concatenate(RT))
        self.updates += 1
        eats0, t0 = self._mark; shark_min = max(1e-9, len(env.sharks) * (env.time - t0) / 60.0)
        epm = (env.stats["eaten"] - eats0) / shark_min
        self._mark = (env.stats["eaten"], env.time)
        stam = float(np.mean([s.stamina for s in env.sharks])) if env.sharks else 0.0
        h = self.hist
        h["mean_reward"].append(float(np.mean(R))); h["eats_per_min"].append(epm); h["stamina"].append(stam)
        h["kl"].append(st["kl"]); h["entropy"].append(st["entropy"]); h["value_loss"].append(st["value_loss"]); h["std"].append(st["std"])
        self.last_stats = dict(st, mean_reward=float(np.mean(R)), eats_per_min=epm, n=len(R))
        self.audit.log(env.step_count, env.time, "INFO", "UPDATE",
                       f"#{self.updates} n={len(R)} mean_r={np.mean(R):+.4f} eats/shark-min={epm:.2f} "
                       f"pol_loss={st['policy_loss']:+.4f} v_loss={st['value_loss']:.3f} ent={st['entropy']:+.3f} "
                       f"kl={st['kl']:+.4f} clip={st['clip_frac']:.2f} std={st['std']:.3f}")

    def _console(self):
        env, cfg = self.env, self.cfg
        if not cfg.console_events or env.time < self._next_console: return
        self._next_console += cfg.console_interval_s
        st, ls = env.stats, self.last_stats
        lines = [f"---- t={env.time:7.1f}s step={env.step_count} updates={self.updates} learning={'ON' if self.training else 'OFF'} "
                 f"fish={len(env.fish)} sharks={len(env.sharks)} | eaten={st['eaten']} missed={st['failed_attacks']} deaths={st['deaths']}"]
        if ls: lines.append(f"     PPO: mean_r={ls['mean_reward']:+.4f} eats/shark-min={ls['eats_per_min']:.2f} "
                            f"ent={ls['entropy']:+.3f} kl={ls['kl']:+.4f} vloss={ls['value_loss']:.3f} std={ls['std']:.3f}")
        for s in env.sharks:
            comp = " ".join(f"{k}{s.points[k]:+.1f}" for k in COMPS)
            lines.append(f"     {s.id:>4} life{s.life:<3} stam={s.stamina:5.1f} ate={s.eaten}/{s.eaten_total} "
                         f"pts(life)={s.life_points:+7.2f} pts(total)={s.total_points:+8.2f} | {comp}")
        print("\n".join(lines), flush=True)
