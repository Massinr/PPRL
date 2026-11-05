"""Actor-critic with PPO-clip, GAE and a Gaussian policy, implemented by hand in NumPy."""
import numpy as np
from neural_network import MLP, Adam, clip_global_norm

LOG2PI = np.log(2 * np.pi)


class ActorCritic:
    def __init__(self, n_in, n_act, cfg, rng):
        self.cfg, self.rng, self.n_act = cfg, rng, n_act
        h = cfg.hidden
        self.actor = MLP([n_in, h, h, n_act], rng, out_scale=0.01)
        self.critic = MLP([n_in, h, 1], rng)
        self.log_std = np.full(n_act, cfg.init_log_std)
        self.opt_a = Adam(self.actor.params() + [self.log_std], cfg.lr_actor)
        self.opt_c = Adam(self.critic.params(), cfg.lr_critic)

    # ---- acting ----
    def act(self, obs, deterministic=False):
        mu, acts = self.actor.forward(obs[None, :])
        mu = mu[0]; std = np.exp(self.log_std)
        a = mu.copy() if deterministic else mu + std * self.rng.standard_normal(mu.shape)
        logp = float(-0.5 * (((a - mu) / std) ** 2).sum() - self.log_std.sum() - 0.5 * self.n_act * LOG2PI)
        v = float(self.critic.forward(obs[None, :])[0][0, 0])
        return a, logp, v, mu, [x[0].copy() for x in acts]

    def value(self, obs):
        return float(self.critic.forward(obs[None, :])[0][0, 0])

    # ---- advantage estimation ----
    def gae(self, r, v, done, last_value):
        g, lam = self.cfg.gamma, self.cfg.gae_lambda
        T = len(r); adv = np.zeros(T); run = 0.0
        for t in reversed(range(T)):
            nv = last_value if t == T - 1 else v[t + 1]
            nt = 1.0 - done[t]
            delta = r[t] + g * nv * nt - v[t]                  # TD error
            run = delta + g * lam * nt * run                    # A_t = sum (g*lam)^l delta_{t+l}
            adv[t] = run
        return adv, adv + np.asarray(v)

    # ---- PPO loss + manual gradients ----
    def actor_loss_grads(self, obs, act, logp_old, adv):
        c = self.cfg; eps = c.clip_eps
        mu, _ = self.actor.forward(obs); std = np.exp(self.log_std)
        z = (act - mu) / std
        logp = -0.5 * (z ** 2).sum(1) - self.log_std.sum() - 0.5 * self.n_act * LOG2PI
        ratio = np.exp(logp - logp_old); B = len(obs)
        surr = np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)
        active = ~(((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps)))
        entropy = (self.log_std + 0.5 * (LOG2PI + 1)).sum()
        loss = -surr.mean() - c.ent_coef * entropy
        g = -(ratio * adv * active) / B                         # dL/dlogp per sample
        dmu = g[:, None] * z / std                              # dlogp/dmu = (a-mu)/sigma^2
        dlog = (g[:, None] * (z ** 2 - 1)).sum(0) - c.ent_coef  # dlogp/dlog sigma = z^2 - 1
        grads = self.actor.backward(dmu) + [dlog]
        return loss, grads, dict(entropy=entropy, clip_frac=float((~active).mean()), kl=float((logp_old - logp).mean()))

    def update(self, obs, act, logp_old, adv, ret):
        c = self.cfg; N = len(obs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        st = {}
        for _ in range(c.ppo_epochs):
            idx = self.rng.permutation(N)
            for s in range(0, N, c.ppo_minibatch):
                b = idx[s:s + c.ppo_minibatch]
                ploss, ga, st = self.actor_loss_grads(obs[b], act[b], logp_old[b], adv[b])
                clip_global_norm(ga, c.max_grad_norm); self.opt_a.step(ga)
                np.clip(self.log_std, -2.0, 0.5, out=self.log_std)
                V, _ = self.critic.forward(obs[b])
                err = V[:, 0] - ret[b]
                vloss = 0.5 * float((err ** 2).mean())
                gc = self.critic.backward((c.vf_coef * err / len(b))[:, None])
                clip_global_norm(gc, c.max_grad_norm); self.opt_c.step(gc)
        st.update(policy_loss=float(ploss), value_loss=vloss, std=float(np.exp(self.log_std).mean()))
        return st

    # ---- persistence ----
    def state(self):
        d = {f"a{i}": p for i, p in enumerate(self.actor.params())}
        d.update({f"c{i}": p for i, p in enumerate(self.critic.params())}); d["log_std"] = self.log_std
        return d

    def load(self, d):
        for i, p in enumerate(self.actor.params()): p[...] = d[f"a{i}"]
        for i, p in enumerate(self.critic.params()): p[...] = d[f"c{i}"]
        self.log_std[...] = d["log_std"]
