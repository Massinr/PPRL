import sys, os, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from config import Config
from audit import AuditLog
from environment import Environment
from neural_network import MLP
from reinforcement_learning import ActorCritic
from training import Trainer
from reward_functions import shark_reward


def make(**kw):
    cfg = Config(**kw); a = AuditLog(tempfile.mkdtemp(), echo=False); return cfg, Environment(cfg, a), a


def test_mlp_gradcheck():
    rng = np.random.default_rng(0); net = MLP([5, 7, 6, 3], rng); x = rng.normal(size=(4, 5)); t = rng.normal(size=(4, 3))
    loss = lambda: 0.5 * ((net.forward(x)[0] - t) ** 2).sum()
    out, _ = net.forward(x); g = net.backward(out - t)
    for p, gp in zip(net.params(), g):
        for idx in [tuple(rng.integers(0, s) for s in p.shape) for _ in range(5)]:
            old = p[idx]; p[idx] = old + 1e-6; lp = loss(); p[idx] = old - 1e-6; lm = loss(); p[idx] = old
            assert abs((lp - lm) / 2e-6 - gp[idx]) < 1e-5


def test_ppo_actor_gradcheck():
    cfg = Config(); rng = np.random.default_rng(1); ac = ActorCritic(15, 2, cfg, rng)
    for p in ac.actor.params(): p += rng.normal(scale=0.2, size=p.shape)
    obs = rng.normal(size=(32, 15)); act = rng.normal(size=(32, 2)); lp0 = rng.normal(-2, .1, 32); adv = rng.normal(size=32)
    _, g, _ = ac.actor_loss_grads(obs, act, lp0 - 0.0, adv)
    params = ac.actor.params() + [ac.log_std]
    f = lambda: ac.actor_loss_grads(obs, act, lp0, adv)[0]
    for p, gp in zip(params, g):
        for idx in [tuple(rng.integers(0, s) for s in p.shape) for _ in range(4)]:
            old = p[idx]; p[idx] = old + 1e-6; lp = f(); p[idx] = old - 1e-6; lm = f(); p[idx] = old
            assert abs((lp - lm) / 2e-6 - gp[idx]) < 1e-4, (p.shape, idx)


def test_gae_hand_case():
    cfg = Config(gamma=0.5, gae_lambda=1.0); ac = ActorCritic(15, 2, cfg, np.random.default_rng(0))
    adv, ret = ac.gae([1.0, 1.0], [0.0, 0.0], [0.0, 1.0], 0.0)     # last step terminal
    assert np.allclose(adv, [1.5, 1.0]) and np.allclose(ret, adv)


def test_reward_signs():
    cfg = Config()
    _, r_eat = shark_reward(cfg, 1, 1.0, .8, 100, 300, False, .1); _, r_idle = shark_reward(cfg, 0, 1.0, .8, 100, 100, False, .1)
    _, r_die = shark_reward(cfg, 0, 1.0, 0.0, 100, 100, True, .1)
    assert r_eat > 9 and r_idle < 0 and r_die < r_idle
    _, r_close = shark_reward(cfg, 0, 0, .8, 200, 100, False, .1); _, r_far = shark_reward(cfg, 0, 0, .8, 100, 200, False, .1)
    assert r_close > r_far


def test_bounds_and_repro():
    cfg, e1, _ = make(); _, e2, _ = make()
    for _ in range(3000): e1.step(); e2.step()
    for a in e1.agents(): assert a.radius - 1e-9 <= a.x <= cfg.width - a.radius + 1e-9 and a.radius - 1e-9 <= a.y <= cfg.height - a.radius + 1e-9
    assert [(a.x, a.y) for a in e1.agents()] == [(a.x, a.y) for a in e2.agents()]


def test_eat_stamina_and_death():
    cfg, env, _ = make(n_fish=1, n_sharks=1); s, f = env.sharks[0], env.fish[0]
    s.stamina = 40; s.x, s.y, f.x, f.y = 500, 300, 510, 300; env._collisions()
    assert s.eaten == 1 and abs(s.stamina - 65) < 1e-9 and env.stats["eaten"] == 1 and math.hypot(f.x - 500, f.y - 300) > 0
    s.stamina = 1e-4; s.cmd_speed = 1.0
    env.step(); assert not s.alive and env.stats["deaths"] == 1


def test_live_shark_count():
    cfg, env, _ = make(n_sharks=3)
    env.set_shark_count(7); assert len(env.sharks) == 7
    rem = env.set_shark_count(2); assert len(env.sharks) == 2 and len(rem) == 5


def test_training_runs_finite():
    cfg, env, a = make(n_sharks=2, n_fish=30, ppo_batch=256, console_events=False); tr = Trainer(cfg, env, a)
    while tr.updates < 3: tr.tick()
    assert all(np.isfinite(p).all() for p in tr.policy.actor.params()) and tr.updates == 3
    tr.save(os.path.join(tempfile.mkdtemp(), "m.npz"))
