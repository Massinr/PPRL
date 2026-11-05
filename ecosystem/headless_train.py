"""Train without any window (fast). Usage: python headless_train.py --minutes 5 --sharks 3 --plot curve.png"""
import argparse, os, time
import numpy as np
from config import Config
from audit import AuditLog
from environment import Environment
from training import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=2.0); ap.add_argument("--sharks", type=int, default=3)
    ap.add_argument("--fish", type=int, default=50); ap.add_argument("--plot", default=None)
    ap.add_argument("--save", default=None); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = Config(n_sharks=a.sharks, n_fish=a.fish, console_events=not a.quiet)
    audit = AuditLog(cfg.audit_dir, echo=not a.quiet); env = Environment(cfg, audit); tr = Trainer(cfg, env, audit)
    t0 = time.time()
    while time.time() - t0 < a.minutes * 60: tr.tick()
    print(f"done: {env.time:.0f} sim-s in {time.time()-t0:.0f} s, updates={tr.updates}, eaten={env.stats['eaten']}")
    if a.save: os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True); tr.save(a.save)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.2)); h = tr.hist
        for x, k, t in zip(ax, ("eats_per_min", "mean_reward", "stamina"), ("Fish eaten / shark / min", "Mean step reward", "Mean stamina")):
            x.plot(h[k], lw=1, alpha=.4); w = 10
            if len(h[k]) > w: x.plot(range(w - 1, len(h[k])), np.convolve(h[k], np.ones(w) / w, "valid"), lw=2)
            x.set_title(t); x.set_xlabel("PPO update"); x.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(a.plot, dpi=160)
    audit.close()


if __name__ == "__main__":
    main()
