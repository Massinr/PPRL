"""2D world: fish wander, sharks obey NN commands, stamina, eating, death, live population control."""
import math
import numpy as np
from config import Config
from audit import AuditLog
from fish import Fish
from shark import Shark


class Environment:
    def __init__(self, cfg: Config, audit: AuditLog):
        self.cfg, self.audit = cfg, audit
        self.reset()

    # ---------- population ----------
    def reset(self):
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        self.step_count, self.time = 0, 0.0
        self.fish, self.sharks = [], []
        self.fish_counter = self.shark_counter = 0
        for _ in range(c.n_fish): self._new_fish()
        for _ in range(c.n_sharks): self.add_shark(log=False)
        self.contacts = set()
        self.stats = dict(wall_hits=0, contact_events=0, fish_fish_overlaps=0, fish_overlap_now=0,
                          eaten=0, failed_attacks=0, deaths=0)
        self._next_summary = c.summary_interval_s
        self.audit.log(0, 0.0, "INFO", "RESET", f"World {c.width}x{c.height} fish={c.n_fish} sharks={c.n_sharks} seed={c.seed}")

    def _rand_pos(self, r):
        return self.rng.uniform(r, self.cfg.width - r), self.rng.uniform(r, self.cfg.height - r)

    def _new_fish(self):
        c = self.cfg
        x, y = self._rand_pos(c.fish_radius)
        self.fish.append(Fish(f"F{self.fish_counter}", "fish", x, y, self.rng.uniform(0, 2 * math.pi), c.fish_cruise, c.fish_radius))
        self.fish_counter += 1

    def add_shark(self, log=True):
        c = self.cfg
        x, y = self._rand_pos(c.shark_radius)
        s = Shark(f"S{self.shark_counter}", x, y, self.rng.uniform(0, 2 * math.pi), c)
        self.shark_counter += 1
        self.sharks.append(s)
        if log: self.audit.log(self.step_count, self.time, "INFO", "SHARK_ADD", f"{s.id} added at ({x:.0f},{y:.0f}); sharks={len(self.sharks)}")
        return s

    def remove_shark(self):
        if not self.sharks: return None
        s = self.sharks.pop()
        self.audit.log(self.step_count, self.time, "INFO", "SHARK_REMOVE", f"{s.id} removed; sharks={len(self.sharks)}")
        return s

    def set_shark_count(self, n):
        removed = []
        while len(self.sharks) < n: self.add_shark()
        while len(self.sharks) > n: removed.append(self.remove_shark())
        self.cfg.n_sharks = n
        return removed

    def set_fish_count(self, n):
        n = max(0, n)
        while len(self.fish) < n: self._new_fish()
        while len(self.fish) > n: self.fish.pop()
        self.cfg.n_fish = n
        self.audit.log(self.step_count, self.time, "INFO", "FISH", f"fish count -> {n}")

    def respawn_shark(self, s):
        c = self.cfg
        s.x, s.y = self._rand_pos(s.radius); s.heading = self.rng.uniform(0, 2 * math.pi)
        s.stamina, s.alive, s.speed = c.stamina_max, True, 0.5 * c.shark_max_speed
        s.life += 1; s.eaten = 0; s.life_points = 0.0

    def agents(self):
        return self.sharks + self.fish

    # ---------- physics ----------
    def step(self):
        c, dt = self.cfg, self.cfg.dt
        sq = math.sqrt(dt)
        noise = self.rng.standard_normal((len(self.fish), 2))
        for a, (n1, n2) in zip(self.fish, noise):
            a.omega += -c.ou_theta * a.omega * dt + c.ou_sigma * sq * n1
            a.omega = max(-c.max_turn, min(c.max_turn, a.omega))
            a.speed += c.speed_theta * (a.cruise - a.speed) * dt + c.speed_sigma * sq * n2
            a.speed = max(0.0, a.speed)
            a.heading = (a.heading + a.omega * dt) % (2 * math.pi)
            a.x += a.speed * math.cos(a.heading) * dt
            a.y += a.speed * math.sin(a.heading) * dt
            self._walls(a)
        for s in self.sharks:
            if s.alive: self._shark_step(s)
        self._collisions()
        self.step_count += 1; self.time += dt
        if self.time >= self._next_summary:
            self._next_summary += c.summary_interval_s
            st = self.stats
            self.audit.log(self.step_count, self.time, "INFO", "SUMMARY",
                           f"eaten={st['eaten']} failed={st['failed_attacks']} deaths={st['deaths']} wall_hits={st['wall_hits']}")

    def _shark_step(self, s):
        c, dt = self.cfg, self.cfg.dt
        s.omega = c.max_turn * s.cmd_turn
        s.heading = (s.heading + s.omega * dt) % (2 * math.pi)
        target = c.shark_max_speed * (s.cmd_speed + 1) / 2
        s.speed += (target - s.speed) * min(1.0, c.shark_accel * dt)
        s.x += s.speed * math.cos(s.heading) * dt
        s.y += s.speed * math.sin(s.heading) * dt
        self._walls(s)
        frac = s.speed / c.shark_max_speed
        spent = (c.shark_e0 + c.shark_e1 * frac * frac) * dt
        s.stamina -= spent; s.interval_energy += spent
        if frac < c.shark_rest_thresh: s.stamina += c.shark_rest_rate * dt
        s.stamina = min(c.stamina_max, s.stamina)
        if s.stamina <= 0:
            s.stamina, s.alive, s.speed = 0.0, False, 0.0
            self.stats["deaths"] += 1
            self.audit.log(self.step_count, self.time, "EVENT", "DEATH",
                           f"{s.id} starved (life {s.life}, ate {s.eaten}, life points {s.life_points:+.2f})")

    def _walls(self, a):
        c, r = self.cfg, a.radius
        hit = False
        if a.x < r: a.x, a.heading, hit = r, math.pi - a.heading, True
        elif a.x > c.width - r: a.x, a.heading, hit = c.width - r, math.pi - a.heading, True
        if a.y < r: a.y, a.heading, hit = r, -a.heading, True
        elif a.y > c.height - r: a.y, a.heading, hit = c.height - r, -a.heading, True
        if hit:
            a.heading %= 2 * math.pi; a.wall_hits += 1; self.stats["wall_hits"] += 1

    def _d2(self, live):
        S = np.array([(s.x, s.y) for s in live]); F = np.array([(f.x, f.y) for f in self.fish])
        return ((S[:, None, :] - F[None, :, :]) ** 2).sum(-1)

    def _eat(self, s, f, d):
        c = self.cfg
        s.stamina = min(c.stamina_max, s.stamina + c.shark_eat_gain)
        s.interval_eaten += 1; s.eaten += 1; s.eaten_total += 1; self.stats["eaten"] += 1
        self.audit.log(self.step_count, self.time, "EVENT", "EAT",
                       f"{s.id} ate {f.id} (d={d:.1f}px, v={s.speed:.0f}) stamina->{s.stamina:.1f}")
        f.x, f.y = self._rand_pos(f.radius); f.heading = self.rng.uniform(0, 2 * math.pi)

    def _collisions(self):
        c = self.cfg
        live = [s for s in self.sharks if s.alive]
        if live and self.fish:
            D2 = self._d2(live); thr2 = (c.shark_attack_range + c.fish_radius) ** 2
            taken = set(); eaten_ids = set()
            for i, s in enumerate(live):
                j = int(np.argmin(D2[i]))
                if D2[i, j] < thr2 and j not in taken:
                    taken.add(j); eaten_ids.add(self.fish[j].id); self._eat(s, self.fish[j], math.sqrt(D2[i, j]))
            if taken: D2 = self._d2(live)
            now = {(live[i].id, self.fish[j].id) for i, j in zip(*np.where(D2 < thr2))}
            old = {p for p in self.contacts if p[1] not in eaten_ids}
            live_ids = {s.id for s in live}
            for p in now - old:
                self.stats["contact_events"] += 1
                self.audit.log(self.step_count, self.time, "EVENT", "ATTACK_RANGE_ENTER", f"{p[0]} -> {p[1]}")
            for p in old - now:
                if p[0] in live_ids:
                    self.stats["failed_attacks"] += 1
                    self.audit.log(self.step_count, self.time, "EVENT", "ATTACK_MISSED", f"{p[0]} -> {p[1]} escaped")
            self.contacts = now
        else:
            self.contacts = set()
        if len(self.fish) > 1:
            F = np.array([(f.x, f.y) for f in self.fish])
            D2 = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
            n = int((D2[np.triu_indices(len(F), 1)] < (2 * c.fish_radius) ** 2).sum())
            if n > self.stats["fish_overlap_now"]: self.stats["fish_fish_overlaps"] += n - self.stats["fish_overlap_now"]
            self.stats["fish_overlap_now"] = n
