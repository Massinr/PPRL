"""Tkinter windows: Controls, Data, Audit Log, Neural Network (live), Learning/Points."""
import tkinter as tk
from tkinter import ttk
import numpy as np
from shark import COMPS, OBS_NAMES


def _hex(t, neg=(70, 130, 255), pos=(255, 70, 70), mid=(45, 50, 60)):
    t = max(-1.0, min(1.0, float(t))); tgt = pos if t >= 0 else neg; k = abs(t)
    return "#%02x%02x%02x" % tuple(int(m + (g - m) * k) for m, g in zip(mid, tgt))


class ControlWindow:
    def __init__(self, root, app):
        self.app = app; w = self.win = tk.Toplevel(root); w.title("Controls"); w.geometry("330x470+1040+0")
        w.protocol("WM_DELETE_WINDOW", w.withdraw)
        c = app.cfg; self.v = {k: tk.IntVar(value=val) for k, val in
                               dict(sharks=c.n_sharks, fish=c.n_fish, attack=int(c.shark_attack_range), gain=int(c.shark_eat_gain)).items()}
        def row(r, label, key, lo, hi, step, cmd):
            ttk.Label(w, text=label).grid(row=r, column=0, sticky="w", padx=8, pady=5)
            sp = ttk.Spinbox(w, from_=lo, to=hi, increment=step, width=7, textvariable=self.v[key],
                             command=lambda: cmd(self.v[key].get()))
            sp.grid(row=r, column=1, padx=6); sp.bind("<Return>", lambda e: cmd(self._get(key)))
            return sp
        self.sp_sharks = row(0, "Sharks (live)", "sharks", 0, 40, 1, app.set_sharks)
        self.sp_fish = row(1, "Fish (live)", "fish", 1, 400, 5, app.set_fish)
        row(2, "Attack range (px)", "attack", 5, 120, 5, app.set_attack)
        row(3, "Stamina per fish eaten", "gain", 0, 100, 5, app.set_gain)
        ttk.Label(w, text="Sim speed").grid(row=4, column=0, sticky="w", padx=8)
        self.sc = tk.Scale(w, from_=0, to=len(app.cfg.speed_levels) - 1, orient="horizontal", showvalue=False,
                           command=lambda v: app.set_speed(int(float(v))))
        self.sc.set(app.si); self.sc.grid(row=4, column=1, padx=6)
        self.lbl_speed = ttk.Label(w, text=""); self.lbl_speed.grid(row=5, column=1, sticky="w", padx=6)
        btns = [("Pause / Resume", app.toggle_pause), ("Learning ON/OFF", app.toggle_learning), ("Reset world", app.reset_world),
                ("Reset brain (random weights)", app.reset_brain), ("Save model", app.save_model), ("Load model", app.load_model),
                ("Console echo ON/OFF", app.toggle_console)]
        for i, (t, f) in enumerate(btns): ttk.Button(w, text=t, command=f).grid(row=6 + i, column=0, columnspan=2, sticky="ew", padx=8, pady=2)

    def _get(self, key):
        try: return int(float(self.win.nametowidget(str(self.v[key])) if False else self.v[key].get()))
        except Exception: return 0

    def update(self):
        a = self.app
        self.lbl_speed.config(text=f"x{a.cfg.speed_levels[a.si]}  {'PAUSED' if a.paused else 'running'}")
        foc = self.win.focus_get()
        if foc is not self.sp_sharks: self.v["sharks"].set(len(a.env.sharks))
        if foc is not self.sp_fish: self.v["fish"].set(len(a.env.fish))


class DataWindow:
    COLS = ("id", "kind", "x", "y", "heading", "speed", "stamina", "eaten", "points", "wall_hits")

    def __init__(self, root):
        self.win = tk.Toplevel(root); self.win.title("Ecosystem Data"); self.win.geometry("760x520+1040+490")
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)
        self.summary = tk.StringVar()
        ttk.Label(self.win, textvariable=self.summary, justify="left", font=("Consolas", 10)).pack(anchor="w", padx=6, pady=4)
        fr = ttk.Frame(self.win); fr.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(fr, columns=self.COLS, show="headings")
        for c in self.COLS: self.tree.heading(c, text=c); self.tree.column(c, width=72, anchor="e")
        sb = ttk.Scrollbar(fr, command=self.tree.yview); self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")

    def update(self, env, speed, paused):
        s = env.stats
        fs = [f for f in env.fish]; av = np.mean([a.stamina for a in env.sharks]) if env.sharks else 0
        self.summary.set(f"t={env.time:.1f}s step={env.step_count} x{speed} {'PAUSED' if paused else 'RUNNING'}\n"
                         f"fish={len(fs)} sharks={len(env.sharks)} avg shark stamina={av:.1f} attack_range={env.cfg.shark_attack_range:.0f}\n"
                         f"eaten={s['eaten']} missed={s['failed_attacks']} deaths={s['deaths']} range-entries={s['contact_events']}\n"
                         f"wall_hits={s['wall_hits']} fish overlaps total/now={s['fish_fish_overlaps']}/{s['fish_overlap_now']}")
        ids = set(self.tree.get_children())
        for a in env.agents():
            sh = a.kind == "shark"
            vals = (a.id, a.kind, f"{a.x:.0f}", f"{a.y:.0f}", f"{a.heading:.2f}", f"{a.speed:.0f}",
                    f"{a.stamina:.1f}" if sh else "-", a.eaten_total if sh else "-", f"{a.total_points:+.1f}" if sh else "-", a.wall_hits)
            if a.id in ids: self.tree.item(a.id, values=vals); ids.discard(a.id)
            else: self.tree.insert("", "end", iid=a.id, values=vals)
        for st in ids: self.tree.delete(st)


class AuditWindow:
    COLS = ("seq", "wall", "step", "sim_t", "level", "category", "message")

    def __init__(self, root, audit):
        self.audit, self.last = audit, 0
        self.win = tk.Toplevel(root); self.win.title("Audit Log"); self.win.geometry("1000x420+0+740")
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)
        self.tree = ttk.Treeview(self.win, columns=self.COLS, show="headings")
        for c, w in zip(self.COLS, (55, 65, 65, 65, 55, 150, 560)): self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(self.win, command=self.tree.yview); self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")

    def update(self):
        new = self.audit.since(self.last)
        for e in new[-300:]: self.tree.insert("", "end", values=e)
        if new: self.last = new[-1][0]
        kids = self.tree.get_children()
        if len(kids) > 3000: self.tree.delete(*kids[:len(kids) - 3000])
        if new: self.tree.see(self.tree.get_children()[-1])


class BrainWindow:
    """Live picture of the actor network: node colour = activation, edge colour/width = w_ij * a_i (signal flow)."""
    def __init__(self, root, app):
        self.app = app; self.win = tk.Toplevel(root); self.win.title("Shark Neural Network (actor)"); self.win.geometry("900x640+330+0")
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)
        top = ttk.Frame(self.win); top.pack(fill="x")
        ttk.Label(top, text="Shark:").pack(side="left", padx=6)
        self.combo = ttk.Combobox(top, width=8, state="readonly"); self.combo.pack(side="left")
        self.info = tk.StringVar(); ttk.Label(top, textvariable=self.info, font=("Consolas", 10)).pack(side="left", padx=12)
        self.cv = tk.Canvas(self.win, width=900, height=590, bg="#10161d", highlightthickness=0); self.cv.pack(fill="both", expand=True)
        self.sizes = None

    def _build(self, sizes):
        cv = self.cv; cv.delete("all"); self.sizes = sizes
        W, H = 900, 590; xs = [170, 400, 620, 800]; self.nodes, self.edges, self.pos = [], [], []
        for L, n in enumerate(sizes):
            top, bot = 45, H - 30; step = (bot - top) / max(1, n - 1) if n > 1 else 0
            ys = [(top + bot) / 2] if n == 1 else [top + i * step for i in range(n)]
            self.pos.append([(xs[L], y) for y in ys])
        for L in range(len(sizes) - 1):
            ids = []
            for i, (x1, y1) in enumerate(self.pos[L]):
                row = []
                for j, (x2, y2) in enumerate(self.pos[L + 1]): row.append(cv.create_line(x1, y1, x2, y2, fill="#2a3340", width=1))
                ids.append(row)
            self.edges.append(ids)
        for L, n in enumerate(sizes):
            r = 9 if n > 10 else 11 if n > 2 else 18
            self.nodes.append([cv.create_oval(x - r, y - r, x + r, y + r, fill="#2d3240", outline="#8890a0") for x, y in self.pos[L]])
        for i, (x, y) in enumerate(self.pos[0]): cv.create_text(x - 16, y, text=OBS_NAMES[i], fill="#9aa5b5", anchor="e", font=("Consolas", 8))
        for L, t in enumerate(("INPUT (observation)", "HIDDEN 1 (tanh)", "HIDDEN 2 (tanh)", "OUTPUT (mean action)")):
            cv.create_text(xs[L], 18, text=t, fill="#d0d8e0", font=("Consolas", 9, "bold"))
        self.out_txt = [cv.create_text(self.pos[-1][k][0] + 24, self.pos[-1][k][1], anchor="w", fill="#ffd9a0", font=("Consolas", 10), text="")
                        for k in range(sizes[-1])]

    def update(self):
        app = self.app; sh = app.env.sharks
        ids = [s.id for s in sh]
        if tuple(self.combo["values"]) != tuple(ids):
            self.combo["values"] = ids
            if self.combo.get() not in ids and ids: self.combo.set(ids[0])
        s = next((x for x in sh if x.id == self.combo.get()), None)
        if s is None or s.view is None: self.info.set("no shark / waiting for first decision"); return
        pol = app.trainer.policy; net = pol.actor
        if self.sizes != net.sizes: self._build(net.sizes)
        acts = s.view["acts"]; cv = self.cv
        for L, a in enumerate(acts):
            for i, v in enumerate(a):
                cv.itemconfig(self.nodes[L][i], fill=_hex(v if L < len(acts) - 1 else v * 3))
        for L, W in enumerate(net.W):
            S = W * acts[L][:, None]                      # signal flow w_ij * a_i
            for i in range(S.shape[0]):
                for j in range(S.shape[1]):
                    sg = S[i, j]; cv.itemconfig(self.edges[L][i][j], fill=_hex(sg * 2.5, mid=(28, 34, 44)), width=0.3 + min(2.8, abs(sg) * 5))
        mu, a = s.view["mu"], s.view["a"]
        names = ("turn cmd", "speed cmd")
        for k in range(len(mu)):
            cv.itemconfig(self.out_txt[k], text=f"{names[k]}\n mu={mu[k]:+.2f}\n act={np.clip(a[k], -1, 1):+.2f}")
        std = np.exp(pol.log_std)
        self.info.set(f"{s.id} life{s.life} stamina={s.stamina:.0f} V(s)={s.view['v']:+.2f} last_r={s.last_reward:+.3f} "
                      f"pts={s.total_points:+.1f} sigma=({std[0]:.2f},{std[1]:.2f}) updates={app.trainer.updates}")


class LearningWindow:
    def __init__(self, root, app):
        self.app = app; self.win = tk.Toplevel(root); self.win.title("Learning & Points"); self.win.geometry("760x700+330+660")
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)
        self.txt = tk.StringVar(); ttk.Label(self.win, textvariable=self.txt, font=("Consolas", 10), justify="left").pack(anchor="w", padx=6, pady=4)
        cols = ("id", "life", "stamina", "ate(life)", "ate(total)", "pts(life)", "pts(total)") + COMPS
        self.tree = ttk.Treeview(self.win, columns=cols, show="headings", height=7)
        for c in cols: self.tree.heading(c, text=c); self.tree.column(c, width=62, anchor="e")
        self.tree.pack(fill="x", padx=4)
        self.cv = tk.Canvas(self.win, width=740, height=440, bg="#10161d", highlightthickness=0); self.cv.pack(padx=4, pady=4)
        self._drawn = -1

    def _chart(self, box, ys, title, color):
        cv = self.cv; x0, y0, x1, y1 = box
        cv.create_rectangle(x0, y0, x1, y1, outline="#39424f"); cv.create_text(x0 + 4, y0 + 8, anchor="w", text=title, fill="#d0d8e0", font=("Consolas", 9))
        ys = ys[-300:]
        if len(ys) < 2: return
        lo, hi = min(ys), max(ys); hi = hi if hi > lo else lo + 1
        pts = []
        for i, v in enumerate(ys): pts += [x0 + (x1 - x0) * i / (len(ys) - 1), y1 - 6 - (y1 - y0 - 24) * (v - lo) / (hi - lo)]
        cv.create_line(*pts, fill=color, width=2)
        cv.create_text(x1 - 4, y0 + 8, anchor="e", text=f"min {lo:.2f}  max {hi:.2f}  last {ys[-1]:.2f}", fill="#8890a0", font=("Consolas", 8))

    def update(self):
        a = self.app; env, tr = a.env, a.trainer; ls = tr.last_stats
        self.txt.set(f"updates={tr.updates} learning={'ON' if tr.training else 'OFF'}  " +
                     (f"mean_r={ls['mean_reward']:+.3f} eats/shark-min={ls['eats_per_min']:.1f} entropy={ls['entropy']:+.2f} "
                      f"KL={ls['kl']:+.4f} vloss={ls['value_loss']:.2f} clip={ls['clip_frac']:.2f}" if ls else "(waiting for first PPO update)"))
        ids = set(self.tree.get_children())
        for s in env.sharks:
            v = (s.id, s.life, f"{s.stamina:.0f}", s.eaten, s.eaten_total, f"{s.life_points:+.1f}", f"{s.total_points:+.1f}") + tuple(f"{s.points[k]:+.1f}" for k in COMPS)
            if s.id in ids: self.tree.item(s.id, values=v); ids.discard(s.id)
            else: self.tree.insert("", "end", iid=s.id, values=v)
        for i in ids: self.tree.delete(i)
        if tr.updates != self._drawn:
            self._drawn = tr.updates; self.cv.delete("all"); h = tr.hist
            self._chart((4, 4, 736, 146), h["eats_per_min"], "fish eaten / shark / min (per PPO update)", "#6fd36f")
            self._chart((4, 150, 736, 292), h["mean_reward"], "mean step reward (points per decision)", "#ffb347")
            self._chart((4, 296, 736, 436), h["entropy"], "policy entropy (exploration)", "#7aa7ff")
