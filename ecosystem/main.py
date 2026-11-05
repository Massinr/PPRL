"""Entry point: Pygame world + Tk windows (controls, data, audit, neural net, learning)."""
import os
import pygame
from config import Config
from audit import AuditLog
from environment import Environment
from training import Trainer
from visualization import draw_world


class App:
    def __init__(self):
        self.cfg = Config(); self.audit = AuditLog(self.cfg.audit_dir)
        self.env = Environment(self.cfg, self.audit); self.trainer = Trainer(self.cfg, self.env, self.audit)
        self.paused, self.si, self.show_ring, self.running = False, self.cfg.default_speed_index, True, True
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.cfg.model_dir, "shark_policy.npz")

    def user(self, msg): self.audit.log(self.env.step_count, self.env.time, "ACTION", "USER", msg)
    def toggle_pause(self): self.paused = not self.paused; self.user("PAUSE" if self.paused else "RESUME")
    def set_speed(self, i):
        i = max(0, min(len(self.cfg.speed_levels) - 1, i))
        if i != self.si: self.si = i; self.user(f"speed x{self.cfg.speed_levels[i]}")
    def set_sharks(self, n):
        n = max(0, min(40, int(n))); self.trainer.drop(self.env.set_shark_count(n))
    def set_fish(self, n): self.env.set_fish_count(int(n))
    def set_attack(self, v): self.cfg.shark_attack_range = float(v); self.user(f"attack_range={v}")
    def set_gain(self, v): self.cfg.shark_eat_gain = float(v); self.user(f"stamina per fish={v}")
    def toggle_learning(self): self.trainer.training = not self.trainer.training; self.user(f"learning {'ON' if self.trainer.training else 'OFF'}")
    def toggle_console(self): self.cfg.console_events = not self.cfg.console_events; self.audit.echo = self.cfg.console_events; self.user(f"console echo {self.cfg.console_events}")
    def reset_world(self): self.user("RESET world (brain kept)"); self.env.reset(); self.trainer.on_world_reset()
    def reset_brain(self): self.trainer.reset_brain()
    def save_model(self): self.trainer.save(self.model_path)
    def load_model(self):
        if os.path.exists(self.model_path): self.trainer.load(self.model_path)
        else: self.audit.log(0, 0, "WARN", "MODEL", f"{self.model_path} not found")


def main():
    app = App(); cfg, env = app.cfg, app.env
    pygame.init(); screen = pygame.display.set_mode((cfg.width, cfg.height)); pygame.display.set_caption("Predator-Prey Ecosystem")
    font, small, clock = pygame.font.SysFont("consolas", 15), pygame.font.SysFont("consolas", 11), pygame.time.Clock()
    root, wins = None, {}
    try:
        import tkinter as tk
        from windows import ControlWindow, DataWindow, AuditWindow, BrainWindow, LearningWindow
        root = tk.Tk(); root.withdraw()
        wins = dict(ctl=ControlWindow(root, app), data=DataWindow(root), audit=AuditWindow(root, app.audit),
                    brain=BrainWindow(root, app), learn=LearningWindow(root, app))
    except Exception as ex:
        print("Tkinter windows unavailable:", ex)
    acc, frame = 0.0, 0
    keys_show = {pygame.K_F1: "ctl", pygame.K_F2: "data", pygame.K_F3: "audit", pygame.K_F4: "brain", pygame.K_F6: "learn"}
    while app.running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: app.running = False
            elif ev.type == pygame.KEYDOWN:
                k, shift = ev.key, ev.mod & pygame.KMOD_SHIFT
                if k == pygame.K_ESCAPE: app.running = False
                elif k == pygame.K_SPACE: app.toggle_pause()
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS): app.set_speed(app.si + 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS): app.set_speed(app.si - 1)
                elif k == pygame.K_r: app.reset_world()
                elif k == pygame.K_f: app.set_fish(len(env.fish) + (5 if shift else -5))
                elif k == pygame.K_s: app.set_sharks(len(env.sharks) + (1 if shift else -1))
                elif k == pygame.K_a: app.set_attack(max(5, cfg.shark_attack_range + (5 if shift else -5)))
                elif k == pygame.K_l: app.toggle_learning()
                elif k == pygame.K_b: app.reset_brain()
                elif k == pygame.K_c: app.toggle_console()
                elif k == pygame.K_g: app.show_ring = not app.show_ring
                elif k == pygame.K_F5: app.save_model()
                elif k == pygame.K_F9: app.load_model()
                elif k in keys_show and keys_show[k] in wins: wins[keys_show[k]].win.deiconify()
        if not app.paused:
            acc += cfg.speed_levels[app.si] / cfg.decision_steps          # decisions per frame
            while acc >= 1:
                app.trainer.tick(); acc -= 1
        draw_world(screen, env, font, small, app.paused, cfg.speed_levels[app.si], app.trainer.training, app.show_ring)
        pygame.display.flip(); frame += 1
        if root and frame % 6 == 0:
            try:
                for n in ("ctl", "data", "audit", "brain", "learn"):
                    w = wins[n]
                    if n == "data": w.update(env, cfg.speed_levels[app.si], app.paused)
                    elif n == "audit": w.update()
                    else: w.update()
                root.update()
            except Exception as ex:
                print("GUI closed/error:", ex); root = None
        clock.tick(60)
    app.audit.log(env.step_count, env.time, "INFO", "SHUTDOWN", "bye"); app.audit.close(); pygame.quit()


if __name__ == "__main__":
    main()
