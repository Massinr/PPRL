"""Pygame rendering: sharks = diamonds (+ stamina bar), fish = thin triangles."""
import math
import pygame

BG, FISH_C, SHARK_C, TXT = (10, 30, 55), (120, 220, 255), (230, 70, 70), (230, 235, 240)


def _poly(a, pts):
    ch, sh = math.cos(a.heading), math.sin(a.heading)
    return [(a.x + px * ch - py * sh, a.y + px * sh + py * ch) for px, py in pts]


def draw_world(screen, env, font, small, paused, speed, learning, show_range=True):
    cfg = env.cfg
    screen.fill(BG)
    for s in env.sharks:
        r = s.radius
        col = SHARK_C if s.alive else (90, 90, 90)
        if show_range: pygame.draw.circle(screen, (150, 50, 50), (int(s.x), int(s.y)), int(cfg.shark_attack_range), 1)
        pygame.draw.polygon(screen, col, _poly(s, [(1.8 * r, 0), (0, 0.9 * r), (-1.4 * r, 0), (0, -0.9 * r)]))
        f = max(0.0, s.stamina / cfg.stamina_max)                      # stamina bar
        pygame.draw.rect(screen, (60, 60, 60), (s.x - 15, s.y - r - 12, 30, 4))
        pygame.draw.rect(screen, (int(255 * (1 - f)), int(220 * f), 60), (s.x - 15, s.y - r - 12, 30 * f, 4))
        screen.blit(small.render(f"{s.id} {s.total_points:+.0f}pts", True, (255, 200, 200)), (s.x - 24, s.y - r - 26))
    for f in env.fish:
        r = f.radius
        pygame.draw.polygon(screen, FISH_C, _poly(f, [(2.0 * r, 0), (-1.2 * r, 0.45 * r), (-1.2 * r, -0.45 * r)]))
    hud = (f"t={env.time:6.1f}s step={env.step_count} speed x{speed} fish={len(env.fish)} sharks={len(env.sharks)} "
           f"eaten={env.stats['eaten']} missed={env.stats['failed_attacks']} deaths={env.stats['deaths']} "
           f"learning={'ON' if learning else 'OFF'} {'PAUSED' if paused else 'RUNNING'}")
    screen.blit(font.render(hud, True, TXT), (8, 6))
    screen.blit(small.render("SPACE pause | +/- speed | R reset world | S/s sharks +/-1 (live) | F/f fish +/-5 | A/a attack range | "
                             "L learning | B reset brain | F5 save | F9 load | C console | F1-F4 windows | ESC", True, (150, 160, 170)),
                (8, cfg.height - 18))
