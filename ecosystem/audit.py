"""Audit log: ring buffer + CSV file + optional console echo."""
import csv, os, time
from collections import deque


class AuditLog:
    ECHO = {"EAT", "DEATH", "UPDATE", "SHARK_ADD", "SHARK_REMOVE", "RESET", "USER", "MODEL", "FISH"}

    def __init__(self, directory="logs", maxlen=20000, echo=True):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, time.strftime("audit_%Y%m%d_%H%M%S.csv"))
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["seq", "wall_time", "sim_step", "sim_time", "level", "category", "message"])
        self.entries, self.total, self.echo = deque(maxlen=maxlen), 0, echo

    def log(self, step, sim_time, level, category, message):
        self.total += 1
        e = (self.total, time.strftime("%H:%M:%S"), step, round(sim_time, 3), level, category, message)
        self.entries.append(e); self._w.writerow(e)
        if self.total % 50 == 0: self._f.flush()
        if self.echo and category in self.ECHO:
            print(f"[{e[1]}|t={sim_time:8.2f}|{level:6s}|{category}] {message}")

    def since(self, seq):
        return [e for e in self.entries if e[0] > seq]

    def close(self):
        self._f.flush(); self._f.close()
