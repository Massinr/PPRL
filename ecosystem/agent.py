import math


class Agent:
    __slots__ = ("id", "kind", "x", "y", "heading", "speed", "omega", "radius", "cruise", "wall_hits")

    def __init__(self, id, kind, x, y, heading, cruise, radius):
        self.id, self.kind, self.x, self.y, self.heading = id, kind, x, y, heading
        self.cruise, self.speed, self.omega, self.radius, self.wall_hits = cruise, cruise, 0.0, radius, 0
