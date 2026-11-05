"""From-scratch MLP (tanh hidden layers, linear output) with manual backprop and Adam."""
import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            lim = np.sqrt(6.0 / (n_in + n_out))                    # Xavier/Glorot uniform
            W = rng.uniform(-lim, lim, (n_in, n_out))
            if i == len(sizes) - 2: W *= out_scale                   # small output layer -> near-random start
            self.W.append(W); self.b.append(np.zeros(n_out))
        self._acts = None

    def forward(self, x):
        acts, h = [x], x
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b                                            # z = hW + b
            h = np.tanh(z) if i < last else z                        # a = tanh(z)
            acts.append(h)
        self._acts = acts
        return h, acts

    def backward(self, gout):
        """gout = dL/d(output). Returns [dW0, db0, dW1, db1, ...] matching params()."""
        L = len(self.W); gW = [None] * L; gb = [None] * L; delta = gout
        for i in reversed(range(L)):
            a_in = self._acts[i]
            gW[i] = a_in.T @ delta; gb[i] = delta.sum(0)
            if i > 0: delta = (delta @ self.W[i].T) * (1.0 - a_in ** 2)   # tanh' = 1 - a^2
        return [g for pair in zip(gW, gb) for g in pair]

    def params(self):
        return [p for pair in zip(self.W, self.b) for p in pair]


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


def clip_global_norm(grads, max_norm):
    n = np.sqrt(sum(float((g * g).sum()) for g in grads))
    if n > max_norm:
        for g in grads: g *= max_norm / (n + 1e-12)
    return n
