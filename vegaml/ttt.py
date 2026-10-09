"""TTT: a cross-validated linear head on Vega's frozen perception features.

Vega's engine already computes pooled features from the frozen backbone. TTT reads those same features
and decides with a small logistic head instead of the physics readout. The head costs one matmul
(~0.2 ms, measured 169 us on a T4), so a TTT decision is the same price as an engine decision.

What it needs that the engine does not: labelled examples for the question you are asking. The head is
fitted per label set, which is why it cannot be a drop-in for an arbitrary one-off question.
"""
from __future__ import annotations

import numpy as np


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def _fit(Z, y, C, l2, iters=400, lr=1.0):
    W = np.zeros((Z.shape[1], C), np.float32)
    b = np.zeros(C, np.float32)
    Y = np.zeros((len(y), C), np.float32)
    Y[np.arange(len(y)), y] = 1
    for _ in range(iters):
        P = _softmax(Z @ W + b)
        G = ((P - Y) / len(y)).astype(np.float32)
        W -= lr * (Z.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


class TTTHead:
    """PCA + multinomial logistic head over frozen features."""

    def __init__(self, labels, k=128, l2=1e-2):
        self.labels = list(labels)
        self.k, self.l2 = k, l2
        self.ready = False

    def fit(self, X, y_labels):
        X = np.asarray(X, dtype=np.float32)
        idx = {name: i for i, name in enumerate(self.labels)}
        y = np.array([idx[str(v)] for v in y_labels])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.c = Z.mean(0)
        k = min(self.k, max(2, min(Z.shape) - 1))
        _, _, Vt = np.linalg.svd(Z - self.c, full_matrices=False)
        self.P = Vt[:k].T
        Zp = (Z - self.c) @ self.P
        self.s = Zp.std(0) + 1e-6
        self.W, self.b = _fit(Zp / self.s, y, len(self.labels), self.l2)
        self.ready = True
        return self

    def predict(self, x, allowed=None):
        Z = (np.asarray(x, dtype=np.float32).reshape(1, -1) - self.mu) / self.sd
        Zp = ((Z - self.c) @ self.P) / self.s
        p = _softmax(Zp @ self.W + self.b)[0]
        out = {name: float(p[i]) for i, name in enumerate(self.labels)}
        if allowed:                      # restrict to the options this case actually offers
            out = {k: v for k, v in out.items() if k in set(allowed)}
        tot = sum(out.values()) or 1.0
        return {k: v / tot for k, v in out.items()}

    def cross_val_accuracy(self, X, y_labels, folds=5):
        """Honest estimate: every example scored by a head that never saw it."""
        X = np.asarray(X, dtype=np.float32)
        y = [str(v) for v in y_labels]
        n = len(y)
        if n < 2 * folds:
            return None
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        hit = 0
        for f in range(folds):
            te = perm[f::folds]
            tr = np.array([i for i in perm if i not in set(te.tolist())])
            h = TTTHead(self.labels, self.k, self.l2).fit(X[tr], [y[i] for i in tr])
            for i in te:
                p = h.predict(X[i])
                hit += int(max(p, key=p.get) == y[i])
        return hit / n
