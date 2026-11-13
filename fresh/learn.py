"""Multinomial logistic regression in numpy, fitted by Newton's method with a ridge.

Deterministic: full-batch, a fixed number of iterations, no random initialisation. The submission
is decoded on integer scores (possession.decode), so last-bit float differences between machines
cannot change the output file.
"""

import json
from dataclasses import dataclass

import numpy as np


def _log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


@dataclass(frozen=True)
class Logistic:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray  # (d + 1, k), last row the bias
    log_prior: np.ndarray  # (k,) class frequencies of the training rows

    def log_proba(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        return _log_softmax(z @ self.weights[:-1] + self.weights[-1])

    def log_likelihood(self, x):
        """log p(x | y) up to a per-frame constant: the posterior with the training prior divided
        out, so the decoder's transition costs carry the prior instead."""
        return self.log_proba(x) - self.log_prior

    def to_json(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "weights": self.weights.tolist(), "log_prior": self.log_prior.tolist()}

    @staticmethod
    def from_json(body):
        return Logistic(*(np.array(body[k], dtype=np.float64)
                          for k in ("mean", "std", "weights", "log_prior")))


def fit(x, y, k=3, l2=1.0, iterations=25, sample_weight=None):
    """Newton-Raphson on the mean cross-entropy plus l2 / n * ||W||^2 / 2 (bias unpenalised)."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = np.column_stack([(x - mean) / std, np.ones(n)])
    w8 = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    w8 = w8 / w8.sum()
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1.0
    weights = np.zeros((d + 1, k))
    penalty = np.full(d + 1, l2 / n)
    penalty[-1] = 1e-8
    for _ in range(iterations):
        p = np.exp(_log_softmax(z @ weights))
        grad = z.T @ ((p - onehot) * w8[:, None]) + penalty[:, None] * weights
        hess = np.zeros(((d + 1) * k, (d + 1) * k))
        for a in range(k):
            for b in range(a, k):
                c = p[:, a] * ((a == b) - p[:, b]) * w8
                block = (z * c[:, None]).T @ z
                hess[a * (d + 1):(a + 1) * (d + 1), b * (d + 1):(b + 1) * (d + 1)] = block
                hess[b * (d + 1):(b + 1) * (d + 1), a * (d + 1):(a + 1) * (d + 1)] = block.T
        hess += np.diag(np.tile(penalty, k))
        # the softmax is invariant to adding one vector to every class; the ridge on the bias
        # (1e-8) keeps the system solvable, and lstsq handles what is left
        step = np.linalg.lstsq(hess, grad.T.reshape(-1), rcond=None)[0]
        weights -= step.reshape(k, d + 1).T
    counts = np.bincount(y, weights=w8 * n, minlength=k)
    return Logistic(mean, std, weights, np.log(counts / counts.sum()))


def save(model, path, extra=None):
    body = {"logistic": model.to_json(), **(extra or {})}
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")


def load(path):
    body = json.loads(path.read_text())
    return Logistic.from_json(body["logistic"]), body
