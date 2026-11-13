"""Alternative frame classifiers to the logistic regression (learn.py), Oct 22 2026.

Both give log_proba and log_likelihood (posterior with the training prior divided out), the only
interface the decoder and the EM adaptation read: adapt.py never updates the classifier, it adds
per-channel tables to its log-likelihood, so any calibrated classifier plugs in unchanged.
Hyperparameters fixed before measuring (fresh/README.md); single-threaded and seeded.

- `gbm`: scikit-learn HistGradientBoostingClassifier (LightGBM's algorithm; LightGBM itself needs
  a system libomp on macOS, absent here, so it was not used).
- `net`: a small temporal net, features at 7 frame offsets (a 1D convolution with fixed taps over
  +-2 s) -> 32 ReLU -> softmax, numpy, Adam, fixed seed. Byte-reproducible on one machine with
  single-threaded BLAS; across BLAS builds last bits may differ.
"""

import os

import numpy as np

OFFSETS = (-50, -25, -10, 0, 10, 25, 50)  # frames, 25 fps


def _log_prior(y, k=3):
    counts = np.bincount(y, minlength=k).astype(np.float64)
    return np.log(counts / counts.sum())


class GBM:
    """Boosted trees read from plain arrays (JSON-safe): per iteration and class, a tree of
    (feature, threshold, left, right, leaf, value) nodes; every column is finite (feats.compute),
    so a frame goes left when its value is at most the threshold."""

    def __init__(self, baseline, trees, log_prior):
        self.baseline, self.trees, self.log_prior = baseline, trees, log_prior

    @staticmethod
    def from_sklearn(model, log_prior):
        trees = []
        for per_class in model._predictors:
            for k, pred in enumerate(per_class):
                nd = pred.nodes
                trees.append({"class": k, "feature": nd["feature_idx"].astype(int).tolist(),
                              "threshold": nd["num_threshold"].astype(float).tolist(),
                              "left": nd["left"].astype(int).tolist(),
                              "right": nd["right"].astype(int).tolist(),
                              "leaf": nd["is_leaf"].astype(int).tolist(),
                              "value": nd["value"].astype(float).tolist()})
        return GBM(np.asarray(model._baseline_prediction, dtype=np.float64).ravel(), trees,
                   log_prior)

    def raw(self, x):
        x = np.asarray(x, dtype=np.float64)
        out = np.tile(self.baseline, (len(x), 1))
        rows = np.arange(len(x))
        for t in self.trees:
            feature, threshold = np.array(t["feature"]), np.array(t["threshold"])
            left, right = np.array(t["left"]), np.array(t["right"])
            leaf, value = np.array(t["leaf"], dtype=bool), np.array(t["value"])
            node = np.zeros(len(x), dtype=np.int64)
            while True:
                active = ~leaf[node]
                if not active.any():
                    break
                n = node[active]
                go_left = x[rows[active], feature[n]] <= threshold[n]
                node[active] = np.where(go_left, left[n], right[n])
            out[:, t["class"]] += value[node]
        return out

    def log_proba(self, x):
        z = self.raw(x)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return np.log(np.clip(p, 1e-6, None))  # as measured (sklearn predict_proba, clipped)

    def to_json(self):
        # -0.0 and 0.0 come out of the fit unpredictably; write them the same way
        return _no_negative_zero({"kind": "gbm", "baseline": self.baseline.tolist(),
                                  "trees": self.trees, "log_prior": self.log_prior.tolist()})

    @staticmethod
    def from_json(body):
        return GBM(np.array(body["baseline"]), body["trees"], np.array(body["log_prior"]))

    def log_likelihood(self, x):
        return self.log_proba(x) - self.log_prior


def _no_negative_zero(v):
    if isinstance(v, dict):
        return {k: _no_negative_zero(u) for k, u in v.items()}
    if isinstance(v, list):
        return [_no_negative_zero(u) for u in v]
    if isinstance(v, float) and v == 0.0:
        return 0.0
    return v


def fit_gbm(x, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=False, random_state=0).fit(x, y)
    return GBM.from_sklearn(model, _log_prior(y))


def windows(x):
    """Each frame's features at OFFSETS, clipped at the array's ends."""
    n = len(x)
    idx = np.clip(np.arange(n)[:, None] + np.array(OFFSETS)[None, :], 0, n - 1)
    return x[idx].reshape(n, -1).astype(np.float32)


def _log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


class Net:
    def __init__(self, mean, std, w1, b1, w2, b2, log_prior):
        self.mean, self.std, self.w1, self.b1, self.w2, self.b2 = mean, std, w1, b1, w2, b2
        self.log_prior = log_prior

    def _rows(self, z):
        h = np.maximum(((z - self.mean) / self.std) @ self.w1 + self.b1, 0)
        return _log_softmax(h @ self.w2 + self.b2)

    def log_proba(self, x):
        z = windows(np.asarray(x, dtype=np.float64)).astype(np.float64)
        return np.concatenate([self._rows(z[i:i + 65536]) for i in range(0, len(z), 65536)])

    def log_likelihood(self, x):
        return self.log_proba(x) - self.log_prior


def fit_net(z, y, hidden=32, epochs=8, batch=2048, lr=1e-3, l2=1e-4, seed=0):
    """z: windowed rows (windows())."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(seed)
    z = np.asarray(z, dtype=np.float32)
    mean, std = z.mean(0), z.std(0)
    std[std < 1e-9] = 1.0
    zs = ((z - mean) / std).astype(np.float32)
    d, k = z.shape[1], 3
    params = [rng.normal(0, np.sqrt(2 / d), (d, hidden)), np.zeros(hidden),
              rng.normal(0, np.sqrt(1 / hidden), (hidden, k)), np.zeros(k)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    onehot = np.eye(k)[y]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(len(zs))
        for s in range(0, len(zs), batch):
            i = order[s:s + batch]
            a, yb = zs[i], onehot[i]
            pre = a @ params[0] + params[1]
            h = np.maximum(pre, 0)
            p = np.exp(_log_softmax(h @ params[2] + params[3]))
            g = (p - yb) / len(i)
            gw2 = h.T @ g + l2 * params[2]
            gh = (g @ params[2].T) * (pre > 0)
            grads = [a.T @ gh + l2 * params[0], gh.sum(0), gw2, g.sum(0)]
            t += 1
            for j, gr in enumerate(grads):
                m[j] = 0.9 * m[j] + 0.1 * gr
                v[j] = 0.999 * v[j] + 0.001 * gr * gr
                params[j] -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
    return Net(mean, std, *params, _log_prior(y))


def from_json(body):
    """A frame classifier stored by pipeline.fit: logistic (learn.py) unless it names its kind."""
    if body.get("kind") == "gbm":
        return GBM.from_json(body)
    import learn
    return learn.Logistic.from_json(body)
