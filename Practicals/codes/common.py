"""Shared helpers for the CSA47 Deep Learning lab experiments."""
import os, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless: figures are saved, not shown
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def out_dir(script_file):
    """outputs/<script-name>/ folder for the given script (__file__)."""
    d = os.path.join(ROOT, "outputs", os.path.splitext(os.path.basename(script_file))[0])
    os.makedirs(d, exist_ok=True)
    return d


def savefig(fig, out, fname):
    fig.tight_layout()
    path = os.path.join(out, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[saved] {os.path.relpath(path, ROOT)}")


def have_tf():
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


def need_tf(name):
    """Used by TensorFlow-only experiments."""
    if not have_tf():
        raise SystemExit(f"{name}: TensorFlow/Keras is required. Install with `pip install tensorflow` and re-run.")


# ----------------------------------------------------------------------------
# NumPy multilayer perceptron (softmax output, Adam, L2, dropout, early stopping)
# ----------------------------------------------------------------------------
class NumpyMLP:
    def __init__(self, sizes, act="relu", l2=0.0, dropout=0.0, lr=1e-3, seed=0):
        self.rng = np.random.default_rng(seed)
        self.sizes, self.act, self.l2, self.p, self.lr = sizes, act, l2, dropout, lr
        W, b = [], []
        for i, o in zip(sizes[:-1], sizes[1:]):
            s = np.sqrt(2.0 / i) if act == "relu" else np.sqrt(1.0 / i)
            W.append(self.rng.normal(0, s, (i, o)))
            b.append(np.zeros(o))
        self.L = len(W)
        self.P = W + b                      # W = P[:L], b = P[L:] (shared arrays)
        self.m = [np.zeros_like(p) for p in self.P]
        self.v = [np.zeros_like(p) for p in self.P]
        self.t = 0

    @property
    def W(self): return self.P[:self.L]
    @property
    def b(self): return self.P[self.L:]

    def _f(self, z):
        if self.act == "relu": return np.maximum(0, z)
        if self.act == "tanh": return np.tanh(z)
        return sigmoid(z)

    def _df(self, a, z):
        if self.act == "relu": return (z > 0).astype(float)
        if self.act == "tanh": return 1 - a ** 2
        return a * (1 - a)

    def forward(self, X, train=False):
        ins, zs, acts, masks = [], [], [], []
        a = X
        for i in range(self.L - 1):
            ins.append(a)
            z = a @ self.W[i] + self.b[i]
            h = self._f(z)
            if train and self.p > 0:
                mask = (self.rng.random(h.shape) >= self.p) / (1 - self.p)   # inverted dropout
            else:
                mask = np.ones_like(h)
            zs.append(z); acts.append(h); masks.append(mask)
            a = h * mask
        ins.append(a)
        z = a @ self.W[-1] + self.b[-1]
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(1, keepdims=True)
        return probs, (ins, zs, acts, masks)

    def _step(self, X, Y):
        probs, (ins, zs, acts, masks) = self.forward(X, train=True)
        n = len(X)
        dz = (probs - Y) / n
        gW, gb = [None] * self.L, [None] * self.L
        for i in reversed(range(self.L)):
            gW[i] = ins[i].T @ dz + self.l2 * self.W[i]
            gb[i] = dz.sum(0)
            if i > 0:
                da = (dz @ self.W[i].T) * masks[i - 1]
                dz = da * self._df(acts[i - 1], zs[i - 1])
        grads = gW + gb
        self.t += 1
        for p, g, m, v in zip(self.P, grads, self.m, self.v):       # Adam
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            mh = m / (1 - 0.9 ** self.t); vh = v / (1 - 0.999 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    def predict_proba(self, X): return self.forward(X)[0]
    def predict(self, X): return self.predict_proba(X).argmax(1)

    def evaluate(self, X, y):
        p = self.predict_proba(X)
        loss = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        return loss, np.mean(p.argmax(1) == y)

    def fit(self, X, y, Xv=None, yv=None, epochs=50, batch=32, patience=None):
        K = self.sizes[-1]; Y = np.eye(K)[y]; n = len(X)
        h = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "stopped_epoch": epochs}
        best, best_w, wait = np.inf, None, 0
        for ep in range(epochs):
            idx = self.rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                self._step(X[bi], Y[bi])
            l, a = self.evaluate(X, y); h["loss"].append(l); h["acc"].append(a)
            if Xv is not None:
                vl, va = self.evaluate(Xv, yv); h["val_loss"].append(vl); h["val_acc"].append(va)
                if patience:
                    if vl < best - 1e-6:
                        best, wait = vl, 0
                        best_w = [p.copy() for p in self.P]
                    else:
                        wait += 1
                        if wait >= patience:
                            h["stopped_epoch"] = ep + 1
                            break
        if patience and best_w is not None:
            for p, q in zip(self.P, best_w): p[...] = q
        return h


# ----------------------------------------------------------------------------
# NumPy many-to-one RNN with full BPTT
# ----------------------------------------------------------------------------
def rnn_init(n_in, n_h, rho=0.9, seed=0):
    rng = np.random.default_rng(seed)
    Wh = rng.normal(size=(n_h, n_h))
    Wh *= rho / max(abs(np.linalg.eigvals(Wh)))          # set spectral radius
    return {"Wx": rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_h)), "Wh": Wh, "b": np.zeros(n_h),
            "Wy": rng.normal(0, 1 / np.sqrt(n_h), (n_h, 1)), "by": np.zeros(1)}


def rnn_loss_grads(P, X, y):
    """X:(N,T,D) y:(N,). Returns loss, grads, per-time-step ||dL/dh_t|| (t=1..T)."""
    N, T, D = X.shape; nh = P["Wh"].shape[0]
    H = np.zeros((N, T + 1, nh))
    for t in range(T):
        H[:, t + 1] = np.tanh(X[:, t] @ P["Wx"] + H[:, t] @ P["Wh"] + P["b"])
    yhat = (H[:, -1] @ P["Wy"] + P["by"])[:, 0]
    loss = 0.5 * np.mean((yhat - y) ** 2)
    d = (yhat - y) / N
    g = {k: np.zeros_like(v) for k, v in P.items()}
    g["Wy"] = H[:, -1].T @ d[:, None]; g["by"] = np.array([d.sum()])
    dh = d[:, None] * P["Wy"].T
    norms = np.zeros(T)
    for t in reversed(range(T)):
        norms[t] = np.mean(np.linalg.norm(dh, axis=1)) * N       # scale back by N
        dz = dh * (1 - H[:, t + 1] ** 2)
        g["Wx"] += X[:, t].T @ dz; g["Wh"] += H[:, t].T @ dz; g["b"] += dz.sum(0)
        dh = dz @ P["Wh"].T
    return loss, g, norms


def rnn_predict(P, X):
    N, T, _ = X.shape; h = np.zeros((N, P["Wh"].shape[0]))
    for t in range(T):
        h = np.tanh(X[:, t] @ P["Wx"] + h @ P["Wh"] + P["b"])
    return (h @ P["Wy"] + P["by"])[:, 0]


def grad_norm(g): return float(np.sqrt(sum((v ** 2).sum() for v in g.values())))


def rnn_train(P, X, y, epochs=60, lr=1e-2, opt="adam", clip=None, batch=32, seed=0, Xv=None, yv=None):
    rng = np.random.default_rng(seed); n = len(X)
    m = {k: np.zeros_like(v) for k, v in P.items()}; s = {k: np.zeros_like(v) for k, v in P.items()}
    t = 0; hist = {"loss": [], "val_loss": [], "grad_norm": []}
    for ep in range(epochs):
        idx = rng.permutation(n); gn = []
        for i in range(0, n, batch):
            bi = idx[i:i + batch]
            loss, g, _ = rnn_loss_grads(P, X[bi], y[bi])
            norm = grad_norm(g); gn.append(norm)
            if clip is not None and norm > clip:
                g = {k: v * clip / norm for k, v in g.items()}
            t += 1
            for k in P:
                if opt == "adam":
                    m[k] = 0.9 * m[k] + 0.1 * g[k]; s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
                    P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
                else:
                    P[k] -= lr * g[k]
        hist["loss"].append(0.5 * np.mean((rnn_predict(P, X) - y) ** 2))
        if Xv is not None: hist["val_loss"].append(0.5 * np.mean((rnn_predict(P, Xv) - yv) ** 2))
        hist["grad_norm"].append(float(np.max(gn)))
    return hist


# ----------------------------------------------------------------------------
# Binary Restricted Boltzmann Machine
# ----------------------------------------------------------------------------
class RBM:
    def __init__(self, nv, nh, lr=0.1, seed=0):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.05, (nv, nh)); self.a = np.zeros(nv); self.b = np.zeros(nh); self.lr = lr

    def ph(self, v): return sigmoid(v @ self.W + self.b)
    def pv(self, h): return sigmoid(h @ self.W.T + self.a)
    def sample(self, p): return (self.rng.random(p.shape) < p).astype(float)

    def energy(self, v, h):
        return -(v @ self.a) - (h @ self.b) - np.sum((v @ self.W) * h, axis=1)

    def free_energy(self, v):
        return -(v @ self.a) - np.sum(np.logaddexp(0, v @ self.W + self.b), axis=1)

    def gibbs_vhv(self, v):
        h = self.sample(self.ph(v)); pv = self.pv(h)
        return pv, self.sample(pv)

    def cd_step(self, v0, k=1):
        ph0 = self.ph(v0); h = self.sample(ph0); vk = v0
        for _ in range(k):
            pvk = self.pv(h); vk = self.sample(pvk); phk = self.ph(vk); h = self.sample(phk)
        n = len(v0)
        self.W += self.lr * (v0.T @ ph0 - vk.T @ phk) / n
        self.a += self.lr * (v0 - vk).mean(0)
        self.b += self.lr * (ph0 - phk).mean(0)

    def recon_error(self, v):
        return float(np.mean((v - self.pv(self.ph(v))) ** 2))

    def fit(self, V, epochs=20, batch=32, k=1, callback=None):
        errs = []
        for ep in range(epochs):
            idx = self.rng.permutation(len(V))
            for s in range(0, len(V), batch):
                self.cd_step(V[idx[s:s + batch]], k)
            errs.append(self.recon_error(V))
            if callback: callback(ep, self)
        return errs


# ----------------------------------------------------------------------------
# Small discrete Bayesian network (binary variables) - used when pgmpy is unavailable
# ----------------------------------------------------------------------------
import itertools


class SimpleBN:
    """cpt[node] = (parents, {parent_value_tuple: P(node=1)})."""
    def __init__(self, order, cpt):
        self.order, self.cpt = order, cpt

    def p_node(self, node, val, assign):
        parents, table = self.cpt[node]
        p1 = table[tuple(assign[p] for p in parents)]
        return p1 if val == 1 else 1 - p1

    def joint(self, assign):
        return float(np.prod([self.p_node(n, assign[n], assign) for n in self.order]))

    def query(self, target, evidence=None):
        evidence = evidence or {}; free = [n for n in self.order if n not in evidence]; res = {0: 0., 1: 0.}
        for vals in itertools.product([0, 1], repeat=len(free)):
            a = dict(evidence); a.update(zip(free, vals)); res[a[target]] += self.joint(a)
        z = res[0] + res[1]; return {k: v / z for k, v in res.items()}

    def forward_sample(self, n, rng):
        out = []
        for _ in range(n):
            a = {}
            for node in self.order:
                parents, table = self.cpt[node]; a[node] = int(rng.random() < table[tuple(a[p] for p in parents)])
            out.append(a)
        return out

    def edges(self):
        return [(p, n) for n in self.order for p in self.cpt[n][0]]


def alarm_network():
    """Classic burglary-alarm network (Russell & Norvig)."""
    cpt = {
        "Burglary": ((), {(): .001}),
        "Earthquake": ((), {(): .002}),
        "Alarm": (("Burglary", "Earthquake"), {(1, 1): .95, (1, 0): .94, (0, 1): .29, (0, 0): .001}),
        "JohnCalls": (("Alarm",), {(1,): .90, (0,): .05}),
        "MaryCalls": (("Alarm",), {(1,): .70, (0,): .01}),
    }
    return SimpleBN(["Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"], cpt)
