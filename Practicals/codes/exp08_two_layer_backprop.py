"""Experiment 8 (CO2): Two-Layer Neural Network with Backpropagation
Libraries: NumPy, Matplotlib"""
import numpy as np
import matplotlib.pyplot as plt
from common import out_dir, savefig

OUT = out_dir(__file__)
sig = lambda z: 1 / (1 + np.exp(-z))

class TwoLayerNet:
    def __init__(self, n_in, n_h, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1, (n_in, n_h)); self.b1 = np.zeros(n_h); self.W2 = r.normal(0, 1, (n_h, 1)); self.b2 = np.zeros(1)
    def forward(self, X):
        self.X = X; self.a1 = np.tanh(X @ self.W1 + self.b1); self.a2 = sig(self.a1 @ self.W2 + self.b2); return self.a2
    def loss(self, y):
        p = np.clip(self.a2, 1e-9, 1 - 1e-9); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    def backward(self, y):
        n = len(y); dz2 = (self.a2 - y) / n                  # dL/dz2 for sigmoid + BCE
        dW2 = self.a1.T @ dz2; db2 = dz2.sum(0)
        dz1 = (dz2 @ self.W2.T) * (1 - self.a1 ** 2)          # tanh'
        return dict(W1=self.X.T @ dz1, b1=dz1.sum(0), W2=dW2, b2=db2)
    def fit(self, X, y, epochs=3000, lr=.5):
        hist = []
        for _ in range(epochs):
            self.forward(X); hist.append(self.loss(y)); g = self.backward(y)
            for k, v in g.items(): setattr(self, k, getattr(self, k) - lr * v)
        return hist

# ---- gradient check --------------------------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); y = np.array([[0], [1], [1], [0]], float)
net = TwoLayerNet(2, 4); net.forward(X); g = net.backward(y); eps = 1e-6; maxdiff = 0
for k in ["W1", "b1", "W2", "b2"]:
    P = getattr(net, k); num = np.zeros_like(P)
    for i in np.ndindex(P.shape):
        old = P[i]; P[i] = old + eps; net.forward(X); lp = net.loss(y); P[i] = old - eps; net.forward(X); lm = net.loss(y); P[i] = old
        num[i] = (lp - lm) / (2 * eps)
    d = np.max(np.abs(num - g[k])); maxdiff = max(maxdiff, d); print(f"Gradient check {k}: max |numerical - backprop| = {d:.2e}")
print("Backpropagation verified" if maxdiff < 1e-6 else "Gradient mismatch!")

# ---- XOR ------------------------------------------------------------------------
net = TwoLayerNet(2, 4, seed=1); h1 = net.fit(X, y, 4000, .5); pred = net.forward(X)
print("\nXOR predictions:", pred.ravel().round(3), "-> classes", (pred > .5).astype(int).ravel(), "final loss %.5f" % h1[-1])

# ---- Two moons ------------------------------------------------------------------
from sklearn.datasets import make_moons
Xm, ym = make_moons(400, noise=.2, random_state=0); ym = ym[:, None].astype(float)
net2 = TwoLayerNet(2, 8, seed=0); h2 = net2.fit(Xm, ym, 3000, .3)
print("Two-moons training accuracy: %.3f, final loss %.4f" % (np.mean((net2.forward(Xm) > .5) == ym), h2[-1]))

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].plot(h1, label="XOR"); ax[0].plot(h2, label="moons"); ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("Training loss (BCE)"); ax[0].set_xlabel("epoch"); ax[0].grid()
for a, (n_, X_, y_, t) in zip(ax[1:], [(net, X, y, "XOR decision regions"), (net2, Xm, ym, "Two moons decision regions")]):
    lo, hi = X_.min(0) - .5, X_.max(0) + .5; xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    a.contourf(xx, yy, n_.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), 25, cmap="coolwarm", alpha=.6)
    a.scatter(*X_.T, c=y_.ravel(), cmap="coolwarm", edgecolor="k", s=25); a.set_title(t)
savefig(fig, OUT, "backprop_results.png")
