"""Experiment 7 (CO2): Perceptron and Forward Propagation
Libraries: NumPy, Matplotlib"""
import numpy as np
import matplotlib.pyplot as plt
from common import out_dir, savefig

OUT = out_dir(__file__)

class Perceptron:
    def __init__(self, n, lr=.1): self.w = np.zeros(n); self.b = 0.; self.lr = lr
    def forward(self, X): return (X @ self.w + self.b >= 0).astype(int)      # step activation
    def fit(self, X, y, epochs=50):
        errs = []
        for _ in range(epochs):
            e = 0
            for xi, yi in zip(X, y):
                upd = self.lr * (yi - self.forward(xi[None])[0]); self.w += upd * xi; self.b += upd; e += int(upd != 0)
            errs.append(e)
            if e == 0: break
        return errs

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
gates = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "NAND": [1, 1, 1, 0], "XOR": [0, 1, 1, 0]}
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
for a, (g, y) in zip(ax, gates.items()):
    y = np.array(y); p = Perceptron(2); errs = p.fit(X, y, 100); pred = p.forward(X)
    print(f"{g:5s}: weights={p.w.round(2)}, bias={p.b:.2f}, predictions={pred}, targets={y}, accuracy={np.mean(pred==y):.2f}, epochs={len(errs)}")
    xx, yy = np.meshgrid(np.linspace(-.5, 1.5, 200), np.linspace(-.5, 1.5, 200))
    a.contourf(xx, yy, p.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), alpha=.3, cmap="coolwarm")
    a.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=120, edgecolor="k"); a.set_title(f"{g} (acc {np.mean(pred==y):.2f})")
savefig(fig, OUT, "logic_gates_perceptron.png")
print("-> XOR is not linearly separable, so a single perceptron cannot learn it.")

# Linearly separable data set
rng = np.random.default_rng(0)
X2 = np.r_[rng.normal([-2, -2], 1, (100, 2)), rng.normal([2, 2], 1, (100, 2))]; y2 = np.r_[np.zeros(100), np.ones(100)].astype(int)
p = Perceptron(2, .01); errs = p.fit(X2, y2, 50); print(f"\n2-D blobs: accuracy {np.mean(p.forward(X2)==y2):.3f}, mistakes per epoch {errs}")

# Forward propagation through a 2-3-1 network (fixed weights) with step-by-step values
sig = lambda z: 1 / (1 + np.exp(-z))
W1 = np.array([[.2, -.4, .1], [.7, .3, -.5]]); b1 = np.array([.1, -.1, .2]); W2 = np.array([[.6], [-.3], [.8]]); b2 = np.array([.05])
x = np.array([[1.0, 2.0]]); z1 = x @ W1 + b1; a1 = sig(z1); z2 = a1 @ W2 + b2; a2 = sig(z2)
print("\nForward propagation (2-3-1 network, sigmoid):")
print("x =", x, "\nz1 = xW1+b1 =", z1.round(4), "\na1 = sigmoid(z1) =", a1.round(4), "\nz2 =", z2.round(4), "\noutput a2 =", a2.round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(*X2[y2 == 0].T, label="class 0", s=15); ax[0].scatter(*X2[y2 == 1].T, label="class 1", s=15)
xs = np.linspace(-5, 5, 10); ax[0].plot(xs, -(p.w[0] * xs + p.b) / p.w[1], "k", label="decision boundary"); ax[0].legend(); ax[0].set_title("Perceptron on separable data")
ax[1].plot(errs, "o-"); ax[1].set_title("Perceptron mistakes per epoch"); ax[1].set_xlabel("epoch"); ax[1].grid()
savefig(fig, OUT, "perceptron_blobs.png")
