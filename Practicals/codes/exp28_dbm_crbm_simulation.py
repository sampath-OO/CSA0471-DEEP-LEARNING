"""Experiment 28 (CO5): Deep Boltzmann Machine / Convolutional Boltzmann Machine Simulation
Libraries: NumPy, SciPy, TensorFlow, Keras, Matplotlib
Part A: Deep Boltzmann Machine (64-40-20) - mean-field positive phase + persistent Gibbs chains (Salakhutdinov & Hinton, 2009), NumPy.
Part B: Convolutional RBM with weight-shared 3x3 filters on 8x8 digits (SciPy correlate2d / convolve2d), NumPy/SciPy.
(TensorFlow is not needed to run this experiment.)"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d, convolve2d
from sklearn.datasets import load_digits
from common import out_dir, savefig, RBM, sigmoid

OUT = out_dir(__file__); rng = np.random.default_rng(0)
digits = load_digits().data / 16; Vb = (digits > .4).astype(float)
bern = lambda p: (rng.random(p.shape) < p).astype(float)

# =============================== Part A: DBM ====================================
class DBM:
    def __init__(self, n0, n1, n2):
        self.W1 = rng.normal(0, .05, (n0, n1)); self.W2 = rng.normal(0, .05, (n1, n2)); self.a = np.zeros(n0); self.b1 = np.zeros(n1); self.b2 = np.zeros(n2)
    def energy(self, v, h1, h2):
        return -(v @ self.a) - (h1 @ self.b1) - (h2 @ self.b2) - np.sum((v @ self.W1) * h1, 1) - np.sum((h1 @ self.W2) * h2, 1)
    def mean_field(self, v, iters=10):
        m1 = sigmoid(v @ self.W1 * 2 + self.b1); m2 = sigmoid(m1 @ self.W2 + self.b2)        # doubled bottom-up input (as in the DBM paper)
        for _ in range(iters):
            m1 = sigmoid(v @ self.W1 + m2 @ self.W2.T + self.b1); m2 = sigmoid(m1 @ self.W2 + self.b2)
        return m1, m2
    def gibbs(self, v, h1, h2):
        h1 = bern(sigmoid(v @ self.W1 + h2 @ self.W2.T + self.b1)); h2 = bern(sigmoid(h1 @ self.W2 + self.b2)); v = bern(sigmoid(h1 @ self.W1.T + self.a))
        return v, h1, h2
    def fit(self, V, epochs=40, batch=25, lr=.02, chains=50):
        v_c = bern(np.full((chains, V.shape[1]), .3)); h1_c = bern(np.full((chains, self.W1.shape[1]), .5)); h2_c = bern(np.full((chains, self.W2.shape[1]), .5)); errs = []
        for ep in range(epochs):
            idx = rng.permutation(len(V))
            for s in range(0, len(V), batch):
                v = V[idx[s:s + batch]]; m1, m2 = self.mean_field(v); n = len(v)
                v_c, h1_c, h2_c = self.gibbs(v_c, h1_c, h2_c)
                self.W1 += lr * (v.T @ m1 / n - v_c.T @ h1_c / chains); self.W2 += lr * (m1.T @ m2 / n - h1_c.T @ h2_c / chains)
                self.a += lr * (v.mean(0) - v_c.mean(0)); self.b1 += lr * (m1.mean(0) - h1_c.mean(0)); self.b2 += lr * (m2.mean(0) - h2_c.mean(0))
            m1, m2 = self.mean_field(V); errs.append(np.mean((V - sigmoid(m1 @ self.W1.T + self.a)) ** 2))
            if ep % 10 == 0: print(f"  DBM epoch {ep+1:3d}: reconstruction MSE {errs[-1]:.4f}")
        self.chains = (v_c, h1_c, h2_c); return errs

print("=== Part A: Deep Boltzmann Machine 64-40-20 ===")
dbm = DBM(64, 40, 20); errs_dbm = dbm.fit(Vb, epochs=50)
rbm = RBM(64, 40, lr=.05, seed=0); errs_rbm = rbm.fit(Vb, epochs=50, batch=25)
print(f"Final reconstruction MSE: DBM (mean-field, 2 hidden layers) = {errs_dbm[-1]:.4f} | single RBM (40 hidden) = {errs_rbm[-1]:.4f}")
v_c, h1_c, h2_c = dbm.chains
for _ in range(300): v_c, h1_c, h2_c = dbm.gibbs(v_c, h1_c, h2_c)
print("Mean pixel activity: data %.3f, DBM samples %.3f" % (Vb.mean(), v_c.mean()))
print("Mean energy of data (with mean-field h): %.2f | of DBM samples: %.2f | of noise: %.2f" % (
    dbm.energy(Vb, *dbm.mean_field(Vb)).mean(), dbm.energy(v_c, h1_c, h2_c).mean(), dbm.energy((rng.random((200, 64)) < .5).astype(float), bern(np.full((200, 40), .5)), bern(np.full((200, 20), .5))).mean()))

# =============================== Part B: Convolutional RBM ====================
class CRBM:
    def __init__(self, K=6, f=3, lr=.02):
        self.K, self.f, self.lr = K, f, lr; self.W = rng.normal(0, .1, (K, f, f)); self.b = np.zeros(K); self.a = 0.
    def up(self, v): return np.array([sigmoid(correlate2d(v, self.W[k], mode="valid") + self.b[k]) for k in range(self.K)])      # (K, 6, 6)
    def down(self, h): return sigmoid(sum(convolve2d(h[k], self.W[k], mode="full") for k in range(self.K)) + self.a)             # (8, 8)
    def step(self, v0):
        h0 = self.up(v0); hs = (rng.random(h0.shape) < h0).astype(float); v1 = self.down(hs); h1 = self.up(v1)
        for k in range(self.K): self.W[k] += self.lr * (correlate2d(v0, h0[k], mode="valid") - correlate2d(v1, h1[k], mode="valid")) / h0[k].size
        self.b += self.lr * (h0 - h1).mean((1, 2)); self.a += self.lr * (v0 - v1).mean()
    def recon(self, v): return self.down(self.up(v))

print("\n=== Part B: Convolutional RBM (6 filters of 3x3, weight sharing) ===")
crbm = CRBM(); sub = Vb[:600]; errs_c = []
for ep in range(30):
    for i in rng.permutation(len(sub)): crbm.step(sub[i].reshape(8, 8))
    errs_c.append(np.mean([(sub[i].reshape(8, 8) - crbm.recon(sub[i].reshape(8, 8))) ** 2 for i in range(0, 600, 6)]))
    if ep % 5 == 0: print(f"  CRBM epoch {ep+1:3d}: reconstruction MSE {errs_c[-1]:.4f}")
print(f"Parameters: CRBM = {crbm.W.size + crbm.K + 1} (shared filters) vs dense RBM with the same number of hidden units (6 x 6 x 6 = 216) = {64 * 216 + 216 + 64}")
print(f"CRBM final reconstruction MSE: {errs_c[-1]:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(errs_dbm, label="DBM (64-40-20)"); ax[0].plot(errs_rbm, label="RBM (64-40)"); ax[0].plot(errs_c, label="Convolutional RBM"); ax[0].set_title("Reconstruction MSE during training"); ax[0].legend(); ax[0].set_xlabel("epoch"); ax[0].grid()
g = np.zeros((8 * 4, 8 * 8))
for j in range(32): g[(j // 8) * 8:(j // 8 + 1) * 8, (j % 8) * 8:(j % 8 + 1) * 8] = v_c[j % len(v_c)].reshape(8, 8) if j < len(v_c) else 0
ax[1].imshow(g, cmap="gray_r"); ax[1].set_title("DBM samples (persistent Gibbs chains)"); ax[1].axis("off")
ax[2].imshow(Vb[:32].reshape(4, 8, 8, 8).transpose(0, 2, 1, 3).reshape(32, 64), cmap="gray_r"); ax[2].set_title("Real (binarised) digits"); ax[2].axis("off")
savefig(fig, OUT, "dbm_crbm_training.png")
fig, ax = plt.subplots(1, 6, figsize=(13, 2.5))
for k in range(6): ax[k].imshow(crbm.W[k], cmap="RdBu"); ax[k].axis("off"); ax[k].set_title(f"filter {k+1}", fontsize=8)
fig.suptitle("CRBM 3x3 filters"); savefig(fig, OUT, "crbm_filters.png")
fig, ax = plt.subplots(3, 8, figsize=(13, 5.2)); h_ = crbm.up(sub[0].reshape(8, 8))
for j in range(8):
    v = sub[j].reshape(8, 8); ax[0, j].imshow(v, cmap="gray_r"); ax[1, j].imshow(crbm.recon(v), cmap="gray_r"); ax[2, j].imshow(crbm.up(v)[j % 6], cmap="viridis"); [ax[i, j].axis("off") for i in range(3)]
ax[0, 0].set_title("input", fontsize=8, loc="left"); ax[1, 0].set_title("CRBM reconstruction", fontsize=8, loc="left"); ax[2, 0].set_title("a feature map", fontsize=8, loc="left")
savefig(fig, OUT, "crbm_reconstructions.png")
