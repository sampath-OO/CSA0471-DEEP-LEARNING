"""Experiment 25 (CO5): Restricted Boltzmann Machine Energy and Reconstruction
Libraries: NumPy, SciPy, Matplotlib"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from common import out_dir, savefig, RBM

OUT = out_dir(__file__); rng = np.random.default_rng(0)
V = (load_digits().data / 16 > .4).astype(float); print("Binarised 8x8 digits:", V.shape)
rbm = RBM(64, 36, lr=.08, seed=0)

e0 = rbm.free_energy(V).mean(); noise = (rng.random(V.shape) < .5).astype(float)
print(f"Before training: mean free energy of data = {e0:.2f}, of random noise = {rbm.free_energy(noise).mean():.2f}")
errs, fe = [], []
def cb(ep, r):
    fe.append((r.free_energy(V).mean(), r.free_energy(noise).mean()))
    if ep % 5 == 0: print(f"epoch {ep+1:3d}: reconstruction MSE {r.recon_error(V):.4f} | free energy data {fe[-1][0]:8.2f} | noise {fe[-1][1]:8.2f}")
errs = rbm.fit(V, epochs=40, batch=20, k=1, callback=cb); fe = np.array(fe)
print(f"\nAfter training: recon MSE {errs[-1]:.4f}; data should have LOWER free energy than noise: {fe[-1,0]:.2f} < {fe[-1,1]:.2f}")

# energy of a joint (v,h) configuration and its relation to free energy
v = V[:1]; h = rbm.sample(rbm.ph(v)); print("Energy E(v,h) of first digit with a sampled hidden state:", rbm.energy(v, h).round(3))
print("Check F(v) = -log sum_h exp(-E(v,h)) via brute-force on a 10-unit sub-model:")
sub = RBM(6, 4, seed=1); sub.W = rng.normal(0, 1, (6, 4)); vv = rng.integers(0, 2, (1, 6)).astype(float)
brute = -np.log(sum(np.exp(-sub.energy(vv, np.array([[int(b) for b in format(i, '04b')]], float))[0]) for i in range(16)))
print(f"   brute force {brute:.6f}  vs  analytic free energy {sub.free_energy(vv)[0]:.6f}")

rec_p = rbm.pv(rbm.ph(V)); idx = np.arange(12)
fig, ax = plt.subplots(3, 12, figsize=(16, 5))
cor = V.copy(); cor[idx, :16] = 0                                                   # occlude the top 2 rows
for j, i in enumerate(idx):
    ax[0, j].imshow(V[i].reshape(8, 8), cmap="gray_r"); ax[1, j].imshow(rec_p[i].reshape(8, 8), cmap="gray_r")
    ax[2, j].imshow(rbm.pv(rbm.ph(cor[i:i + 1]))[0].reshape(8, 8), cmap="gray_r"); [ax[k, j].axis("off") for k in range(3)]
ax[0, 0].set_title("data", fontsize=8, loc="left"); ax[1, 0].set_title("reconstruction", fontsize=8, loc="left"); ax[2, 0].set_title("from occluded input", fontsize=8, loc="left")
savefig(fig, OUT, "rbm_reconstructions.png")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(errs); ax[0].set_title("Reconstruction MSE per epoch"); ax[0].set_xlabel("epoch"); ax[0].grid()
ax[1].plot(fe[:, 0], label="data"); ax[1].plot(fe[:, 1], label="random noise"); ax[1].legend(); ax[1].set_title("Mean free energy F(v)"); ax[1].set_xlabel("epoch"); ax[1].grid()
ax[2].hist(rbm.free_energy(V), 40, alpha=.6, label="data"); ax[2].hist(rbm.free_energy(noise), 40, alpha=.6, label="noise"); ax[2].legend(); ax[2].set_title("Free-energy distribution after training")
savefig(fig, OUT, "rbm_energy.png")
fig, ax = plt.subplots(4, 9, figsize=(13, 6.5))
for a, k in zip(ax.flat, range(36)): a.imshow(rbm.W[:, k].reshape(8, 8), cmap="RdBu"); a.axis("off")
fig.suptitle("Learned RBM weights (receptive field of each hidden unit)"); savefig(fig, OUT, "rbm_filters.png")
