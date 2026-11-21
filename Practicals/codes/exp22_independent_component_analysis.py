"""Experiment 22 (CO4): Independent Component Analysis and Component Visualization
Libraries: Scikit-learn, NumPy, Matplotlib  (blind source separation / 'cocktail party' problem)"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from common import out_dir, savefig

OUT = out_dir(__file__)
rng = np.random.default_rng(0); t = np.linspace(0, 40, 4000)
S = np.c_[np.sin(2 * t), np.sign(np.sin(3 * t)), signal.sawtooth(2 * np.pi * t * .37)]; S += .05 * rng.normal(size=S.shape); S /= S.std(0)
A = np.array([[1, 1, 1], [.5, 2, 1], [1.5, 1, 2]]); Xm = S @ A.T
print("True mixing matrix A:\n", A, "\nMixed signals:", Xm.shape)

ica = FastICA(3, whiten="unit-variance", random_state=0, max_iter=2000, tol=1e-6); Sh = ica.fit_transform(Xm)
pca = PCA(3, whiten=True).fit(Xm); Sp = pca.transform(Xm)
def match(est, true):
    C = np.abs(np.corrcoef(est.T, true.T)[:3, 3:]); return C, C.argmax(0), C.max(0)
Ci, ii, mi = match(Sh, S); Cp, ip, mp = match(Sp, S)
print("\n|correlation| between estimated and true sources")
print("FastICA:\n", Ci.round(3), "\nbest-match per true source:", mi.round(3))
print("PCA (whitened):\n", Cp.round(3), "\nbest-match per true source:", mp.round(3))
print(f"\nMean best-match correlation: ICA = {mi.mean():.4f}, PCA = {mp.mean():.4f}")
from scipy.stats import kurtosis
print("Excess kurtosis of true sources:", kurtosis(S).round(3), "\nof ICA components:", kurtosis(Sh).round(3), "\nof PCA components:", kurtosis(Sp).round(3), "\nof mixtures:", kurtosis(Xm).round(3))
print(f"ICA converged in {ica.n_iter_} iterations")

fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
for a, d, n in zip(ax, [S, Xm, Sh, Sp], ["True sources", "Observed mixtures", "FastICA estimates", "PCA components"]):
    for i in range(3): a.plot(t[:500], d[:500, i] + 4 * i, lw=1)
    a.set_title(n); a.set_yticks([])
savefig(fig, OUT, "ica_signals.png")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, d, n in zip(ax, [S, Xm, Sh], ["Sources (non-Gaussian)", "Mixtures", "ICA recovered"]): a.scatter(d[:, 0], d[:, 1], s=4, alpha=.4); a.set_title(n); a.set_aspect("equal")
savefig(fig, OUT, "ica_scatter.png")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, M, n in zip(ax, [Ci, Cp, np.abs(np.linalg.pinv(ica.mixing_) @ A)], ["ICA |corr|", "PCA |corr|", "ICA unmixing x A (≈ permutation·scale)"]):
    a.imshow(M, cmap="viridis"); a.set_title(n); [a.text(j, i, f"{v:.2f}", ha="center", color="w") for (i, j), v in np.ndenumerate(M)]
savefig(fig, OUT, "ica_matching.png")
