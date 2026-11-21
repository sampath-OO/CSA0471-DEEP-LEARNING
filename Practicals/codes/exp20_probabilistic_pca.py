"""Experiment 20 (CO4): Probabilistic PCA and Reconstruction Analysis
Libraries: NumPy, Scikit-learn, Matplotlib
Closed-form maximum-likelihood PPCA (Tipping & Bishop, 1999) + EM version, compared with scikit-learn PCA."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from common import out_dir, savefig

OUT = out_dir(__file__)
rng = np.random.default_rng(0)

def ppca_ml(X, q):
    mu = X.mean(0); S = np.cov(X.T); vals, vecs = np.linalg.eigh(S); idx = vals.argsort()[::-1]; vals, vecs = vals[idx], vecs[:, idx]
    s2 = vals[q:].mean(); W = vecs[:, :q] * np.sqrt(np.maximum(vals[:q] - s2, 0)); return mu, W, s2

def ppca_loglik(X, mu, W, s2):
    d = X.shape[1]; C = W @ W.T + s2 * np.eye(d); S = np.cov(X.T, bias=True); _, logdet = np.linalg.slogdet(C)
    return -.5 * len(X) * (d * np.log(2 * np.pi) + logdet + np.trace(np.linalg.solve(C, S)))

def ppca_em(X, q, iters=50):
    Xc = X - X.mean(0); n, d = Xc.shape; W = rng.normal(size=(d, q)); s2 = 1.; ll = []
    for _ in range(iters):
        M = W.T @ W + s2 * np.eye(q); Minv = np.linalg.inv(M); Ez = Xc @ W @ Minv; Ezz = n * s2 * Minv + Ez.T @ Ez
        W_new = Xc.T @ Ez @ np.linalg.inv(Ezz); s2 = (np.sum(Xc ** 2) - 2 * np.sum(Ez * (Xc @ W_new)) + np.trace(Ezz @ W_new.T @ W_new)) / (n * d); W = W_new
        ll.append(ppca_loglik(X, X.mean(0), W, s2))
    return W, s2, ll

# --- synthetic data with known generative parameters ---------------------------------------
d, q, n = 10, 2, 2000; Wtrue = rng.normal(size=(d, q)) * 2; s2_true = .3
Z = rng.normal(size=(n, q)); Xs = Z @ Wtrue.T + rng.normal(0, np.sqrt(s2_true), (n, d)) + 5
mu, W, s2 = ppca_ml(Xs, q)
print(f"Synthetic data: true noise variance = {s2_true}, PPCA estimate = {s2:.4f}")
print("Subspace error ||P_true - P_est||_F =", np.linalg.norm(Wtrue @ np.linalg.pinv(Wtrue) - W @ np.linalg.pinv(W)).round(4))
We, s2e, ll = ppca_em(Xs, q); print(f"EM estimate of noise variance after 50 iterations = {s2e:.4f}; log-likelihood {ll[0]:.1f} -> {ll[-1]:.1f}")
print("Closed-form vs EM log-likelihood:", round(ppca_loglik(Xs, mu, W, s2), 2), round(ll[-1], 2))

# --- digits: reconstruction and model selection ----------------------------------------------------
Xd = load_digits().data; Xd = Xd + 0.0; qs = [1, 2, 5, 10, 15, 20, 30, 40, 50]; rec, lls, s2s = [], [], []
Xn = Xd + rng.normal(0, 3.0, Xd.shape)                        # noisy copy to illustrate denoising
Xd_c = Xd[:, Xd.std(0) > 0]; Xn_c = Xn[:, Xd.std(0) > 0]
for k in qs:
    mu_, W_, s_ = ppca_ml(Xd_c, k); M = W_.T @ W_ + s_ * np.eye(k)
    post = (Xd_c - mu_) @ W_ @ np.linalg.inv(M); R = post @ W_.T + mu_                      # posterior-mean reconstruction
    rec.append(np.mean((Xd_c - R) ** 2)); lls.append(ppca_loglik(Xd_c, mu_, W_, s_) / len(Xd_c)); s2s.append(s_)
sk = PCA(10).fit(Xd_c); print("\nq | recon MSE | noise var sigma^2 | avg log-lik")
for k, r, s_, l in zip(qs, rec, s2s, lls): print(f"{k:2d} | {r:9.4f} | {s_:9.4f} | {l:9.2f}")
print("scikit-learn PCA(10).score (avg PPCA log-lik):", round(sk.score(Xd_c), 2), "vs ours q=10:", round(lls[qs.index(10)], 2))
n_par = [k * (Xd_c.shape[1] - (k - 1) / 2) + 1 + Xd_c.shape[1] for k in qs]
bic = [-2 * l * len(Xd_c) + p * np.log(len(Xd_c)) for l, p in zip(lls, n_par)]; print("BIC-selected q:", qs[int(np.argmin(bic))])

mu_, W_, s_ = ppca_ml(Xn_c, 15); M = W_.T @ W_ + s_ * np.eye(15); den = ((Xn_c - mu_) @ W_ @ np.linalg.inv(M)) @ W_.T + mu_
print(f"Denoising: MSE noisy->clean {np.mean((Xn_c - Xd_c)**2):.3f}, PPCA(15) reconstruction->clean {np.mean((den - Xd_c)**2):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(ll, "o-"); ax[0].set_title("PPCA EM log-likelihood (synthetic)"); ax[0].set_xlabel("iteration"); ax[0].grid()
ax[1].plot(qs, rec, "o-", label="reconstruction MSE"); ax[1].set_xlabel("latent dim q"); ax[1].set_title("Reconstruction error (digits)"); ax[1].grid()
ax[2].plot(qs, bic, "s-"); ax[2].set_title("BIC for model selection"); ax[2].set_xlabel("latent dim q"); ax[2].grid()
savefig(fig, OUT, "ppca_analysis.png")
ids = range(8); fig, ax = plt.subplots(3, 8, figsize=(13, 5.5)); full = lambda a: np.zeros((len(a), 64)) + 0
def put(a): o = np.zeros((a.shape[0], 64)); o[:, Xd.std(0) > 0] = a; return o
for i in ids:
    ax[0, i].imshow(Xd[i].reshape(8, 8), cmap="gray_r"); ax[1, i].imshow(put(Xn_c[i:i + 1])[0].reshape(8, 8), cmap="gray_r"); ax[2, i].imshow(put(den[i:i + 1])[0].reshape(8, 8), cmap="gray_r")
    [ax[j, i].axis("off") for j in range(3)]
ax[0, 0].set_title("clean", fontsize=8, loc="left"); ax[1, 0].set_title("noisy", fontsize=8, loc="left"); ax[2, 0].set_title("PPCA q=15", fontsize=8, loc="left")
savefig(fig, OUT, "ppca_denoising.png")
