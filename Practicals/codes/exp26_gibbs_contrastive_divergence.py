"""Experiment 26 (CO5): Gibbs Sampling and Contrastive Divergence Analysis
Libraries: NumPy, SciPy, Matplotlib
A tiny RBM (8 visible, 4 hidden) lets us compute the exact log-likelihood by enumeration and compare CD-k variants."""
import numpy as np, itertools
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from common import out_dir, savefig, RBM

OUT = out_dir(__file__); rng = np.random.default_rng(0)
NV, NH = 8, 4
patterns = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0]], float)
data = np.array([patterns[rng.integers(0, 3)] for _ in range(600)]); data = np.abs(data - (rng.random(data.shape) < .05))   # 5% bit-flip noise
allv = np.array(list(itertools.product([0., 1.], repeat=NV)))

def exact_ll(r):
    """Average log-likelihood of the data under the RBM (exact partition function by enumeration)."""
    logp_unnorm = -r.free_energy(allv); logZ = logsumexp(logp_unnorm)
    return float(np.mean(-r.free_energy(data)) - logZ), np.exp(logp_unnorm - logZ)

# ---- 1. Gibbs sampling convergence to the model distribution --------------------------------
r = RBM(NV, NH, lr=.1, seed=1); r.fit(data, epochs=60, batch=20, k=5); _, p_model = exact_ll(r)
print("Trained tiny RBM. Top-3 most probable visible states (exact):")
for i in np.argsort(p_model)[::-1][:3]: print("  ", allv[i].astype(int), f"p = {p_model[i]:.4f}")
chains = 2000; v = (rng.random((chains, NV)) < .5).astype(float); tv = []; idx_of = lambda M: (M @ (2 ** np.arange(NV - 1, -1, -1))).astype(int)
for step in range(1, 61):
    _, v = r.gibbs_vhv(v); emp = np.bincount(idx_of(v), minlength=2 ** NV) / chains; tv.append(.5 * np.abs(emp - p_model).sum())
print(f"\nTotal-variation distance between {chains} Gibbs chains and the exact model distribution: step 1 = {tv[0]:.3f}, step 10 = {tv[9]:.3f}, step 60 = {tv[-1]:.3f}")

# ---- 2. CD-k comparison ----------------------------------------------------------------------------------
res = {}
for k in [1, 5, 20]:
    rr = RBM(NV, NH, lr=.05, seed=2); lls = []
    for ep in range(80):
        idx = rng.permutation(len(data))
        for s in range(0, len(data), 20): rr.cd_step(data[idx[s:s + 20]], k)
        lls.append(exact_ll(rr)[0])
    res[f"CD-{k}"] = (lls, rr)
print("\nExact average log-likelihood (higher is better; max possible = -entropy of the data):")
for n, (l, rr) in res.items(): print(f"  {n:5s}: epoch 10 = {l[9]:.3f}, epoch 40 = {l[39]:.3f}, epoch 80 = {l[-1]:.3f} | recon MSE {rr.recon_error(data):.4f}")

# ---- 3. Persistent CD (negative chains kept between updates) ------------------------------------------------
rp = RBM(NV, NH, lr=.05, seed=2); chain = (rng.random((20, NV)) < .5).astype(float); lls = []
for ep in range(80):
    idx = rng.permutation(len(data))
    for s in range(0, len(data), 20):
        v0 = data[idx[s:s + 20]]; ph0 = rp.ph(v0); _, chain = rp.gibbs_vhv(chain); phk = rp.ph(chain)
        rp.W += rp.lr * (v0.T @ ph0 - chain.T @ phk) / 20; rp.a += rp.lr * (v0 - chain).mean(0); rp.b += rp.lr * (ph0 - phk).mean(0)
    lls.append(exact_ll(rp)[0])
res["PCD-1"] = (lls, rp); print(f"  PCD-1: epoch 10 = {lls[9]:.3f}, epoch 40 = {lls[39]:.3f}, epoch 80 = {lls[-1]:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(range(1, 61), tv); ax[0].set_title("Gibbs chains → model distribution (TV distance)"); ax[0].set_xlabel("Gibbs step"); ax[0].grid()
for n, (l, _) in res.items(): ax[1].plot(l, label=n)
ax[1].set_title("Exact data log-likelihood during training"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid()
ax[2].bar(range(2 ** NV), p_model, width=1.0); ax[2].set_title("Exact model distribution over 256 visible states"); ax[2].set_xlabel("state index")
savefig(fig, OUT, "gibbs_cd_analysis.png")
