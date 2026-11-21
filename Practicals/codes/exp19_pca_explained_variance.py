"""Experiment 19 (CO4): PCA and Explained Variance Analysis
Libraries: NumPy, Scikit-learn, Matplotlib"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from common import out_dir, savefig

OUT = out_dir(__file__)
X, y = load_digits(return_X_y=True); Xc = X - X.mean(0)
pca = PCA().fit(X); evr = pca.explained_variance_ratio_; cum = np.cumsum(evr)
print("Data:", X.shape)
print("Variance explained by first 5 PCs:", evr[:5].round(4))
for t in [.80, .90, .95, .99]: print(f"Components needed for {int(t*100)}% variance: {np.searchsorted(cum, t) + 1}")

# manual PCA through the SVD, checked against scikit-learn
U, S, Vt = np.linalg.svd(Xc, full_matrices=False); ev = S ** 2 / (len(X) - 1)
print("Manual SVD eigenvalues match sklearn:", np.allclose(ev, pca.explained_variance_))
print("Principal axes match (up to sign):", np.allclose(np.abs(Vt[:10]), np.abs(pca.components_[:10]), atol=1e-6))

# reconstruction error vs number of components
ks = [1, 2, 5, 10, 20, 30, 40, 64]; errs = []
for k in ks:
    p = PCA(k).fit(X); errs.append(np.mean((X - p.inverse_transform(p.transform(X))) ** 2))
print("\nk | reconstruction MSE"); [print(f"{k:2d} | {e:.4f}") for k, e in zip(ks, errs)]

# classification accuracy vs number of components
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y); accs = []
for k in ks:
    p = PCA(k).fit(Xtr); accs.append(LogisticRegression(max_iter=2000).fit(p.transform(Xtr), ytr).score(p.transform(Xte), yte))
print("\nk | accuracy of logistic regression on PCA features"); [print(f"{k:2d} | {a:.4f}") for k, a in zip(ks, accs)]

fig, ax = plt.subplots(2, 3, figsize=(17, 10))
ax[0, 0].bar(range(1, 21), evr[:20]); ax[0, 0].set_title("Scree plot (explained variance ratio)"); ax[0, 0].set_xlabel("component")
ax[0, 1].plot(range(1, 65), cum, "o-", ms=3); [ax[0, 1].axhline(t, ls="--", c="gray") for t in (.9, .95)]; ax[0, 1].set_title("Cumulative explained variance"); ax[0, 1].set_xlabel("components"); ax[0, 1].grid()
Z = pca.transform(X)[:, :2]; sc = ax[0, 2].scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8); plt.colorbar(sc, ax=ax[0, 2]); ax[0, 2].set_title("Digits projected on PC1-PC2")
ax[1, 0].plot(ks, errs, "o-"); ax[1, 0].set_title("Reconstruction MSE vs components"); ax[1, 0].grid()
ax[1, 1].plot(ks, accs, "o-"); ax[1, 1].set_title("Classifier accuracy vs components"); ax[1, 1].grid()
ax[1, 2].axis("off")
sub = fig.add_gridspec(2, 3)[1, 2].subgridspec(2, 5)
for i in range(10): a = fig.add_subplot(sub[i // 5, i % 5]); a.imshow(pca.components_[i].reshape(8, 8), cmap="RdBu"); a.axis("off"); a.set_title(f"PC{i+1}", fontsize=7)
savefig(fig, OUT, "pca_analysis.png")

fig, ax = plt.subplots(5, 8, figsize=(13, 8)); ids = [0, 1, 2, 3, 4, 5, 6, 7]
for j, k in enumerate([2, 5, 10, 20]):
    p = PCA(k).fit(X); R = p.inverse_transform(p.transform(X[ids]))
    for i in range(8): ax[j + 1, i].imshow(R[i].reshape(8, 8), cmap="gray_r"); ax[j + 1, i].axis("off"); 
    ax[j + 1, 0].set_title(f"k={k}", fontsize=8, loc="left")
for i in range(8): ax[0, i].imshow(X[ids[i]].reshape(8, 8), cmap="gray_r"); ax[0, i].axis("off")
ax[0, 0].set_title("original", fontsize=8, loc="left"); savefig(fig, OUT, "pca_reconstructions.png")
