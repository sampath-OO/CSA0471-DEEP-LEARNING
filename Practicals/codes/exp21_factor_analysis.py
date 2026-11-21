"""Experiment 21 (CO4): Factor Analysis and Latent Factor Interpretation
Libraries: Scikit-learn, NumPy, Pandas, Matplotlib"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from common import out_dir, savefig

OUT = out_dir(__file__)
data = load_wine(); df = pd.DataFrame(data.data, columns=data.feature_names); X = StandardScaler().fit_transform(df)
print("Wine data:", df.shape, "| classes:", list(data.target_names))

# choose number of factors by cross-validated log-likelihood
ks = range(1, 9); cv = [cross_val_score(FactorAnalysis(k, random_state=0), X, cv=5).mean() for k in ks]
print("factors | CV log-likelihood"); [print(f"{k:7d} | {s:.4f}") for k, s in zip(ks, cv)]
best = list(ks)[int(np.argmax(cv))]; print("Best number of factors by CV:", best)

K = 3
fa = FactorAnalysis(K, rotation="varimax", random_state=0).fit(X)
L = pd.DataFrame(fa.components_.T, index=df.columns, columns=[f"F{i+1}" for i in range(K)])
print(f"\nVarimax-rotated loadings ({K} factors):\n", L.round(3))
comm = (L ** 2).sum(1); print("\nCommunalities (variance explained by common factors):\n", comm.round(3))
print("Uniqueness (noise variance):\n", pd.Series(fa.noise_variance_, index=df.columns).round(3))
print("\nSS loadings per factor:", (L ** 2).sum().round(3).values, "| total variance explained: %.1f%%" % (100 * (L ** 2).values.sum() / X.shape[1]))
print("\nInterpretation: top-loading variables per factor")
for f in L:
    top = L[f].abs().sort_values(ascending=False).head(4).index; print(f"  {f}:", ", ".join(f"{t} ({L.loc[t, f]:+.2f})" for t in top))

Fs = fa.transform(X); print("\nFactor score correlation with class label:", [round(np.corrcoef(Fs[:, i], data.target)[0, 1], 3) for i in range(K)])
fig, ax = plt.subplots(1, 3, figsize=(20, 6.5))
ax[0].plot(list(ks), cv, "o-"); ax[0].set_title("Choosing #factors (CV log-likelihood)"); ax[0].set_xlabel("factors"); ax[0].grid()
im = ax[1].imshow(L.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto"); ax[1].set_xticks(range(K)); ax[1].set_xticklabels(L.columns)
ax[1].set_yticks(range(len(L))); ax[1].set_yticklabels(L.index, fontsize=8); plt.colorbar(im, ax=ax[1]); ax[1].set_title("Varimax factor loadings")
sc = ax[2].scatter(Fs[:, 0], Fs[:, 1], c=data.target, cmap="viridis", s=20); ax[2].set_xlabel("Factor 1"); ax[2].set_ylabel("Factor 2"); ax[2].set_title("Factor scores coloured by wine class"); plt.colorbar(sc, ax=ax[2])
savefig(fig, OUT, "factor_analysis.png")
