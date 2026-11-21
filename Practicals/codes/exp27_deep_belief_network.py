"""Experiment 27 (CO5): Deep Belief Network Representation Learning
Libraries: NumPy, SciPy, TensorFlow, Keras, Matplotlib
Greedy layer-wise pre-training of stacked RBMs (64-48-24) implemented in NumPy; features are evaluated with a logistic-regression
read-out. If TensorFlow is installed a Keras MLP initialised with the DBN weights is fine-tuned as well."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from common import out_dir, savefig, RBM, have_tf

OUT = out_dir(__file__)
X, y = load_digits(return_X_y=True); X = X / 16
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)

sizes = [64, 48, 24]; rbms = []; inp = Xtr; curves = []
for i in range(len(sizes) - 1):
    r = RBM(sizes[i], sizes[i + 1], lr=.05, seed=i); errs = r.fit(inp, epochs=40, batch=20, k=1); curves.append(errs); rbms.append(r)
    print(f"RBM layer {i+1} ({sizes[i]} -> {sizes[i+1]}): reconstruction MSE {errs[0]:.4f} -> {errs[-1]:.4f}")
    inp = r.ph(inp)                                                       # hidden probabilities feed the next RBM

def encode(A, upto=None):
    for r in rbms[:upto]: A = r.ph(A)
    return A

def score(Ftr, Fte, C=100.0): return LogisticRegression(max_iter=3000, C=C).fit(Ftr, ytr).score(Fte, yte)
res = {"Raw pixels (64)": score(Xtr, Xte), "DBN layer 1 (48)": score(encode(Xtr, 1), encode(Xte, 1)), "DBN layer 2 (24)": score(encode(Xtr), encode(Xte)),
       "PCA (24)": score(PCA(24).fit(Xtr).transform(Xtr), PCA(24).fit(Xtr).transform(Xte))}
print("\nLogistic-regression test accuracy on different representations:")
for n, a in res.items(): print(f"  {n:18s}: {a:.4f}")

# few-label regime: only 50 labelled samples
idx = np.random.default_rng(0).choice(len(Xtr), 50, replace=False); print("\nWith only 50 labelled training samples:")
for n, (A, B) in {"Raw pixels": (Xtr, Xte), "DBN layer 2": (encode(Xtr), encode(Xte))}.items():
    print(f"  {n:12s}: {LogisticRegression(max_iter=3000, C=100.0).fit(A[idx], ytr[idx]).score(B, yte):.4f}")

# generate by top-down pass from layer-2 hidden states with Gibbs sampling in top RBM
top = rbms[-1]; h = (np.random.default_rng(1).random((16, 24)) < .5).astype(float)
for _ in range(200): v48 = top.sample(top.pv(h)); h = top.sample(top.ph(v48))       # alternating Gibbs sampling in the top RBM
gen = rbms[0].pv(top.sample(top.pv(h)))                                               # top-down pass to pixel space

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for i, c in enumerate(curves): ax[0].plot(c, label=f"RBM {i+1}")
ax[0].legend(); ax[0].set_title("Layer-wise pre-training (reconstruction MSE)"); ax[0].set_xlabel("epoch"); ax[0].grid()
ax[1].barh(list(res), list(res.values())); ax[1].set_xlim(.8, 1); ax[1].set_title("Read-out accuracy by representation")
Z = PCA(2).fit_transform(encode(Xte)); sc = ax[2].scatter(Z[:, 0], Z[:, 1], c=yte, cmap="tab10", s=10); ax[2].set_title("DBN top-layer features (PCA 2-D)"); plt.colorbar(sc, ax=ax[2])
savefig(fig, OUT, "dbn_representations.png")
fig, ax = plt.subplots(2, 8, figsize=(13, 3.6))
for j in range(8): ax[0, j].imshow(Xte[j].reshape(8, 8), cmap="gray_r"); ax[1, j].imshow(rbms[0].pv(rbms[0].ph(Xte[j:j + 1]))[0].reshape(8, 8), cmap="gray_r"); ax[0, j].axis("off"); ax[1, j].axis("off")
ax[0, 0].set_title("input", fontsize=8, loc="left"); ax[1, 0].set_title("layer-1 reconstruction", fontsize=8, loc="left"); savefig(fig, OUT, "dbn_reconstructions.png")
fig, ax = plt.subplots(2, 8, figsize=(13, 3.6))
for j in range(16): ax.flat[j].imshow(gen[j].reshape(8, 8), cmap="gray_r"); ax.flat[j].axis("off")
fig.suptitle("Samples generated from the DBN (Gibbs sampling in top RBM, then top-down pass)"); savefig(fig, OUT, "dbn_generated.png")

if have_tf():
    from tensorflow import keras
    for init in [False, True]:
        keras.utils.set_random_seed(0)
        m = keras.Sequential([keras.layers.Input((64,)), keras.layers.Dense(48, activation="sigmoid"), keras.layers.Dense(24, activation="sigmoid"), keras.layers.Dense(10, activation="softmax")])
        if init:
            for k, r in enumerate(rbms): m.layers[k].set_weights([r.W, r.b])
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"]); m.fit(Xtr[idx], ytr[idx], epochs=60, verbose=0)
        print(f"Keras fine-tuning with 50 labels, DBN-initialised={init}: test acc {m.evaluate(Xte, yte, verbose=0)[1]:.4f}")
