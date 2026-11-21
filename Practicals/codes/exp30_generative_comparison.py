"""Experiment 30 (CO5): Generative Model Output Comparison and Evaluation
Libraries: TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib
Requires TensorFlow. Compares VAE and GAN samples with real data using (i) a classifier-based Inception-style score,
(ii) a Fréchet distance in PCA space, (iii) sample diversity and (iv) per-pixel statistics."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from common import out_dir, savefig, need_tf

need_tf("Experiment 30")
from tensorflow import keras
import gen_models as gm

OUT = out_dir(__file__); keras.utils.set_random_seed(0); rng = np.random.default_rng(0)
X, y, Xt, yt = gm.load_mnist(20000)
vae = gm.VAE(); vae.compile(keras.optimizers.Adam(1e-3)); vae.fit(X, epochs=20, batch_size=128, verbose=0)
G, D = gm.build_gan(); gm.train_gan(G, D, X, epochs=25)
N = 2000
samples = {"Real (held-out)": Xt[:N], "VAE": vae.sample(N), "GAN": G(rng.normal(size=(N, gm.LATENT)).astype("float32")).numpy(), "Noise (reference)": rng.random((N, 784)).astype("float32")}

clf = LogisticRegression(max_iter=300).fit(X, y); pca = PCA(30).fit(X)
def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0); c1, c2 = np.cov(a.T), np.cov(b.T); cs = sqrtm(c1 @ c2).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * cs))
def inception_score(p): py = p.mean(0); return float(np.exp(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(py + 1e-12)), 1))))

real_z = pca.transform(Xt[:N]); rows = []
for name, s in samples.items():
    p = clf.predict_proba(s); d = np.linalg.norm(s[:500, None] - s[None, :500], axis=-1); div = d[np.triu_indices(500, 1)].mean()
    rows.append(dict(model=name, IS_like=inception_score(p), mean_confidence=float(p.max(1).mean()), frechet_pca=frechet(real_z, pca.transform(s)),
                     diversity=float(div), pixel_mean=float(s.mean()), pixel_std=float(s.std()), classes_covered=int(len(np.unique(p.argmax(1))))))
df = pd.DataFrame(rows).round(3); print(df.to_string(index=False)); df.to_csv(f"{OUT}/metrics.csv", index=False)

fig, ax = plt.subplots(1, 4, figsize=(20, 5)); 
for a, (n, s) in zip(ax, samples.items()): a.imshow(s[:36].reshape(6, 6, 28, 28).transpose(0, 2, 1, 3).reshape(168, 168), cmap="gray"); a.set_title(n); a.axis("off")
savefig(fig, OUT, "sample_comparison.png")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5)); sub = df[df.model != "Noise (reference)"]
ax[0].bar(sub.model, sub.IS_like); ax[0].set_title("Inception-style score (higher better)")
ax[1].bar(sub.model, sub.frechet_pca); ax[1].set_title("Fréchet distance in PCA space (lower better)")
for n, s in list(samples.items())[:3]: ax[2].hist(clf.predict(s), bins=np.arange(11) - .5, alpha=.5, label=n, density=True)
ax[2].legend(); ax[2].set_title("Predicted class histogram (mode coverage)")
savefig(fig, OUT, "generative_metrics.png")
