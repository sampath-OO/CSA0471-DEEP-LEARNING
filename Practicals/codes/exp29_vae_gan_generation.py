"""Experiment 29 (CO5): VAE/GAN Sample Generation
Libraries: TensorFlow, Keras, NumPy, Matplotlib
Requires TensorFlow. Trains a VAE and a GAN on 20k MNIST digits and generates new samples."""
import numpy as np
import matplotlib.pyplot as plt
from common import out_dir, savefig, need_tf

need_tf("Experiment 29")
from tensorflow import keras
import gen_models as gm

OUT = out_dir(__file__); keras.utils.set_random_seed(0)
X, y, Xt, yt = gm.load_mnist(20000)

vae = gm.VAE(); vae.compile(keras.optimizers.Adam(1e-3)); h = vae.fit(X, epochs=25, batch_size=128, verbose=2).history
G, D = gm.build_gan(); gh = gm.train_gan(G, D, X, epochs=30)

def grid(imgs, rows=8, cols=8): return imgs[:rows * cols].reshape(rows, cols, 28, 28).transpose(0, 2, 1, 3).reshape(rows * 28, cols * 28)

fig, ax = plt.subplots(1, 3, figsize=(18, 6)); ax[0].imshow(grid(X[:64]), cmap="gray"); ax[0].set_title("Real MNIST")
ax[1].imshow(grid(vae.sample(64)), cmap="gray"); ax[1].set_title("VAE samples (z ~ N(0, I))")
ax[2].imshow(grid(G(np.random.normal(size=(64, gm.LATENT)).astype("float32")).numpy()), cmap="gray"); ax[2].set_title("GAN samples")
for a in ax: a.axis("off")
savefig(fig, OUT, "generated_samples.png")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5)); ax[0].plot(h["loss"]); ax[0].set_title("VAE loss (neg. ELBO)"); ax[0].grid()
ax[1].plot(gh[:, 0], label="discriminator"); ax[1].plot(gh[:, 1], label="generator"); ax[1].legend(); ax[1].set_title("GAN losses"); ax[1].grid()
savefig(fig, OUT, "training_curves.png")

# VAE reconstruction and latent interpolation
mu, _ = vae.encode(Xt[:2]); alphas = np.linspace(0, 1, 10)[:, None]; z = (1 - alphas) * mu[0].numpy() + alphas * mu[1].numpy()
fig, ax = plt.subplots(2, 10, figsize=(14, 3.2))
for i in range(10): ax[0, i].imshow(vae.dec(z[i:i + 1]).numpy().reshape(28, 28), cmap="gray"); ax[0, i].axis("off")
zg = np.random.normal(size=(2, gm.LATENT)); zz = (1 - alphas) * zg[0] + alphas * zg[1]
for i in range(10): ax[1, i].imshow(G(zz[i:i + 1].astype("float32")).numpy().reshape(28, 28), cmap="gray"); ax[1, i].axis("off")
fig.suptitle("Latent-space interpolation: VAE (top) and GAN (bottom)"); savefig(fig, OUT, "latent_interpolation.png")
