"""Experiment 17 (CO3): RNN Gradient Analysis and Gradient Clipping
Libraries: TensorFlow, Keras, NumPy, Matplotlib
NumPy RNN trained with plain SGD on a long-sequence task: unclipped vs. clip-by-norm. Keras equivalent (clipnorm) shown if TF exists."""
import numpy as np
import matplotlib.pyplot as plt
from common import out_dir, savefig, have_tf, rnn_init, rnn_train, rnn_loss_grads, grad_norm

OUT = out_dir(__file__)
rng = np.random.default_rng(0)
T, N = 50, 256
X = rng.normal(size=(N, T, 1)); y = np.tanh(X[:, -5:, 0].sum(1) * .5)        # target depends on the last 5 inputs
Xv = rng.normal(size=(128, T, 1)); yv = np.tanh(Xv[:, -5:, 0].sum(1) * .5)

# ---- gradient-norm analysis at initialisation for several spectral radii ----------------
print("Gradient norm at initialisation (T=50):")
for rho in [.5, .9, 1.3, 2.0, 3.0]:
    _, g, _ = rnn_loss_grads(rnn_init(1, 20, rho=rho, seed=2), X, y); print(f"  rho(Wh)={rho:3.1f}: ||grad|| = {grad_norm(g):10.4f}")

# ---- training with / without clipping ----------------------------------------------------
runs = {}
for name, clip in [("No clipping", None), ("Clip norm = 1.0", 1.0), ("Clip norm = 0.25", .25)]:
    P = rnn_init(1, 20, rho=3.0, seed=2)
    runs[name] = rnn_train(P, X, y, epochs=30, lr=.3, opt="sgd", clip=clip, batch=32, Xv=Xv, yv=yv)
    h = runs[name]; print(f"{name:17s}: final train loss {h['loss'][-1]:.4f} | val loss {h['val_loss'][-1]:.4f} | max grad-norm seen {max(h['grad_norm']):.2f}")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for n, h in runs.items():
    ax[0].plot(h["grad_norm"], label=n); ax[1].plot(h["loss"], label=n); ax[2].plot(h["val_loss"], label=n)
ax[0].set_yscale("log"); ax[0].set_title("Max gradient norm per epoch (before clipping)"); ax[1].set_title("Training loss"); ax[2].set_title("Validation loss")
for a in ax: a.legend(); a.grid(); a.set_xlabel("epoch")
savefig(fig, OUT, "gradient_clipping.png")

# Gradient-norm histogram over mini-batches at initialisation
norms = [grad_norm(rnn_loss_grads(rnn_init(1, 20, rho=3.0, seed=2), X[i:i + 32], y[i:i + 32])[1]) for i in range(0, N, 32)]
fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(range(len(norms)), norms); ax.axhline(1.0, c="r", ls="--", label="clip threshold 1.0"); ax.legend()
ax.set_xlabel("mini-batch"); ax.set_ylabel("gradient norm"); ax.set_title("Mini-batch gradient norms at initialisation (ρ=3.0)")
savefig(fig, OUT, "gradient_norm_batches.png")

if have_tf():
    from tensorflow import keras
    for clip in [None, 1.0]:
        keras.utils.set_random_seed(0)
        m = keras.Sequential([keras.layers.Input((T, 1)), keras.layers.SimpleRNN(20), keras.layers.Dense(1)])
        m.compile(keras.optimizers.SGD(.1, clipnorm=clip), "mse"); h = m.fit(X.astype("float32"), y.astype("float32"), epochs=20, batch_size=32, verbose=0)
        print(f"Keras SGD clipnorm={clip}: final loss {h.history['loss'][-1]:.4f}")
