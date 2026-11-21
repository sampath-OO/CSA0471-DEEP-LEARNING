"""Experiment 15 (CO3): BPTT Flow and Gradient Analysis
Libraries: TensorFlow, Keras, NumPy, Matplotlib
Manual NumPy BPTT; if TensorFlow is installed the gradient is also cross-checked with tf.GradientTape."""
import numpy as np
import matplotlib.pyplot as plt
from common import out_dir, savefig, have_tf, rnn_init, rnn_loss_grads

OUT = out_dir(__file__)
rng = np.random.default_rng(0)
T, N, nh = 40, 64, 16
X = rng.normal(size=(N, T, 1)); y = rng.normal(size=N)

# ---- 1. Numerical gradient check of BPTT ----------------------------------------
P = rnn_init(1, 6, rho=.9, seed=1); Xs, ys = X[:8, :10], y[:8]
loss, g, _ = rnn_loss_grads(P, Xs, ys); eps = 1e-6
for k in ["Wx", "Wh", "b", "Wy"]:
    num = np.zeros_like(P[k])
    for i in np.ndindex(P[k].shape):
        o = P[k][i]; P[k][i] = o + eps; lp = rnn_loss_grads(P, Xs, ys)[0]; P[k][i] = o - eps; lm = rnn_loss_grads(P, Xs, ys)[0]; P[k][i] = o; num[i] = (lp - lm) / (2 * eps)
    print(f"BPTT gradient check {k:2s}: max abs difference {np.max(np.abs(num - g[k])):.2e}")

if have_tf():
    import tensorflow as tf
    Pt = {k: tf.Variable(v) for k, v in P.items()}; xt = tf.constant(Xs); yt = tf.constant(ys)
    with tf.GradientTape() as tape:
        h = tf.zeros((8, 6), tf.float64)
        for t in range(10): h = tf.tanh(xt[:, t] @ Pt["Wx"] + h @ Pt["Wh"] + Pt["b"])
        l = 0.5 * tf.reduce_mean(((h @ Pt["Wy"] + Pt["by"])[:, 0] - yt) ** 2)
    gt = tape.gradient(l, Pt); print("TensorFlow GradientTape check on Wh:", np.max(np.abs(gt["Wh"].numpy() - g["Wh"])))

# ---- 2. Gradient flow back through time for different recurrent weight scales ---
rhos = [0.5, 0.9, 1.0, 1.2, 1.6]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5)); print("\n||dL/dh_t|| at t=1 vs t=T for several spectral radii of Wh")
for rho in rhos:
    P = rnn_init(1, nh, rho=rho, seed=3); _, _, norms = rnn_loss_grads(P, X, y)
    ax[0].semilogy(range(1, T + 1), norms, label=f"ρ(Wh)={rho}")
    print(f"  rho={rho:3.1f}: ||dL/dh_T|| = {norms[-1]:.3e}   ||dL/dh_1|| = {norms[0]:.3e}   ratio h_1/h_T = {norms[0]/norms[-1]:.3e}")
ax[0].set_xlabel("time step t"); ax[0].set_ylabel("||dL/dh_t||"); ax[0].legend(); ax[0].grid(); ax[0].set_title("Gradient flow in BPTT (loss at t=T)")

# ---- 3. Contribution of the T-step Jacobian product ----------------------------------
P = rnn_init(1, nh, rho=.9, seed=3); h = np.zeros(nh); J = np.eye(nh); prod_norm = []
for t in range(T):
    h = np.tanh(X[0, t] @ P["Wx"] + h @ P["Wh"] + P["b"]); J = J @ (P["Wh"].T @ np.diag(1 - h ** 2)); prod_norm.append(np.linalg.norm(J, 2))
ax[1].semilogy(range(1, T + 1), prod_norm); ax[1].set_xlabel("number of steps back-propagated k"); ax[1].set_ylabel("||∏ Jacobians||₂"); ax[1].grid()
ax[1].set_title("Norm of Jacobian product ∂h_T/∂h_(T-k) (ρ=0.9)")
print(f"\nNorm of Jacobian product after {T} steps: {prod_norm[-1]:.3e} -> vanishing gradients for long dependencies")
savefig(fig, OUT, "bptt_gradient_flow.png")

# ---- 4. Heat-map of gradients per unit over time ------------------------------------
P = rnn_init(1, nh, rho=.9, seed=3); h = np.zeros((1, nh)); H = [h]
for t in range(T): h = np.tanh(X[:1, t] @ P["Wx"] + h @ P["Wh"] + P["b"]); H.append(h)
dh = np.ones((1, nh)) * .1; G = []
for t in reversed(range(T)): G.append(np.abs(dh[0])); dh = (dh * (1 - H[t + 1] ** 2)) @ P["Wh"].T
G = np.array(G[::-1]); fig, ax = plt.subplots(figsize=(10, 4.5)); im = ax.imshow(np.log10(G.T + 1e-30), aspect="auto", cmap="magma")
ax.set_xlabel("time step"); ax.set_ylabel("hidden unit"); ax.set_title("log10 |dL/dh_t| per hidden unit (BPTT)"); plt.colorbar(im)
savefig(fig, OUT, "bptt_gradient_heatmap.png")
