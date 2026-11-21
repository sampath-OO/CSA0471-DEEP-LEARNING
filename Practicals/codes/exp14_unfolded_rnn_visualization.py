"""Experiment 14 (CO3): Unfolded RNN Architecture Visualization
Libraries: TensorFlow, Keras, NetworkX, Matplotlib  (Keras part optional: model summary + weight sharing check)"""
import numpy as np, networkx as nx
import matplotlib.pyplot as plt
from common import out_dir, savefig, have_tf, rnn_init

OUT = out_dir(__file__)

def draw_folded(ax):
    G = nx.DiGraph(); G.add_edges_from([("x", "h"), ("h", "h"), ("h", "y")])
    pos = {"x": (0, 0), "h": (0, 1), "y": (0, 2)}
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=["#9ecae1", "#fdae6b", "#a1d99b"], node_size=1800, edgecolors="k")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=13)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[("x", "h"), ("h", "y")], arrowsize=20, node_size=1800)
    ax.annotate("", xy=(.07, 1.12), xytext=(.07, .88), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-2.2", lw=1.5))
    ax.text(-.32, 1.0, "Wh", fontsize=12); ax.text(.05, .45, "Wx", fontsize=12); ax.text(.05, 1.5, "Wy", fontsize=12)
    ax.set_title("Folded RNN"); ax.set_xlim(-.6, .8); ax.set_ylim(-.4, 2.4); ax.axis("off")

def draw_unfolded(ax, T):
    G = nx.DiGraph(); pos = {}; colors = []
    for t in range(1, T + 1):
        for n, y_, c in [(f"x{t}", 0, "#9ecae1"), (f"h{t}", 1, "#fdae6b"), (f"y{t}", 2, "#a1d99b")]:
            G.add_node(n); pos[n] = (t, y_); colors.append(c)
        G.add_edge(f"x{t}", f"h{t}"); G.add_edge(f"h{t}", f"y{t}")
        if t > 1: G.add_edge(f"h{t-1}", f"h{t}")
    G.add_node("h0"); pos["h0"] = (0, 1); colors.append("#d9d9d9"); G.add_edge("h0", "h1")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=1300, edgecolors="k")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10); nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=18, node_size=1300)
    for t in range(2, T + 1): ax.text(t - .5, 1.08, "Wh", ha="center", color="C3", fontsize=10)
    for t in range(1, T + 1): ax.text(t + .08, .5, "Wx", color="C3", fontsize=9); ax.text(t + .08, 1.5, "Wy", color="C3", fontsize=9)
    ax.set_title(f"Unfolded through time (T={T}): the same Wx, Wh, Wy are reused at every step"); ax.set_ylim(-.4, 2.4); ax.axis("off")

fig = plt.figure(figsize=(18, 6)); gs = fig.add_gridspec(1, 3)
draw_folded(fig.add_subplot(gs[0])); draw_unfolded(fig.add_subplot(gs[1:]), 5)
savefig(fig, OUT, "rnn_folded_unfolded.png")

# Parameter sharing: parameter count does not depend on sequence length
n_in, n_h, n_out = 3, 8, 1
p = n_in * n_h + n_h * n_h + n_h + n_h * n_out + n_out
print(f"RNN parameters (in={n_in}, hidden={n_h}, out={n_out}): Wx {n_in*n_h} + Wh {n_h*n_h} + b {n_h} + Wy {n_h*n_out} + by {n_out} = {p}")
print("Parameter count is the same for any sequence length (weight sharing).")
print("Without sharing, an unfolded T=5 net would need", 5 * (n_in * n_h + n_h * n_h + n_h) + 5 * (n_h * n_out + n_out), "parameters.")

# Hidden-state trajectory of an unfolded RNN on a sample sequence (NumPy)
P = rnn_init(n_in, n_h, rho=.9, seed=0); x = np.random.default_rng(0).normal(size=(12, n_in)); h = np.zeros(n_h); H = []
for xt in x: h = np.tanh(xt @ P["Wx"] + h @ P["Wh"] + P["b"]); H.append(h)
H = np.array(H); fig, ax = plt.subplots(figsize=(9, 4.5)); im = ax.imshow(H.T, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xlabel("time step"); ax.set_ylabel("hidden unit"); ax.set_title("Hidden states h_t across the unrolled network"); plt.colorbar(im)
savefig(fig, OUT, "rnn_hidden_states.png")

if have_tf():
    from tensorflow import keras
    m = keras.Sequential([keras.layers.Input((5, n_in)), keras.layers.SimpleRNN(n_h, return_sequences=True), keras.layers.Dense(1)]); m.summary()
    print("Keras SimpleRNN parameters:", m.layers[0].count_params(), "(= Wx + Wh + b)")
