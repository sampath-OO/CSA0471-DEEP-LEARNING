"""Experiment 16 (CO3): LSTM and RNN Performance Comparison
Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib
Requires TensorFlow. Task: predict the next value of a noisy multi-frequency series, then a long-memory task."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import out_dir, savefig, need_tf

need_tf("Experiment 16")
from tensorflow import keras

OUT = out_dir(__file__)
rng = np.random.default_rng(0)

def windows(s, look):
    X = np.array([s[i:i + look] for i in range(len(s) - look)])[..., None]; y = s[look:]; return X, y

def build(kind, look, units=32):
    layer = {"SimpleRNN": keras.layers.SimpleRNN, "LSTM": keras.layers.LSTM}[kind]
    m = keras.Sequential([keras.layers.Input((look, 1)), layer(units), keras.layers.Dense(1)])
    m.compile(keras.optimizers.Adam(1e-3), "mse"); return m

# ---- Task A: time-series forecasting --------------------------------------------------------
t = np.arange(1500); s = np.sin(2 * np.pi * t / 40) + .5 * np.sin(2 * np.pi * t / 13) + rng.normal(0, .1, t.size)
s = (s - s.mean()) / s.std(); split = int(.8 * len(s)); LOOK = 30
X, y = windows(s, LOOK); Xtr, ytr, Xte, yte = X[:split - LOOK], y[:split - LOOK], X[split - LOOK:], y[split - LOOK:]
rows, hist, preds = [], {}, {}
for kind in ["SimpleRNN", "LSTM"]:
    keras.utils.set_random_seed(0); m = build(kind, LOOK)
    h = m.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=30, batch_size=32, verbose=0).history
    p = m.predict(Xte, verbose=0).ravel(); hist[kind] = h; preds[kind] = p
    rows.append(dict(task="forecast", model=kind, params=m.count_params(), test_mse=float(np.mean((p - yte) ** 2))))

# ---- Task B: long-memory task (output = first element of a length-T sequence) ----------------
for T in [10, 30, 60]:
    Xb = rng.normal(size=(3000, T, 1)).astype("float32"); yb = Xb[:, 0, 0]
    for kind in ["SimpleRNN", "LSTM"]:
        keras.utils.set_random_seed(0); m = build(kind, T, 32)
        m.fit(Xb[:2400], yb[:2400], epochs=15, batch_size=64, verbose=0)
        rows.append(dict(task=f"memory T={T}", model=kind, params=m.count_params(), test_mse=float(m.evaluate(Xb[2400:], yb[2400:], verbose=0))))

df = pd.DataFrame(rows); print(df.to_string(index=False)); df.to_csv(f"{OUT}/results.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(19, 5))
for k, h in hist.items(): ax[0].plot(h["val_loss"], label=k)
ax[0].set_title("Validation MSE (forecast task)"); ax[0].legend(); ax[0].grid()
ax[1].plot(yte[:200], "k", label="actual")
for k, p in preds.items(): ax[1].plot(p[:200], label=k)
ax[1].legend(); ax[1].set_title("Test predictions")
piv = df[df.task.str.startswith("memory")].pivot(index="task", columns="model", values="test_mse"); piv.plot.bar(ax=ax[2], rot=0); ax[2].set_title("Long-memory task MSE (lower is better)")
savefig(fig, OUT, "lstm_vs_rnn.png")
