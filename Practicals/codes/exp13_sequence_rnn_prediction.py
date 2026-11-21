"""Experiment 13 (CO3): Sequential Dataset Preparation and RNN Prediction
Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib
Keras SimpleRNN if TensorFlow is installed, otherwise the NumPy BPTT RNN from common.py."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import out_dir, savefig, have_tf, rnn_init, rnn_train, rnn_predict

OUT = out_dir(__file__)
rng = np.random.default_rng(0)

# ---- 1. Build a time-series data frame (trend + seasonality + noise) -----------
t = np.arange(600)
df = pd.DataFrame({"t": t, "value": 0.002 * t + np.sin(2 * np.pi * t / 50) + .5 * np.sin(2 * np.pi * t / 17) + rng.normal(0, .08, t.size)})
print(df.head(), "\n", df.describe().round(3))

# ---- 2. Scale, window and split (no shuffling, no leakage) --------------------
split = int(.8 * len(df)); mu, sd = df.value[:split].mean(), df.value[:split].std()
s = ((df.value - mu) / sd).values
LOOK = 20
Xw = np.array([s[i:i + LOOK] for i in range(len(s) - LOOK)])[..., None]; yw = np.array([s[i + LOOK] for i in range(len(s) - LOOK)])
Xtr, ytr, Xte, yte = Xw[:split - LOOK], yw[:split - LOOK], Xw[split - LOOK:], yw[split - LOOK:]
print(f"\nWindowed data: X{Xw.shape} y{yw.shape} | train {Xtr.shape[0]} | test {Xte.shape[0]} | look-back = {LOOK}")

# ---- 3. Model --------------------------------------------------------------------
if have_tf():
    from tensorflow import keras
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.layers.Input((LOOK, 1)), keras.layers.SimpleRNN(32), keras.layers.Dense(1)]); m.compile("adam", "mse"); m.summary()
    h = m.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=40, batch_size=32, verbose=0).history
    ptr, pte = m.predict(Xtr).ravel(), m.predict(Xte).ravel(); hist = {"loss": h["loss"], "val_loss": h["val_loss"]}; backend = "Keras SimpleRNN"
else:
    P = rnn_init(1, 24, rho=.9, seed=0); hist = rnn_train(P, Xtr, ytr, epochs=60, lr=5e-3, Xv=Xte, yv=yte)
    ptr, pte = rnn_predict(P, Xtr), rnn_predict(P, Xte); backend = "NumPy RNN (BPTT)"
    for e in range(0, 60, 10): print(f"epoch {e+1:2d}: train loss {hist['loss'][e]:.4f}  val loss {hist['val_loss'][e]:.4f}")

rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2)) * sd
print(f"\nBackend: {backend}\nTrain RMSE (original units): {rmse(ptr, ytr):.4f} | Test RMSE: {rmse(pte, yte):.4f}")
base = rmse(Xte[:, -1, 0], yte); print(f"Naive baseline (repeat last value) test RMSE: {base:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(19, 5))
ax[0].plot(df.t, df.value, lw=.8); ax[0].axvline(split, c="r", ls="--", label="train | test"); ax[0].legend(); ax[0].set_title("Time series")
ax[1].plot(hist["loss"], label="train"); ax[1].plot(hist["val_loss"], label="val"); ax[1].set_title("Loss (MSE, scaled)"); ax[1].legend(); ax[1].grid()
ax[2].plot(yte * sd + mu, label="actual"); ax[2].plot(pte * sd + mu, label="predicted"); ax[2].legend(); ax[2].set_title(f"Test predictions ({backend})")
savefig(fig, OUT, "rnn_prediction.png")
df.to_csv(f"{OUT}/generated_series.csv", index=False)
