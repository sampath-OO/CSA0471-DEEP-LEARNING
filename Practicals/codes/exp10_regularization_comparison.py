"""Experiment 10 (CO2): L2, Dropout and Early Stopping Comparison
Libraries: TensorFlow, Keras, NumPy, Matplotlib
Keras if available, otherwise the NumPy MLP (identical set-up)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from common import out_dir, savefig, have_tf, NumpyMLP

OUT = out_dir(__file__)
X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=1, stratify=y)
Xtr, ytr = Xtr[:300], ytr[:300]                       # small training set -> easy to overfit
Xva, Xte, yva, yte = train_test_split(Xte, yte, test_size=.5, random_state=1, stratify=yte)
EPOCHS = 150
configs = {"Baseline": {}, "L2 (λ=1e-2)": {"l2": 1e-2}, "Dropout (0.5)": {"dropout": .5}, "Early stopping": {"patience": 10}}

def run_numpy(cfg):
    net = NumpyMLP([64, 256, 128, 10], l2=cfg.get("l2", 0.), dropout=cfg.get("dropout", 0.), lr=1e-3, seed=0)
    h = net.fit(Xtr, ytr, Xva, yva, epochs=EPOCHS, batch=32, patience=cfg.get("patience"))
    return h, net.evaluate(Xte, yte)[1]

def run_keras(cfg):
    from tensorflow import keras
    keras.utils.set_random_seed(0)
    reg = keras.regularizers.l2(cfg["l2"]) if "l2" in cfg else None
    layers = [keras.layers.Input((64,)), keras.layers.Dense(256, activation="relu", kernel_regularizer=reg)]
    if "dropout" in cfg: layers.append(keras.layers.Dropout(cfg["dropout"]))
    layers.append(keras.layers.Dense(128, activation="relu", kernel_regularizer=reg))
    if "dropout" in cfg: layers.append(keras.layers.Dropout(cfg["dropout"]))
    layers.append(keras.layers.Dense(10, activation="softmax"))
    m = keras.Sequential(layers); m.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [keras.callbacks.EarlyStopping(patience=cfg["patience"], restore_best_weights=True)] if "patience" in cfg else []
    hk = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=32, callbacks=cb, verbose=0).history
    h = {"loss": hk["loss"], "val_loss": hk["val_loss"], "acc": hk["accuracy"], "val_acc": hk["val_accuracy"], "stopped_epoch": len(hk["loss"])}
    return h, m.evaluate(Xte, yte, verbose=0)[1]

runner = run_keras if have_tf() else run_numpy
print("Backend:", "Keras" if have_tf() else "NumPy fallback", "| train samples:", len(Xtr))
res = {n: runner(c) for n, c in configs.items()}
print(f"\n{'Method':16s} {'train acc':>9s} {'val acc':>8s} {'test acc':>9s} {'val loss(min)':>14s} {'epochs run':>10s}")
for n, (h, te) in res.items():
    print(f"{n:16s} {h['acc'][-1]:9.4f} {h['val_acc'][-1]:8.4f} {te:9.4f} {min(h['val_loss']):14.4f} {len(h['loss']):10d}")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for n, (h, te) in res.items():
    ax[0].plot(h["val_loss"], label=n); ax[1].plot(h["loss"], label=n); ax[2].plot(h["val_acc"], label=n)
ax[0].set_title("Validation loss"); ax[1].set_title("Training loss"); ax[2].set_title("Validation accuracy")
for a in ax: a.legend(); a.grid(); a.set_xlabel("epoch")
ax[0].set_ylim(0, 2)
savefig(fig, OUT, "regularization_comparison.png")
fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(list(res), [r[1] for r in res.values()], color=["gray", "C0", "C1", "C2"])
ax.set_ylim(.8, 1); ax.set_ylabel("test accuracy"); ax.set_title("Test accuracy by regularisation"); plt.xticks(rotation=15)
savefig(fig, OUT, "regularization_test_accuracy.png")
