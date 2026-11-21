"""Experiment 12 (CO2): CNN Architecture and Hyperparameter Comparison
Libraries: TensorFlow, Keras, NumPy, Matplotlib
Requires TensorFlow. Uses a 10k-image MNIST subset so every configuration trains quickly."""
import numpy as np
import matplotlib.pyplot as plt
from common import out_dir, savefig, need_tf

need_tf("Experiment 12")
from tensorflow import keras

OUT = out_dir(__file__)
(xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
xtr = xtr[:10000, ..., None].astype("float32") / 255; ytr = ytr[:10000]; xte = xte[..., None].astype("float32") / 255

def build(filters=(16, 32), kernel=3, dense=64, dropout=0., lr=1e-3):
    layers = [keras.layers.Input((28, 28, 1))]
    for f in filters: layers += [keras.layers.Conv2D(f, kernel, activation="relu", padding="same"), keras.layers.MaxPooling2D()]
    layers += [keras.layers.Flatten(), keras.layers.Dense(dense, activation="relu"), keras.layers.Dropout(dropout), keras.layers.Dense(10, activation="softmax")]
    m = keras.Sequential(layers); m.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"]); return m

experiments = {
    "Baseline (16-32, k3)": {},
    "Shallow (16)": dict(filters=(16,)),
    "Deep (16-32-64)": dict(filters=(16, 32, 64)),
    "Wide (32-64)": dict(filters=(32, 64)),
    "Kernel 5": dict(kernel=5),
    "Dropout 0.4": dict(dropout=.4),
    "LR 1e-2": dict(lr=1e-2),
    "LR 1e-4": dict(lr=1e-4),
}
rows, hist = [], {}
for name, kw in experiments.items():
    keras.utils.set_random_seed(0); m = build(**kw)
    h = m.fit(xtr, ytr, validation_split=.2, epochs=8, batch_size=64, verbose=0).history
    loss, acc = m.evaluate(xte, yte, verbose=0); hist[name] = h
    rows.append((name, m.count_params(), h["val_accuracy"][-1], acc)); print(f"{name:22s} params={m.count_params():8d} val_acc={h['val_accuracy'][-1]:.4f} test_acc={acc:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(19, 5))
for n, h in hist.items(): ax[0].plot(h["val_accuracy"], label=n); ax[1].plot(h["val_loss"], label=n)
ax[0].set_title("Validation accuracy"); ax[1].set_title("Validation loss"); ax[0].legend(fontsize=7)
ax[2].barh([r[0] for r in rows], [r[3] for r in rows]); ax[2].set_xlim(.9, 1); ax[2].set_title("Test accuracy")
savefig(fig, OUT, "cnn_hyperparameter_comparison.png")
import csv
with open(f"{OUT}/results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["config", "params", "val_acc", "test_acc"]); w.writerows(rows)
