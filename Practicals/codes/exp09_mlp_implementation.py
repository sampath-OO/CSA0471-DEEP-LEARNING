"""Experiment 9 (CO2): Multilayer Perceptron Implementation
Libraries: TensorFlow, Keras, NumPy, Matplotlib
Uses Keras if TensorFlow is installed; otherwise falls back to the NumPy MLP in common.py (same architecture)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from common import out_dir, savefig, have_tf, NumpyMLP

OUT = out_dir(__file__)
X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=0, stratify=y)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=.2, random_state=0, stratify=ytr)
print("Data: 8x8 digits ->", Xtr.shape, Xva.shape, Xte.shape)

if have_tf():
    from tensorflow import keras
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.layers.Input((64,)), keras.layers.Dense(64, activation="relu"),
                              keras.layers.Dense(32, activation="relu"), keras.layers.Dense(10, activation="softmax")])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"]); model.summary()
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=32, verbose=2).history
    pred = model.predict(Xte).argmax(1); backend = "Keras"
else:
    net = NumpyMLP([64, 64, 32, 10], act="relu", lr=1e-3, seed=0)
    print("Architecture: 64 -> 64 (ReLU) -> 32 (ReLU) -> 10 (softmax); parameters =", sum(p.size for p in net.P))
    h = net.fit(Xtr, ytr, Xva, yva, epochs=40, batch=32)
    for e in range(0, 40, 5): print(f"epoch {e+1:2d}: loss {h['loss'][e]:.4f} acc {h['acc'][e]:.4f} | val_loss {h['val_loss'][e]:.4f} val_acc {h['val_acc'][e]:.4f}")
    pred = net.predict(Xte); backend = "NumPy fallback"
    h = {"loss": h["loss"], "val_loss": h["val_loss"], "accuracy": h["acc"], "val_accuracy": h["val_acc"]}

print(f"\nBackend: {backend}\nTest accuracy: {accuracy_score(yte, pred):.4f}")
cm = confusion_matrix(yte, pred); print("Confusion matrix:\n", cm)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid()
ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid()
im = ax[2].imshow(cm, cmap="Blues"); ax[2].set_title(f"Confusion matrix (test acc {accuracy_score(yte, pred):.3f})"); plt.colorbar(im, ax=ax[2])
for (i, j), v in np.ndenumerate(cm): ax[2].text(j, i, v, ha="center", va="center", fontsize=7)
savefig(fig, OUT, "mlp_training.png")
fig, ax = plt.subplots(2, 8, figsize=(14, 4))
for a, i in zip(ax.flat, range(16)):
    a.imshow(Xte[i].reshape(8, 8), cmap="gray_r"); a.set_title(f"T:{yte[i]} P:{pred[i]}", color="g" if yte[i] == pred[i] else "r", fontsize=9); a.axis("off")
savefig(fig, OUT, "mlp_predictions.png")
