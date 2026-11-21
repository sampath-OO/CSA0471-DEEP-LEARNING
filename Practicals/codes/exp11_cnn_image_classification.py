"""Experiment 11 (CO2): CNN for Image Classification
Libraries: TensorFlow, Keras, NumPy, Matplotlib
Requires TensorFlow (downloads MNIST through keras.datasets on first run)."""
import numpy as np
import matplotlib.pyplot as plt
from common import out_dir, savefig, need_tf

need_tf("Experiment 11")
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

OUT = out_dir(__file__)
keras.utils.set_random_seed(0)
(xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
xtr = xtr[..., None].astype("float32") / 255; xte = xte[..., None].astype("float32") / 255
print("Train:", xtr.shape, "Test:", xte.shape)

model = keras.Sequential([
    keras.layers.Input((28, 28, 1)),
    keras.layers.Conv2D(32, 3, activation="relu"), keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"), keras.layers.MaxPooling2D(),
    keras.layers.Flatten(), keras.layers.Dropout(.3), keras.layers.Dense(10, activation="softmax")])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"]); model.summary()
h = model.fit(xtr, ytr, validation_split=.1, epochs=8, batch_size=128, verbose=2).history
loss, acc = model.evaluate(xte, yte, verbose=0); print(f"\nTest loss {loss:.4f}  Test accuracy {acc:.4f}")
pred = model.predict(xte).argmax(1); print(classification_report(yte, pred, digits=4))

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
cm = confusion_matrix(yte, pred); ax[2].imshow(cm, cmap="Blues"); ax[2].set_title("Confusion matrix")
savefig(fig, OUT, "cnn_training.png")

fig, ax = plt.subplots(2, 8, figsize=(14, 4))
for a, i in zip(ax.flat, range(16)):
    a.imshow(xte[i, :, :, 0], cmap="gray"); a.set_title(f"T:{yte[i]} P:{pred[i]}", color="g" if yte[i] == pred[i] else "r"); a.axis("off")
savefig(fig, OUT, "cnn_predictions.png")

# Learned first-layer filters and feature maps
W = model.layers[0].get_weights()[0]
fig, ax = plt.subplots(4, 8, figsize=(12, 6))
for a, k in zip(ax.flat, range(32)): a.imshow(W[:, :, 0, k], cmap="gray"); a.axis("off")
fig.suptitle("Learned 3x3 filters (conv1)"); savefig(fig, OUT, "cnn_filters.png")
fm = keras.Model(model.inputs, model.layers[0].output).predict(xte[:1])
fig, ax = plt.subplots(2, 8, figsize=(14, 4))
for a, k in zip(ax.flat, range(16)): a.imshow(fm[0, :, :, k], cmap="viridis"); a.axis("off")
fig.suptitle("Feature maps of conv1 for the first test image"); savefig(fig, OUT, "cnn_feature_maps.png")
