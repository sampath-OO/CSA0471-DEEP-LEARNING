"""Experiment 18 (CO3): RNN, LSTM and GRU Hyperparameter Experimentation
Libraries: TensorFlow, Keras, NumPy, KerasTuner, Matplotlib
Requires TensorFlow and keras-tuner (pip install keras-tuner)."""
import numpy as np
import matplotlib.pyplot as plt
from common import out_dir, savefig, need_tf

need_tf("Experiment 18")
from tensorflow import keras
import keras_tuner as kt

OUT = out_dir(__file__)
rng = np.random.default_rng(0)
t = np.arange(1200); s = np.sin(2 * np.pi * t / 35) + .4 * np.sin(2 * np.pi * t / 9) + rng.normal(0, .1, t.size); s = (s - s.mean()) / s.std()
LOOK = 24; X = np.array([s[i:i + LOOK] for i in range(len(s) - LOOK)])[..., None]; y = s[LOOK:]
n = int(.7 * len(X)); m_ = int(.85 * len(X)); Xtr, ytr, Xva, yva, Xte, yte = X[:n], y[:n], X[n:m_], y[n:m_], X[m_:], y[m_:]

def build_model(hp):
    cell = {"SimpleRNN": keras.layers.SimpleRNN, "LSTM": keras.layers.LSTM, "GRU": keras.layers.GRU}[hp.Choice("cell", ["SimpleRNN", "LSTM", "GRU"])]
    m = keras.Sequential([keras.layers.Input((LOOK, 1)), cell(hp.Int("units", 16, 64, step=16)),
                          keras.layers.Dropout(hp.Float("dropout", 0., .4, step=.2)), keras.layers.Dense(1)])
    m.compile(keras.optimizers.Adam(hp.Choice("lr", [1e-2, 3e-3, 1e-3])), "mse"); return m

tuner = kt.RandomSearch(build_model, objective="val_loss", max_trials=12, executions_per_trial=1, directory=f"{OUT}/kt", project_name="rnn_lstm_gru", overwrite=True, seed=0)
tuner.search(Xtr, ytr, validation_data=(Xva, yva), epochs=15, batch_size=32, verbose=0)
tuner.results_summary(3)
rows = []
for tr in tuner.oracle.get_best_trials(12):
    rows.append((tr.hyperparameters.values["cell"], tr.hyperparameters.values["units"], tr.hyperparameters.values["dropout"], tr.hyperparameters.values["lr"], tr.score))
print("\ncell | units | dropout | lr | val_loss")
for r in rows: print(*r, sep=" | ")
best = tuner.get_best_hyperparameters(1)[0]; print("\nBest hyper-parameters:", best.values)
model = tuner.hypermodel.build(best); h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=30, batch_size=32, verbose=0).history
print("Test MSE of best model:", model.evaluate(Xte, yte, verbose=0))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for cell, c in zip(["SimpleRNN", "LSTM", "GRU"], ["C0", "C1", "C2"]):
    v = [r[4] for r in rows if r[0] == cell]; ax[0].scatter([cell] * len(v), v, c=c, s=60)
ax[0].set_ylabel("validation MSE"); ax[0].set_title("Tuning trials by cell type"); ax[0].grid()
ax[1].plot(h["loss"], label="train"); ax[1].plot(h["val_loss"], label="val"); ax[1].legend(); ax[1].set_title("Best model training")
savefig(fig, OUT, "rnn_lstm_gru_tuning.png")
