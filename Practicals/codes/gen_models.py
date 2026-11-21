"""Keras VAE and GAN builders shared by Experiments 29 and 30 (requires TensorFlow)."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LATENT = 16

def load_mnist(n=20000):
    (x, y), (xt, yt) = keras.datasets.mnist.load_data()
    return x[:n].reshape(-1, 784).astype("float32") / 255, y[:n], xt.reshape(-1, 784).astype("float32") / 255, yt

class VAE(keras.Model):
    def __init__(self, latent=LATENT, beta=1.0):
        super().__init__(); self.latent, self.beta = latent, beta
        self.enc = keras.Sequential([layers.Input((784,)), layers.Dense(256, activation="relu"), layers.Dense(2 * latent)])
        self.dec = keras.Sequential([layers.Input((latent,)), layers.Dense(256, activation="relu"), layers.Dense(784, activation="sigmoid")])
        self.tracker = keras.metrics.Mean(name="loss")
    @property
    def metrics(self): return [self.tracker]
    def encode(self, x): mu, logv = tf.split(self.enc(x), 2, axis=1); return mu, logv
    def train_step(self, x):
        with tf.GradientTape() as tape:
            mu, logv = self.encode(x); z = mu + tf.exp(.5 * logv) * tf.random.normal(tf.shape(mu)); rec = self.dec(z)
            rl = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(x[..., None], rec[..., None]), axis=1))
            kl = tf.reduce_mean(-.5 * tf.reduce_sum(1 + logv - mu ** 2 - tf.exp(logv), axis=1)); loss = rl + self.beta * kl
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        self.tracker.update_state(loss); return {"loss": self.tracker.result()}
    def sample(self, n): return self.dec(tf.random.normal((n, self.latent))).numpy()

def build_gan(latent=LATENT):
    G = keras.Sequential([layers.Input((latent,)), layers.Dense(256), layers.LeakyReLU(.2), layers.Dense(512), layers.LeakyReLU(.2), layers.Dense(784, activation="sigmoid")])
    D = keras.Sequential([layers.Input((784,)), layers.Dense(512), layers.LeakyReLU(.2), layers.Dropout(.3), layers.Dense(256), layers.LeakyReLU(.2), layers.Dense(1)])
    return G, D

def train_gan(G, D, X, epochs=30, batch=128, latent=LATENT, lr=2e-4):
    bce = keras.losses.BinaryCrossentropy(from_logits=True); go = keras.optimizers.Adam(lr, .5); do = keras.optimizers.Adam(lr, .5)
    @tf.function
    def step(real):
        z = tf.random.normal((tf.shape(real)[0], latent))
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = G(z, training=True); ro = D(real, training=True); fo = D(fake, training=True)
            dl = bce(tf.ones_like(ro) * .9, ro) + bce(tf.zeros_like(fo), fo); gl = bce(tf.ones_like(fo), fo)
        do.apply_gradients(zip(dt.gradient(dl, D.trainable_variables), D.trainable_variables))
        go.apply_gradients(zip(gt.gradient(gl, G.trainable_variables), G.trainable_variables)); return dl, gl
    ds = tf.data.Dataset.from_tensor_slices(X).shuffle(10000).batch(batch, drop_remainder=True); hist = []
    for ep in range(epochs):
        d_, g_ = [], []
        for b in ds: dl, gl = step(b); d_.append(float(dl)); g_.append(float(gl))
        hist.append((np.mean(d_), np.mean(g_))); print(f"GAN epoch {ep+1}/{epochs}: D loss {hist[-1][0]:.3f}  G loss {hist[-1][1]:.3f}")
    return np.array(hist)
