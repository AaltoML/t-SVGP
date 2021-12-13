# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# MNIST Classification
"""

# %%


import logging
import time

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.optimizers import NaturalGradient
from tqdm import tqdm

from src.models.tsvgp import t_SVGP

tf.get_logger().setLevel(logging.ERROR)

rng = np.random.RandomState(1)
tf.random.set_seed(1)

# %% [markdown]
"""
## Loading MNIST data
"""

# %%


def load_mnist():
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

    x, y = mnist_train
    x = tf.reshape(x, [x.shape[0], -1]).numpy()
    x = x.astype(np.float64) / 255
    y = np.reshape(y, (-1, 1))
    y = np.int64(y)

    xt, yt = mnist_test
    xt = tf.reshape(xt, [xt.shape[0], -1]).numpy()
    xt = xt.astype(np.float64) / 255
    yt = np.reshape(yt, (-1, 1))
    yt = np.int64(yt)

    perm = rng.permutation(x.shape[0])
    np.take(x, perm, axis=0, out=x)
    np.take(y, perm, axis=0, out=y)

    return x, y, xt, yt


M = 100  # Number of inducing points
C = 10  # Number of classes
mb_size = 200  # Size of minibatch during training
nit = 150  # Number of training iterations
nat_lr = 0.03  # Learning rate for E-step (variational params)
adam_lr = 0.02  # Learning rate for M-step (hyperparams)
n_e_steps = 1  # Number of E-steps per step
n_m_steps = 1  # Number of M-steps per step


# Initial hyperparameters
ell = 1.0
var = 1.0

# Load data
X, Y, XT, YT = load_mnist()

# Training data
train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(X.shape[0])

# Initialize inducing locations to the first M inputs in the data set
Z = X[:M, :].copy()

# %% [markdown]
"""
## Declaring Classification model
"""


# %%

models = []
names = []

# Set up the 'standard' q-SVGP model
m = gpflow.models.SVGP(
    kernel=gpflow.kernels.Matern52(lengthscales=np.ones((1, X.shape[1])) * ell, variance=var),
    likelihood=gpflow.likelihoods.Softmax(C),
    inducing_variable=Z.copy(),
    num_data=X.shape[0],
    whiten=True,
    num_latent_gps=C,
)

gpflow.set_trainable(m.q_mu, False)
gpflow.set_trainable(m.q_sqrt, False)

models.append(m)
names.append("q-SVGP")

# Set up the t-SVGP model
m = t_SVGP(
    kernel=gpflow.kernels.Matern52(lengthscales=np.ones((1, X.shape[1])) * ell, variance=var),
    likelihood=gpflow.likelihoods.Softmax(C),
    inducing_variable=Z.copy(),
    num_data=X.shape[0],
    num_latent_gps=C,
)

gpflow.set_trainable(m.lambda_1, False)
gpflow.set_trainable(m.lambda_2_sqrt, False)

models.append(m)
names.append("t-SVGP")

# %% [markdown]
"""
## Training model
"""
# %%


def train(model, iterations):
    """
    Utility function for training SVGP models with natural gradients

    :param model: GPflow model
    :param iterations: number of iterations
    """

    print("Optimizing model: ", model.name)

    natgrad_opt = NaturalGradient(gamma=nat_lr)

    tf.random.set_seed(4)
    train_iter = iter(train_dataset.batch(mb_size))

    tf.random.set_seed(4)
    train_iter2 = iter(train_dataset.batch(mb_size))

    training_loss = model.training_loss_closure(train_iter, compile=True)
    training_loss2 = model.training_loss_closure(train_iter2, compile=True)

    # Define the M-step (that is called in the same way for both)
    optimizer = tf.optimizers.Adam(adam_lr)

    @tf.function
    def optimization_m_step(training_loss, params):
        optimizer.minimize(training_loss, var_list=params)

    # Define the E-steps
    def optimization_step_nat(training_loss, variational_params):
        natgrad_opt.minimize(training_loss, var_list=variational_params)

    @tf.function
    def optimization_e_step(model, data):
        model.natgrad_step(data, lr=nat_lr)

    for _ in tqdm(range(iterations)):
        data = next(train_iter)

        if model.name == "svgp" and model.q_mu.trainable == False:
            variational_params = [(model.q_mu, model.q_sqrt)]
            optimization_e_step = tf.function(
                lambda loss: optimization_step_nat(loss, variational_params)
            )

            for i in range(n_e_steps):
                optimization_e_step(training_loss)
            for j in range(n_m_steps):
                optimization_m_step(training_loss2, model.trainable_variables)

        elif model.name == "t_svgp":
            for i in range(n_e_steps):
                optimization_e_step(model, data)
            for i in range(n_m_steps):
                optimization_m_step(training_loss2, model.trainable_variables)

        else:
            raise ("No training setup for this model.")


for m, name in zip(models, names):
    t0 = time.time()
    train(m, nit)
    t = time.time() - t0

    # Calculate NLPD on test set
    nlpd = -tf.reduce_mean(m.predict_log_density((XT, YT))).numpy()

    # Calculate accuracy on test set
    pred = m.predict_y(XT)[0]
    pred_argmax = tf.reshape(tf.argmax(pred, axis=1), (-1, 1))
    acc = np.mean(pred_argmax == YT)

    print("Training time for", name, "was", t, "seconds")
    print(name, "test NLPD =", nlpd)
    print(name, "test accuracy =", acc)
