# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import set_trainable
from gpflow.optimizers import NaturalGradient
from tqdm import tqdm

from src.tsvgp import t_SVGP

# +
X = np.loadtxt("data/banana_X_train", delimiter=",")
Y = np.loadtxt("data/banana_Y_train", delimiter=",").reshape(-1, 1)
mask = Y[:, 0] == 1
N = len(Y)
print(N)

plt.figure(figsize=(6, 6))
plt.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.5)
_ = plt.plot(X[np.logical_not(mask), 0], X[np.logical_not(mask), 1], "oC1", mew=0, alpha=0.5)
plt.show()


# +
M = 100  # Number of inducing locations
Z = X[:M, :]

# GP parameters
var_gp = 0.6
len_gp = 0.5

m_t = t_SVGP(
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli(),
    Z,
    num_data=N,
)

m_q_white = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli(),
    Z,
    num_data=N,
    whiten=True,
)

m_q = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli(),
    Z,
    num_data=N,
    whiten=False,
)

set_trainable(m_q_white.kernel.lengthscales, False)
set_trainable(m_q.kernel.lengthscales, False)
set_trainable(m_t.kernel.lengthscales, False)


# +
lr_natgrad = 0.8
nit = 10

data = (tf.convert_to_tensor(X), tf.convert_to_tensor(Y))

print("Elbos at initial parameter")

[m_t.natgrad_step(X, Y, lr_natgrad) for _ in range(nit)]
print("t-SVGP elbo:", m_t.elbo(data).numpy())

natgrad_opt = NaturalGradient(gamma=lr_natgrad)
training_loss = m_q.training_loss_closure(data)
training_loss_white = m_q_white.training_loss_closure(data)

# q-SVGP
variational_params = [(m_q.q_mu, m_q.q_sqrt)]
[natgrad_opt.minimize(training_loss, var_list=variational_params) for _ in range(nit)]
print("q-SVGP elbo:", -training_loss().numpy())

variational_params_white = [(m_q_white.q_mu, m_q_white.q_sqrt)]
[natgrad_opt.minimize(training_loss_white, var_list=variational_params_white) for _ in range(nit)]
print("q-SVGP (white) elbo:", -training_loss_white().numpy())

elbo = m_t.elbo(data).numpy()

# -

n_grid = 40
x_grid = np.linspace(-3, 3, n_grid)
xx, yy = np.meshgrid(x_grid, x_grid)
Xplot = np.vstack((xx.flatten(), yy.flatten())).T
p, _ = m_t.predict_y(Xplot)  # here we only care about the mean
plt.imshow(p.numpy().reshape(n_grid, n_grid), alpha=0.3, extent=[-3, 3, -3, 3], origin="lower")
# plt.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.5)
# _ = plt.plot(
#    X[np.logical_not(mask), 0], X[np.logical_not(mask), 1], "oC1", mew=0, alpha=0.5
# )
plt.show()
plt.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.5)
_ = plt.plot(X[np.logical_not(mask), 0], X[np.logical_not(mask), 1], "oC1", mew=0, alpha=0.5)
plt.show()

# +

print("Computing elbos for new parameter grid")


# ======================================== ELBO for different theta
N_grid = 100
llh_svgp = np.zeros((N_grid,))
llh_svgp_white = np.zeros((N_grid,))
llh_scvi = np.zeros((N_grid,))
vars_gp = np.linspace(0.05, 1.0, N_grid)

for i, v in enumerate(tqdm(vars_gp)):
    m_t.kernel.variance.assign(tf.constant(v))
    llh_scvi[i] = m_t.elbo(data).numpy()
    m_q.kernel.variance.assign(tf.constant(v))
    llh_svgp[i] = m_q.elbo(data).numpy()
    m_q_white.kernel.variance.assign(tf.constant(v))
    llh_svgp_white[i] = m_q_white.elbo(data).numpy()

print("done.")


# -

plt.figure()
plt.plot(vars_gp, llh_scvi, label="t-SVGP", linewidth=4)
plt.plot(vars_gp, llh_svgp, label="q-SVGP", linewidth=4)
plt.plot(vars_gp, llh_svgp_white, label="q-SVGP (whitened)", linewidth=4)
plt.vlines(
    var_gp,
    ymin=llh_scvi.min() - 10,
    ymax=llh_scvi.max() + 10,
    color=[0, 0, 0, 1.0],
    linewidth=1.5,
    linestyle="dashed",
)
plt.xlim([0.0, 1.0])
plt.ylim(
    [
        llh_scvi.min() - 0.0 * (llh_scvi.max() - llh_scvi.min()),
        llh_scvi.max() + 0.4 * (llh_scvi.max() - llh_scvi.min()),
    ]
)
plt.legend()
plt.xlabel("$\\theta$")
plt.ylabel("ELBO")
plt.show()
