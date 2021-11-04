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
# 1D Regression
"""

# %%
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.optimizers import NaturalGradient
from tqdm import tqdm

from src.models.tsvgp import t_SVGP
from src.models.tsvgp_white import t_SVGP_white

# For reproducibility
rng = np.random.RandomState(123)
tf.random.set_seed(42)

# %% [markdown]
"""
## Generating toy data for regression
"""

# %%
# Simulate data
func = lambda x: np.sin(15 * x)
N = 200  # Number of training observations
X = rng.rand(N, 1) * 2 - 1  # X values
F = func(X)
var_noise = 1.**2
Y = F + np.sqrt(var_noise) * rng.randn(N, 1)

# GP parameters
var_gp = 0.3
len_gp = 0.1

x_grid = np.linspace(-1, 1, 100)
plt.plot(x_grid, func(x_grid))
plt.plot(X, Y, "o")

# %% [markdown]
"""
## Declaring Regression models

The tested models are exact GPR and 4 versions parameterizations of SVGP
* q-SVGP : $q(u) = N(u; m, LL^T)$
* q-SVGP (white) : $u = Lv $, $q(v) = N(v; m, LL^T)$
* t-SVGP : $q(u) = p(u)t(u)$, with $\eta^{q}_{2} =  K^{-1} + \Lambda_{2}$ via natural parameters
* t-SVGP (white) : $q(u) = p(u)t(u)$, with $\eta^{q}_{2} =  K^{-1} + K^{-1}\Lambda_{2}K^{-1}$ via natural parameters
"""
# %%
# =============================================== Set up models
M = 20  # Number of inducing locations
Z = np.linspace(X.min(), X.max(), M).reshape(-1, 1)

m_gpr = gpflow.models.GPR(
    data=(X, Y),
    kernel=gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    noise_variance=var_noise
)

m_t = t_SVGP(
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Gaussian(variance=var_noise),
    Z,
    num_data=N,
)

m_t_white = t_SVGP_white(
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Gaussian(variance=var_noise),
    Z,
    num_data=N,
)

m_q_white = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Gaussian(variance=var_noise),
    Z,
    num_data=N,
    whiten=True,
)

m_q = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Gaussian(variance=var_noise),
    Z,
    num_data=N,
    whiten=False,
)



# %% [markdown]
"""
## Training model
"""
# %%
lr_natgrad = .9
nit = 5

data = (tf.convert_to_tensor(X), tf.convert_to_tensor(Y))

print("Elbos at initial parameter")

print("GRR llh:", m_gpr.log_marginal_likelihood().numpy())

[m_t_white.natgrad_step(X, Y, lr_natgrad) for _ in range(nit)]
print("t-SVGP_white elbo:", m_t_white.elbo(data).numpy())

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


# %%
n_grid = 100
x_grid = np.linspace(-1, 1, n_grid).reshape(-1, 1)
m, v = [a.numpy() for a in m_t.predict_y(x_grid)]

plt.plot(x_grid, m)
plt.fill_between(x_grid.reshape(-1,), 
                 y1=(m-2*np.sqrt(v)).reshape(-1,), 
                 y2=(m+2*np.sqrt(v)).reshape(-1,), 
                 alpha=.2)
plt.vlines(
    Z,
    ymin=F.min() - .1,
    ymax=F.max() + .1,
    linewidth=1.5,
    color='grey',
    alpha=.3
)
plt.plot(x_grid, func(x_grid))
plt.plot(X, Y, ".", alpha=.3)
plt.show()

# %% [markdown]
"""
## Computing elbos for new parameter grid
"""

# %%
N_grid = 100
llh_gpr = np.zeros((N_grid,))
llh_qsvgp = np.zeros((N_grid,))
llh_qsvgp_white = np.zeros((N_grid,))
llh_tsvgp = np.zeros((N_grid,))
llh_tsvgp_white = np.zeros((N_grid,))
vars_gp = np.linspace(0.05, 0.2, N_grid)

for i, v in enumerate(tqdm(vars_gp)):
    m_gpr.kernel.lengthscales.assign(tf.constant(v))
    llh_gpr[i] = m_gpr.log_marginal_likelihood().numpy()
    m_t.kernel.lengthscales.assign(tf.constant(v))
    llh_tsvgp[i] = m_t.elbo(data).numpy()
    m_t_white.kernel.lengthscales.assign(tf.constant(v))
    llh_tsvgp_white[i] = m_t_white.elbo(data).numpy()
    m_q.kernel.lengthscales.assign(tf.constant(v))
    llh_qsvgp[i] = m_q.elbo(data).numpy()
    m_q_white.kernel.lengthscales.assign(tf.constant(v))
    llh_qsvgp_white[i] = m_q_white.elbo(data).numpy()

print("done.")
# %%


plt.figure()
plt.plot(vars_gp, llh_gpr, label="GPR", linewidth=4)
plt.plot(vars_gp, llh_tsvgp, label="t-SVGP", linewidth=2)
plt.plot(vars_gp, llh_tsvgp_white, label="t-SVGP (white)", linewidth=2)
plt.plot(vars_gp, llh_qsvgp, label="q-SVGP", linewidth=2)
plt.plot(vars_gp, llh_qsvgp_white, label="q-SVGP (white)", linewidth=2)
plt.vlines(
    len_gp,
    ymin=llh_tsvgp.min() - 10,
    ymax=llh_tsvgp.max() + 10,
    color=[0, 0, 0, 1.0],
    linewidth=1,
    linestyle="dashed",
)
plt.xlim([0.05, 0.2])
plt.ylim(
    [
        llh_gpr.min() - 0.2 * (llh_gpr.max() - llh_gpr.min()),
        llh_gpr.max() + 0.2 * (llh_gpr.max() - llh_gpr.min()),
    ]
)
plt.legend()
plt.xlabel("$\\theta$")
plt.ylabel("ELBO")
plt.title("ELBO for M-step")
plt.show()
