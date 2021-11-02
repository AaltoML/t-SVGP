import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.tsvgp import t_SVGP

rng = np.random.RandomState(123)
tf.random.set_seed(42)

# # Loading motorcycle accident data

csv_data = np.loadtxt("data/mcycle.csv", delimiter=",", skiprows=1)
X = csv_data[:, 0].reshape(-1, 1)
Y = csv_data[:, 1].reshape(-1, 1)
data = (X, Y)
N = len(X)
Y /= Y.std()
plt.figure()
plt.plot(X, Y, "*")
plt.show()

# # Declaring heteroskedastic regression model

# +
likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
    distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
    scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
)
kernel = gpf.kernels.SeparateIndependent(
    [
        gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
        gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
    ]
)
# Initial inducing points position Z
M = 50
Z = np.linspace(X.min(), X.max(), M)[:, None]  # Z must be of shape [M, 1]

inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
    gpf.inducing_variables.InducingPoints(Z),
)

m_tsvgp = t_SVGP(kernel, likelihood, inducing_variable, num_data=N, num_latent_gps=2)


# -

# # Plot pre-training (a priori) prediction

# +
def plot_distribution(X, Y, loc, scale, index=0):
    plt.figure(figsize=(15, 5))
    x = X.squeeze()
    for k in (1, 2):
        lb = (loc - k * scale).squeeze()
        ub = (loc + k * scale).squeeze()
        plt.fill_between(x, lb, ub, color="silver", alpha=1 - 0.05 * k ** 3)
    plt.plot(x, lb, color="silver")
    plt.plot(x, ub, color="silver")
    plt.plot(X, loc, color="black")
    plt.scatter(X, Y, color="gray", alpha=0.8)
    plt.savefig("het.png")
    plt.show()


Ymean, Yvar = m_tsvgp.predict_y(X)
Ymean = Ymean.numpy().squeeze()
Ystd = tf.sqrt(Yvar).numpy().squeeze()
plot_distribution(X, Y, Ymean, Ystd, -1)
# -

# # Training model


# +
lr_adam = 0.1
lr_natgrad = 0.5

nit_e = 2
nit_m = 1


def E_step():
    [m_tsvgp.natgrad_step(X, Y, lr_natgrad) for _ in range(nit_e)]


optimizer = tf.optimizers.Adam(lr_adam)


@tf.function
def M_step():
    [
        optimizer.minimize(m_tsvgp.training_loss_closure(data), m_tsvgp.kernel.trainable_variables)
        for _ in range(nit_m)
    ]


# -

# # Run Optimization

nrep = 100
for r in range(nrep):
    E_step()
    M_step()
    if r % 10 == 0:
        print(r, m_tsvgp.elbo(data))


Ymean, Yvar = m_tsvgp.predict_y(X)
Ymean = Ymean.numpy().squeeze()
Ystd = tf.sqrt(Yvar).numpy().squeeze()
plot_distribution(X, Y, Ymean, Ystd, r)
