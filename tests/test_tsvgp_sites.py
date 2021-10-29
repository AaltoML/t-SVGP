"""Module containing the integration tests for the `CVI` class."""
import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow.likelihoods import Bernoulli, Gaussian
from gpflow.optimizers import NaturalGradient

from src.tsvgp_sites import t_SVGP_sites

LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 8
NOISE_VARIANCE = 0.3

rng = np.random.RandomState(123)
tf.random.set_seed(42)


@pytest.fixture(name="tsvgp_gpr_optim_setup")
def _tsvgp_gpr_optim_setup():
    """Creates a GPR model and a matched Sparse CVI model (via natural gradient descent - single step)"""

    time_points, observations, kernel, noise_variance = _setup()
    input_data = (tf.constant(time_points), tf.constant(observations))
    gpr = gpflow.models.GPR(
        data=input_data,
        kernel=kernel,
        mean_function=None,
        noise_variance=noise_variance,
    )

    likelihood = Gaussian(variance=noise_variance)
    tsvgp = t_SVGP_sites(
        data=input_data,
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
    )
    for _ in range(10):
        print(tsvgp.lambda_2)
        tsvgp.natgrad_step(lr=0.9)

    return tsvgp, gpr


@pytest.fixture(name="tsvgp_svgp_optim_setup")
def _tsvgp_svgp_optim_setup():
    """Creates a SVGP model and a matched Sparse CVI model (E-step)"""

    time_points, observations, kernel, noise_variance = _setup()
    input_data = (
        tf.constant(time_points),
        tf.constant((observations > 0.0).astype(float)),
    )

    likelihood = Bernoulli()

    svgp = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
    )

    tsvgp = t_SVGP_sites(
        data=input_data,
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
    )

    natgrad_rep = 10
    lr = 0.9
    # one step of natgrads for tsvgp
    for _ in range(natgrad_rep):
        print(_)
        tsvgp.natgrad_step(lr=lr)
    # one step of natgrads for SVGP
    natgrad_opt = NaturalGradient(gamma=lr)
    variational_params = [(svgp.q_mu, svgp.q_sqrt)]
    training_loss = svgp.training_loss_closure(input_data)
    [
        natgrad_opt.minimize(training_loss, var_list=variational_params)
        for _ in range(natgrad_rep)
    ]

    return tsvgp, svgp, input_data


def _setup():
    """Data, kernel and likelihood setup"""

    def func(x):
        return (
            np.sin(x * 3 * 3.14)
            + 0.3 * np.cos(x * 9 * 3.14)
            + 0.5 * np.sin(x * 7 * 3.14)
        )

    input_points = rng.rand(NUM_DATA, 1) * 2 - 1  # X values
    observations = func(input_points) + 0.2 * rng.randn(NUM_DATA, 1)

    kernel = gpflow.kernels.SquaredExponential(
        lengthscales=LENGTH_SCALE, variance=VARIANCE
    )
    variance = tf.constant(NOISE_VARIANCE, dtype=tf.float64)

    return input_points, observations, kernel, variance


def test_tsvgp_elbo_optimal(tsvgp_gpr_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    tsvgp, gpr = tsvgp_gpr_optim_setup
    np.testing.assert_almost_equal(
        tsvgp.elbo(), gpr.log_marginal_likelihood(), decimal=4
    )


def test_tsvgp_unchanged_at_optimum(tsvgp_gpr_optim_setup):
    """Test that the update does not change sites at the optimum"""
    tsvgp, gpr = tsvgp_gpr_optim_setup
    # ELBO at optimum
    data = gpr.data
    optim_elbo = tsvgp.elbo()
    # site update step
    tsvgp.natgrad_step(lr=0.5)
    # ELBO after step
    new_elbo = tsvgp.elbo()

    np.testing.assert_almost_equal(optim_elbo, new_elbo, decimal=4)


def test_gradient_wrt_hyperparameters(tsvgp_svgp_optim_setup):
    """Test that for matched posteriors (through natgrads, gradients wrt hyperparameters match for tsvgp and svgp"""

    # cvi and vgp models after exact E-step
    tsvgp, svgp, data = tsvgp_svgp_optim_setup

    # gradient of SVGP elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(svgp.kernel.trainable_variables)
        objective = svgp.training_loss(data)
    grads_svgp = tape.gradient(objective, svgp.kernel.trainable_variables)

    # gradient of tsvgp elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(tsvgp.kernel.trainable_variables)
        objective = -tsvgp.elbo()
    grads_tsvgp = tape.gradient(objective, tsvgp.kernel.trainable_variables)

    # compare gradients
    np.testing.assert_array_almost_equal(grads_svgp, grads_tsvgp, decimal=4)


def test_optimal_sites(tsvgp_gpr_optim_setup):
    """Test that the optimal value of the exact sites match the true sites"""
    tsvgp, gpr = tsvgp_gpr_optim_setup

    tsvgp_nat1 = tsvgp.sites.lambda_1.numpy()
    tsvgp_nat2 = tsvgp.sites.lambda_2.numpy()

    # manually compute the optimal sites
    s2 = gpr.likelihood.variance.numpy()
    _, Y = gpr.data
    gpr_nat1 = Y / s2
    # ! note the alternative parameterization from CVI
    gpr_nat2 = 1.0 / s2 * np.ones_like(tsvgp_nat2)

    np.testing.assert_allclose(tsvgp_nat1, gpr_nat1)
    np.testing.assert_allclose(tsvgp_nat2, gpr_nat2)
