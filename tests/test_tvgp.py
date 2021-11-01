"""Module containing the integration tests for the `tVGP` class."""
import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow.likelihoods import Bernoulli, Gaussian
from gpflow.optimizers import NaturalGradient

from src.tvgp import t_VGP

LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 8
NOISE_VARIANCE = 0.3

rng = np.random.RandomState(123)
tf.random.set_seed(42)


@pytest.fixture(name="tvgp_gpr_optim_setup")
def _tvgp_gpr_optim_setup():
    """Creates a GPR model and a matched tVGP model (via natural gradient descent - single step)"""

    time_points, observations, kernel, noise_variance = _setup()
    input_data = (tf.constant(time_points), tf.constant(observations))
    gpr = gpflow.models.GPR(
        data=input_data,
        kernel=kernel,
        mean_function=None,
        noise_variance=noise_variance,
    )

    likelihood = Gaussian(variance=noise_variance)
    tvgp = t_VGP(
        data=(time_points, observations),
        kernel=kernel,
        likelihood=likelihood,
    )

    tvgp.update_variational_parameters(beta=1.0)

    return tvgp, gpr


@pytest.fixture(name="tvgp_qvgp_optim_setup")
def _tvgp_qvgp_optim_setup():
    """Creates a VGP model and a matched tVGP model"""

    time_points, observations, kernel, noise_variance = _setup()
    input_data = (
        tf.constant(time_points),
        tf.constant((observations > 0.5).astype(float)),
    )

    likelihood = Bernoulli()
    tvgp = t_VGP(
        data=input_data,
        kernel=kernel,
        likelihood=likelihood,
    )

    qvgp = gpflow.models.VGP(
        data=input_data,
        kernel=kernel,
        mean_function=None,
        likelihood=likelihood,
    )

    natgrad_rep = 20
    # one step of natgrads for tVGP
    [tvgp.update_variational_parameters(beta=1.0) for _ in range(natgrad_rep)]
    # one step of natgrads for VGP
    natgrad_opt = NaturalGradient(gamma=1.0)
    variational_params = [(qvgp.q_mu, qvgp.q_sqrt)]
    training_loss = qvgp.training_loss_closure()
    [
        natgrad_opt.minimize(training_loss, var_list=variational_params)
        for _ in range(natgrad_rep)
    ]

    return tvgp, qvgp


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


def test_tvgp_elbo_optimal(tvgp_gpr_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    tvgp, gpr = tvgp_gpr_optim_setup
    np.testing.assert_almost_equal(tvgp.elbo(), gpr.log_marginal_likelihood(), decimal=4)


def test_tvgp_unchanged_at_optimum(tvgp_gpr_optim_setup):
    """Test that the update does not change sites at the optimum"""
    tvgp, _ = tvgp_gpr_optim_setup
    # ELBO at optimum
    optim_elbo = tvgp.elbo()
    # site update step
    tvgp.update_variational_parameters(beta=1.0)
    # ELBO after step
    new_elbo = tvgp.elbo()

    np.testing.assert_almost_equal(optim_elbo, new_elbo, decimal=4)


def test_optimal_sites(tvgp_gpr_optim_setup):
    """Test that the optimal value of the exact sites match the true sites"""
    tvgp, gpr = tvgp_gpr_optim_setup

    tvgp_nat1 = tvgp.lambda_1.numpy()
    tvgp_nat2 = tvgp.lambda_2.numpy()

    # manually compute the optimal sites
    s2 = gpr.likelihood.variance.numpy()
    _, Y = gpr.data
    gpr_nat1 = Y / s2
    gpr_nat2 = 1.0 / s2 * np.ones_like(tvgp_nat2)

    np.testing.assert_allclose(tvgp_nat1, gpr_nat1)
    np.testing.assert_allclose(tvgp_nat2, gpr_nat2)


def test_gradient_wrt_hyperparameters(tvgp_qvgp_optim_setup):
    """Test that for matched posteriors, gradients wrt hyperparameters match for tvgp and qvgp"""

    # tvgp and qvgp models after exact E-step
    tvgp, qvgp = tvgp_qvgp_optim_setup

    # gradient of qVGP elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(qvgp.kernel.trainable_variables)
        objective = qvgp.training_loss()
    grads_qvgp = tape.gradient(objective, qvgp.kernel.trainable_variables)

    # gradient of tVGP elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(tvgp.kernel.trainable_variables)
        objective = -tvgp.elbo()
    grads_tvgp = tape.gradient(objective, tvgp.kernel.trainable_variables)

    # compare gradients
    np.testing.assert_array_almost_equal(grads_qvgp, grads_tvgp, decimal=4)
