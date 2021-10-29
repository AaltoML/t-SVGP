"""Module containing the integration tests for the `CVI` class."""
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


@pytest.fixture(name="cvi_gpr_optim_setup")
def _cvi_gpr_optim_setup():
    """Creates a GPR model and a matched CVI model (via natural gradient descent - single step)"""

    time_points, observations, kernel, noise_variance = _setup()
    input_data = (tf.constant(time_points), tf.constant(observations))
    gpr = gpflow.models.GPR(
        data=input_data,
        kernel=kernel,
        mean_function=None,
        noise_variance=noise_variance,
    )

    likelihood = Gaussian(variance=noise_variance)
    cvi = t_VGP(
        data=(time_points, observations),
        kernel=kernel,
        likelihood=likelihood,
    )

    cvi.update_variational_parameters(beta=1.0)

    return cvi, gpr


@pytest.fixture(name="cvi_vgp_optim_setup")
def _cvi_vgp_optim_setup():
    """Creates a VGP model and a matched CVI model"""

    time_points, observations, kernel, noise_variance = _setup()
    input_data = (
        tf.constant(time_points),
        tf.constant((observations > 0.5).astype(float)),
    )

    likelihood = Bernoulli()
    cvi = t_VGP(
        data=input_data,
        kernel=kernel,
        likelihood=likelihood,
    )

    vgp = gpflow.models.VGP(
        data=input_data,
        kernel=kernel,
        mean_function=None,
        likelihood=likelihood,
    )

    natgrad_rep = 20
    # one step of natgrads for CVI
    [cvi.update_variational_parameters(beta=1.0) for _ in range(natgrad_rep)]
    # one step of natgrads for VGP
    natgrad_opt = NaturalGradient(gamma=1.0)
    variational_params = [(vgp.q_mu, vgp.q_sqrt)]
    training_loss = vgp.training_loss_closure()
    [
        natgrad_opt.minimize(training_loss, var_list=variational_params)
        for _ in range(natgrad_rep)
    ]

    return cvi, vgp


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


def test_cvi_elbo_optimal(cvi_gpr_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    cvi, gpr = cvi_gpr_optim_setup
    np.testing.assert_almost_equal(cvi.elbo(), gpr.log_marginal_likelihood(), decimal=4)


def test_cvi_unchanged_at_optimum(cvi_gpr_optim_setup):
    """Test that the update does not change sites at the optimum"""
    cvi, _ = cvi_gpr_optim_setup
    # ELBO at optimum
    optim_elbo = cvi.elbo()
    # site update step
    cvi.update_variational_parameters(beta=1.0)
    # ELBO after step
    new_elbo = cvi.elbo()

    np.testing.assert_almost_equal(optim_elbo, new_elbo, decimal=4)


def test_optimal_sites(cvi_gpr_optim_setup):
    """Test that the optimal value of the exact sites match the true sites"""
    cvi, gpr = cvi_gpr_optim_setup

    cvi_nat1 = cvi.lambda_1.numpy()
    cvi_nat2 = cvi.lambda_2.numpy()

    # manually compute the optimal sites
    s2 = gpr.likelihood.variance.numpy()
    _, Y = gpr.data
    gpr_nat1 = Y / s2
    gpr_nat2 = 1.0 / s2 * np.ones_like(cvi_nat2)

    np.testing.assert_allclose(cvi_nat1, gpr_nat1)
    np.testing.assert_allclose(cvi_nat2, gpr_nat2)


def test_gradient_wrt_hyperparameters(cvi_vgp_optim_setup):
    """Test that for matched posteriors, gradients wrt hyperparameters match for cvi and vgp"""

    # cvi and vgp models after exact E-step
    cvi, vgp = cvi_vgp_optim_setup

    # gradient of VGP elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(vgp.kernel.trainable_variables)
        objective = vgp.training_loss()
    grads_vgp = tape.gradient(objective, vgp.kernel.trainable_variables)

    # gradient of CVI elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(cvi.kernel.trainable_variables)
        objective = -cvi.elbo()
    grads_cvi = tape.gradient(objective, cvi.kernel.trainable_variables)

    # compare gradients
    np.testing.assert_array_almost_equal(grads_vgp, grads_cvi, decimal=4)
