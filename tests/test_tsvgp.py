"""Module containing the integration tests for the `CVI` class."""
import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow.likelihoods import Bernoulli, Gaussian
from gpflow.optimizers import NaturalGradient

from src.tsvgp import t_SVGP

LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 8
NOISE_VARIANCE = 0.3

rng = np.random.RandomState(123)
tf.random.set_seed(42)


@pytest.fixture(name="scvi_gpr_optim_setup")
def _scvi_gpr_optim_setup():
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
    scvi = t_SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
    )
    for i in range(10):
        scvi.natgrad_step(*input_data, lr=0.9)

    return scvi, gpr


@pytest.fixture(name="scvi_svgp_optim_setup")
def _scvi_svgp_optim_setup():
    """Creates a SVGP model and a matched Sparse CVI model (E-step)"""

    def __scvi_svgp_optim_setup(num_latent_gps=1):

        time_points, observations, kernel, noise_variance = _setup()

        # duplicating and weighting data to test multidimensional case
        observations = tf.tile((observations > 0.0).astype(float), [1, num_latent_gps])
        observations *= np.random.rand(1, num_latent_gps)
        input_data = (tf.constant(time_points), tf.constant(observations))

        likelihood = Bernoulli()

        svgp = gpflow.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
            num_latent_gps=num_latent_gps,
        )

        scvi = t_SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
            num_latent_gps=num_latent_gps,
        )

        natgrad_rep = 20
        # one step of natgrads for SCVI
        [scvi.natgrad_step(*input_data, lr=1.0) for _ in range(natgrad_rep)]
        # one step of natgrads for SVGP
        natgrad_opt = NaturalGradient(gamma=1.0)
        variational_params = [(svgp.q_mu, svgp.q_sqrt)]
        training_loss = svgp.training_loss_closure(input_data)
        [
            natgrad_opt.minimize(training_loss, var_list=variational_params)
            for _ in range(natgrad_rep)
        ]

        return scvi, svgp, input_data

    return __scvi_svgp_optim_setup


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


def test_scvi_elbo_optimal(scvi_gpr_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    scvi, gpr = scvi_gpr_optim_setup
    data = gpr.data
    np.testing.assert_almost_equal(
        scvi.elbo(data), gpr.log_marginal_likelihood(), decimal=4
    )


def test_predictions_match_scvi_gpr_optimal(scvi_gpr_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    scvi, gpr = scvi_gpr_optim_setup
    X = gpr.data[0] + 1.0
    mu_scvi, var_scvi = scvi.predict_f(X)
    mu_gpr, var_gpr = gpr.predict_f(X)
    np.testing.assert_array_almost_equal(mu_scvi, mu_gpr, decimal=4)
    np.testing.assert_array_almost_equal(var_scvi, var_gpr, decimal=4)


def test_predictions_match_scvi_svgp_optimal(scvi_svgp_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    scvi, svgp, data = scvi_svgp_optim_setup(1)
    X = data[0] + 0.1
    mu_scvi, var_scvi = scvi.predict_f(X)
    mu_svgp, var_svgp = svgp.predict_f(X)
    np.testing.assert_array_almost_equal(mu_scvi, mu_svgp, decimal=4)
    np.testing.assert_array_almost_equal(var_scvi, var_svgp, decimal=4)


def test_scvi_unchanged_at_optimum(scvi_gpr_optim_setup):
    """Test that the update does not change sites at the optimum"""
    scvi, gpr = scvi_gpr_optim_setup
    # ELBO at optimum
    data = gpr.data
    optim_elbo = scvi.elbo(data)
    # site update step
    scvi.natgrad_step(*data, lr=0.9)
    # ELBO after step
    new_elbo = scvi.elbo(data)

    np.testing.assert_almost_equal(optim_elbo, new_elbo, decimal=4)


def test_scvi_minibatch_same_elbo(scvi_gpr_optim_setup):
    """Test that the ELBO from minibactch is same as feeding full data for replicated data points"""

    scvi1, gpr = scvi_gpr_optim_setup
    scvi2, _ = scvi_gpr_optim_setup

    data = gpr.data
    scvi1.num_data = NUM_DATA
    x_d, y_d = data
    # Repeat the same data point
    x = x_d.numpy()[0].repeat(NUM_DATA)[:, None]
    y = y_d.numpy()[0].repeat(NUM_DATA)[:, None]
    data_repeat = (x, y)
    # Single data point elbo scaled for minibatch and same datapoint repeated
    optim_elbo2 = scvi2.elbo(data_repeat)
    optim_elbo1 = scvi1.elbo((x_d.numpy()[0][:, None], y_d.numpy()[0][:, None]))

    np.testing.assert_almost_equal(optim_elbo2, optim_elbo1, decimal=4)


@pytest.mark.parametrize("num_latent_gps", [2])
def test_gradient_wrt_hyperparameters(scvi_svgp_optim_setup, num_latent_gps):
    """Test that for matched posteriors (through natgrads, gradients wrt hyperparameters match for scvi and svgp"""

    # cvi and vgp models after exact E-step
    scvi, svgp, data = scvi_svgp_optim_setup(num_latent_gps)

    # gradient of SVGP elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(svgp.kernel.trainable_variables)
        objective = svgp.training_loss(data)
    grads_svgp = tape.gradient(objective, svgp.kernel.trainable_variables)

    # gradient of SCVI elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(scvi.kernel.trainable_variables)
        objective = -scvi.elbo(data)
    grads_scvi = tape.gradient(objective, scvi.kernel.trainable_variables)

    # compare gradients
    np.testing.assert_array_almost_equal(grads_svgp, grads_scvi, decimal=4)
