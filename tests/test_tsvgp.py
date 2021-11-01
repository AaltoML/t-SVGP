"""Module containing the integration tests for the `t_SVGP` class."""
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


@pytest.fixture(name="tsvgp_gpr_optim_setup")
def _tsvgp_gpr_optim_setup():
    """Creates a GPR model and a matched tSVGP model (via natural gradient descent - single step)"""

    time_points, observations, kernel, noise_variance = _setup()
    input_data = (tf.constant(time_points), tf.constant(observations))
    gpr = gpflow.models.GPR(
        data=input_data,
        kernel=kernel,
        mean_function=None,
        noise_variance=noise_variance,
    )

    likelihood = Gaussian(variance=noise_variance)
    tsvgp = t_SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
    )
    for i in range(10):
        tsvgp.natgrad_step(*input_data, lr=0.9)

    return tsvgp, gpr


@pytest.fixture(name="tsvgp_qsvgp_optim_setup")
def _tsvgp_qsvgp_optim_setup():
    """Creates a SVGP model and a matched tSVGP model (E-step)"""

    def __tsvgp_qsvgp_optim_setup(num_latent_gps=1):

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

        tsvgp = t_SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
            num_latent_gps=num_latent_gps,
        )

        natgrad_rep = 20
        # one step of natgrads for tSVGP
        [tsvgp.natgrad_step(*input_data, lr=1.0) for _ in range(natgrad_rep)]
        # one step of natgrads for qSVGP
        natgrad_opt = NaturalGradient(gamma=1.0)
        variational_params = [(svgp.q_mu, svgp.q_sqrt)]
        training_loss = svgp.training_loss_closure(input_data)
        [
            natgrad_opt.minimize(training_loss, var_list=variational_params)
            for _ in range(natgrad_rep)
        ]

        return tsvgp, svgp, input_data

    return __tsvgp_qsvgp_optim_setup


def _setup():
    """Data, kernel and likelihood setup"""

    def func(x):
        return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

    input_points = rng.rand(NUM_DATA, 1) * 2 - 1  # X values
    observations = func(input_points) + 0.2 * rng.randn(NUM_DATA, 1)

    kernel = gpflow.kernels.SquaredExponential(lengthscales=LENGTH_SCALE, variance=VARIANCE)
    variance = tf.constant(NOISE_VARIANCE, dtype=tf.float64)

    return input_points, observations, kernel, variance


def test_tsvgp_elbo_optimal(tsvgp_gpr_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    tsvgp, gpr = tsvgp_gpr_optim_setup
    data = gpr.data
    np.testing.assert_almost_equal(tsvgp.elbo(data), gpr.log_marginal_likelihood(), decimal=4)


def test_predictions_match_tsvgp_gpr_optimal(tsvgp_gpr_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    tsvgp, gpr = tsvgp_gpr_optim_setup
    X = gpr.data[0] + 1.0
    mu_tsvgp, var_tsvgp = tsvgp.predict_f(X)
    mu_gpr, var_gpr = gpr.predict_f(X)
    np.testing.assert_array_almost_equal(mu_tsvgp, mu_gpr, decimal=4)
    np.testing.assert_array_almost_equal(var_tsvgp, var_gpr, decimal=4)


@pytest.mark.parametrize("num_latent_gps", [1, 2])
def test_predictions_match_tsvgp_qsvgp_optimal(tsvgp_qsvgp_optim_setup, num_latent_gps):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    tsvgp, qsvgp, data = tsvgp_qsvgp_optim_setup(num_latent_gps)
    X = data[0] + 0.1
    mu_tsvgp, var_tsvgp = tsvgp.new_predict_f(X)
    mu_qsvgp, var_qsvgp = qsvgp.predict_f(X)
    np.testing.assert_array_almost_equal(mu_tsvgp, mu_qsvgp, decimal=4)
    np.testing.assert_array_almost_equal(var_tsvgp, var_qsvgp, decimal=4)


def test_tsvgp_unchanged_at_optimum(tsvgp_gpr_optim_setup):
    """Test that the update does not change sites at the optimum"""
    tsvgp, gpr = tsvgp_gpr_optim_setup
    # ELBO at optimum
    data = gpr.data
    optim_elbo = tsvgp.elbo(data)
    # site update step
    tsvgp.natgrad_step(*data, lr=0.9)
    # ELBO after step
    new_elbo = tsvgp.elbo(data)

    np.testing.assert_almost_equal(optim_elbo, new_elbo, decimal=4)


def test_tsvgp_minibatch_same_elbo(tsvgp_gpr_optim_setup):
    """Test that the ELBO from minibactch is same as feeding full data for replicated data points"""

    tsvgp1, gpr = tsvgp_gpr_optim_setup
    tsvgp2, _ = tsvgp_gpr_optim_setup

    data = gpr.data
    tsvgp1.num_data = NUM_DATA
    x_d, y_d = data
    # Repeat the same data point
    x = x_d.numpy()[0].repeat(NUM_DATA)[:, None]
    y = y_d.numpy()[0].repeat(NUM_DATA)[:, None]
    data_repeat = (x, y)
    # Single data point elbo scaled for minibatch and same datapoint repeated
    optim_elbo2 = tsvgp2.elbo(data_repeat)
    optim_elbo1 = tsvgp1.elbo((x_d.numpy()[0][:, None], y_d.numpy()[0][:, None]))

    np.testing.assert_almost_equal(optim_elbo2, optim_elbo1, decimal=4)


@pytest.mark.parametrize("num_latent_gps", [1, 2])
def test_gradient_wrt_hyperparameters(tsvgp_qsvgp_optim_setup, num_latent_gps):
    """Test that for matched posteriors (through natgrads, gradients wrt hyperparameters match for tsvgp and qsvgp"""

    # tsvgp and vgp models after exact E-step
    tsvgp, qsvgp, data = tsvgp_qsvgp_optim_setup(num_latent_gps)

    # gradient of SVGP elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(qsvgp.kernel.trainable_variables)
        objective = qsvgp.training_loss(data)
    grads_qsvgp = tape.gradient(objective, qsvgp.kernel.trainable_variables)

    # gradient of tsvgp elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(tsvgp.kernel.trainable_variables)
        objective = -tsvgp.elbo(data)
    grads_tsvgp = tape.gradient(objective, tsvgp.kernel.trainable_variables)

    # compare gradients
    np.testing.assert_array_almost_equal(grads_qsvgp, grads_tsvgp, decimal=4)
