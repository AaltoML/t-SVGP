"""Module containing the integration tests for the `t_SVGP_white` class."""
import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow.likelihoods import Bernoulli, Gaussian
from gpflow.optimizers import NaturalGradient

from src.tsvgp import t_SVGP
from src.tsvgp_white import t_SVGP_white

LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 8
NOISE_VARIANCE = 0.3

rng = np.random.RandomState(123)
tf.random.set_seed(42)


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


@pytest.fixture(name="t_svgp_normal_vs_white_init")
def _t_svgp_normal_vs_white_init():
    """Creates a GPR model and a matched tSVGP model (via natural gradient descent - single step)"""

    time_points, observations, kernel, noise_variance = _setup()
    input_data = (tf.constant(time_points), tf.constant(observations))

    likelihood = Gaussian(variance=noise_variance)
    t_svgp = t_SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
    )

    t_svgp_white = t_SVGP_white(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
    )
    return input_data, t_svgp, t_svgp_white


def test_init_elbo(t_svgp_normal_vs_white_init):
    data, t_svgp, t_svgp_white = t_svgp_normal_vs_white_init
    np.testing.assert_almost_equal(
        t_svgp.elbo(data), t_svgp_white.elbo(data), decimal=4
    )


def test_init_predictions(t_svgp_normal_vs_white_init):
    data, t_svgp, t_svgp_white = t_svgp_normal_vs_white_init
    m, v = t_svgp.predict_f(data[0])
    m_white, v_white = t_svgp_white.predict_f(data[0])
    np.testing.assert_array_almost_equal(m, m_white, decimal=4)
    np.testing.assert_array_almost_equal(v, v_white, decimal=4)


def test_prediction_after_update(t_svgp_normal_vs_white_init):
    data, t_svgp, t_svgp_white = t_svgp_normal_vs_white_init
    X, Y = data
    lr = 0.9

    # step on the t_svgp
    t_svgp.natgrad_step(X, Y, lr=lr)
    t_svgp_white.natgrad_step(X, Y, lr=lr)

    m, v = t_svgp.predict_f(data[0])
    m_white, v_white = t_svgp_white.predict_f(data[0])
    print(m)
    print(m_white)
    np.testing.assert_array_almost_equal(m, m_white, decimal=4)
    np.testing.assert_array_almost_equal(v, v_white, decimal=4)


@pytest.fixture(name="t_svgp_gpr_optim_setup")
def _t_svgp_gpr_optim_setup():
    """Creates a GPR model and a matched tSVGP model (via natural gradient descent - single step)"""

    time_points, observations, kernel, noise_variance = _setup()
    input_data = (tf.constant(time_points), tf.constant(observations) * 0)
    gpr = gpflow.models.GPR(
        data=input_data,
        kernel=kernel,
        mean_function=None,
        noise_variance=noise_variance,
    )

    likelihood = Gaussian(variance=noise_variance)
    t_svgp = t_SVGP_white(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
    )

    t_svgp.natgrad_step(*input_data, lr=1.0)

    return t_svgp, gpr


@pytest.fixture(name="t_svgp_svgp_optim_setup")
def _t_svgp_svgp_optim_setup():
    """Creates a SVGP model and a matched tSVGP model (E-step)"""

    def __t_svgp_svgp_optim_setup(num_latent_gps=1):

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

        t_svgp = t_SVGP_white(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
            num_latent_gps=num_latent_gps,
        )

        natgrad_rep = 20
        # one step of natgrads for t_svgp
        [t_svgp.natgrad_step(*input_data, lr=1.0) for _ in range(natgrad_rep)]
        # one step of natgrads for SVGP
        natgrad_opt = NaturalGradient(gamma=1.0)
        variational_params = [(svgp.q_mu, svgp.q_sqrt)]
        training_loss = svgp.training_loss_closure(input_data)
        [
            natgrad_opt.minimize(training_loss, var_list=variational_params)
            for _ in range(natgrad_rep)
        ]

        return t_svgp, svgp, input_data

    return __t_svgp_svgp_optim_setup


def test_t_svgp_elbo_optimal(t_svgp_gpr_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    t_svgp, gpr = t_svgp_gpr_optim_setup
    data = gpr.data
    np.testing.assert_almost_equal(
        t_svgp.elbo(data), gpr.log_marginal_likelihood(), decimal=4
    )


def test_t_svgp_unchanged_at_optimum(t_svgp_gpr_optim_setup):
    """Test that the update does not change sites at the optimum"""
    t_svgp, gpr = t_svgp_gpr_optim_setup
    # ELBO at optimum
    data = gpr.data
    optim_elbo = t_svgp.elbo(data)
    # site update step
    for i in range(10):
        t_svgp.natgrad_step(*data, lr=0.9)
        print(t_svgp.lambda_1)
    # ELBO after step
    new_elbo = t_svgp.elbo(data)

    np.testing.assert_almost_equal(optim_elbo, new_elbo, decimal=4)


def test_t_svgp_minibatch_same_elbo(t_svgp_gpr_optim_setup):
    """Test that the ELBO from minibactch is same as feeding full data for replicated data points"""

    t_svgp1, gpr = t_svgp_gpr_optim_setup
    t_svgp2, _ = t_svgp_gpr_optim_setup

    data = gpr.data
    t_svgp1.num_data = NUM_DATA
    x_d, y_d = data
    # Repeat the same data point
    x = x_d.numpy()[0].repeat(NUM_DATA)[:, None]
    y = y_d.numpy()[0].repeat(NUM_DATA)[:, None]
    data_repeat = (x, y)
    # Single data point elbo scaled for minibatch and same datapoint repeated
    optim_elbo2 = t_svgp2.elbo(data_repeat)
    optim_elbo1 = t_svgp1.elbo((x_d.numpy()[0][:, None], y_d.numpy()[0][:, None]))

    np.testing.assert_almost_equal(optim_elbo2, optim_elbo1, decimal=4)

# todo make broadcastable
@pytest.mark.parametrize("num_latent_gps", [1])
def test_gradient_wrt_hyperparameters(t_svgp_svgp_optim_setup, num_latent_gps):
    """Test that for matched posteriors (through natgrads, gradients wrt hyperparameters match for t_svgp and svgp"""

    # tsvgp and qsvgp models after exact E-step
    t_svgp, svgp, data = t_svgp_svgp_optim_setup(num_latent_gps)

    # gradient of SVGP elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(svgp.kernel.trainable_variables)
        objective = svgp.elbo(data)
    grads_svgp = tape.gradient(objective, svgp.kernel.trainable_variables)

    # gradient of t_svgp elbo wrt kernel hyperparameters
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(t_svgp.kernel.trainable_variables)
        objective = t_svgp.elbo(data)
    grads_t_svgp = tape.gradient(objective, t_svgp.kernel.trainable_variables)

    # compare elbos
    np.testing.assert_array_almost_equal(svgp.elbo(data), t_svgp.elbo(data), decimal=4)

    # compare gradients
    for g_svgp, g_t_svgp in zip(grads_svgp, grads_t_svgp):
        np.testing.assert_array_almost_equal(g_svgp, g_t_svgp, decimal=4)
