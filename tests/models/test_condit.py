import tensorflow as tf
import gpflow
import numpy as np
import pytest
from gpflow.utilities import to_default_float
from src.models.tsvgp_white import t_SVGP_white

LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 8
EXTRA = 2
NOISE_VARIANCE = 0.3

rng = np.random.RandomState(123)
tf.random.set_seed(42)

def setup():

    N, M = 10, 3
    kernel = gpflow.kernels.RBF(lengthscales=LENGTH_SCALE,variance=VARIANCE)
    X = to_default_float(np.random.randn(N, 1))
    Y = to_default_float(np.random.randn(N, 1))
    Z = to_default_float(np.random.randn(M, 1))

    sgpr = gpflow.models.SGPR(
        data=(X, Y), kernel=kernel, inducing_variable=Z, noise_variance=NOISE_VARIANCE
    )

    tsgpr = t_SVGP_white(
        likelihood=gpflow.likelihoods.Gaussian(variance=NOISE_VARIANCE), kernel=kernel, inducing_variable=Z
    )

    return (X, Y), sgpr, tsgpr

def setup_extra():

    N, M = 10, 3
    kernel = gpflow.kernels.RBF(lengthscales=LENGTH_SCALE,variance=VARIANCE)
    X = to_default_float(np.random.randn(N, 1))
    Y = to_default_float(np.random.randn(N, 1))
    Z = to_default_float(np.random.randn(M, 1))
    X_extra = to_default_float(np.random.randn(N, 1))
    Y_extra = to_default_float(np.random.randn(N, 1))

    X_concat = tf.concat([X, X_extra], axis=0)
    Y_concat = tf.concat([Y, Y_extra], axis=0)

    sgpr = gpflow.models.SGPR(
        data=(X_concat, Y_concat), kernel=kernel, inducing_variable=Z, noise_variance=NOISE_VARIANCE
    )

    tsgpr = t_SVGP_white(
        likelihood=gpflow.likelihoods.Gaussian(variance=NOISE_VARIANCE), kernel=kernel, inducing_variable=Z
    )

    return (X, Y), (X_extra, Y_extra), sgpr, tsgpr

# def _setup():
#     """Data, kernel and likelihood setup"""
#
#     def func(x):
#         return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)
#
#     input_points = rng.rand(NUM_DATA, 1) * 2 - 1  # X values
#     observations = func(input_points) + 0.2 * rng.randn(NUM_DATA, 1)
#
#     extra_input
#     extra_observation =
#
#     kernel = gpflow.kernels.SquaredExponential(lengthscales=LENGTH_SCALE, variance=VARIANCE)
#     variance = tf.constant(NOISE_VARIANCE, dtype=tf.float64)
#
#     return input_points, observations, kernel, variance

def setup_classification():
    """Data, kernel and likelihood setup"""

    def func(x):
        return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

    input_points = rng.rand(NUM_DATA, 1) * 2 - 1  # X values
    observations = func(input_points) + 0.2 * rng.randn(NUM_DATA, 1)

    kernel = gpflow.kernels.SquaredExponential(lengthscales=LENGTH_SCALE, variance=VARIANCE)
    variance = tf.constant(NOISE_VARIANCE, dtype=tf.float64)

    return input_points, observations, kernel, variance


def test_dsgpr_predict_against_gpflow_sgpr():
    """
    Testing marginal predictions of dsgpr
    """

    data, sgpr, tsgpr = setup()
    X, Y = data

    tsgpr.natgrad_step(X, Y, lr=1.0)
    X = data[1]
    means, vars = sgpr.predict_f(X)
    means_, vars_ = tsgpr.predict_f(X)

    np.testing.assert_array_almost_equal(means, means_, decimal=4)
    np.testing.assert_array_almost_equal(vars, vars_, decimal=4)

# @pytest.fixture(name="tsvgp_models_setup")
# def test_tsgpr_predict_extra_cond_against_gpflow_sgpr():
#     """
#     Testing marginal predictions of tsgpr
#     """
#
#     data, extra_data, sgpr, tsgpr = setup_extra()
#
#     X, Y = data
#     X_extra, Y_extra = extra_data
#
#     means, vars = sgpr.predict_f(X)
#
#     tsgpr.natgrad_step(X, Y, lr=1.0)
#     means_, vars_ = tsgpr.predict_f_extra_data(X, extra_data=(X_extra, Y_extra))
#
#     np.testing.assert_array_almost_equal(means, means_, decimal=4)
#     np.testing.assert_array_almost_equal(vars, vars_, decimal=4)
#
#
# def test_tsvgp_predict_extra_cond_against_gpflow_svgp():
#     """
#     Testing marginal predictions of tsvgp
#     """
#     """Creates a qSVGP model and a matched tSVGP model (E-step)"""
#
#     time_points, observations, kernel, noise_variance = _setup()
#     input_data = (
#         tf.constant(time_points),
#         tf.constant((observations > 0.0).astype(float)),
#     )
#
#     likelihood = gpflow.likelihoods.Bernoulli()
#
#     tsvgp_full = t_SVGP_white(
#         kernel=kernel,
#         likelihood=likelihood,
#         inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
#     )
#
#     tsvgp_cond = t_SVGP_white(
#         kernel=kernel,
#         likelihood=likelihood,
#         inducing_variable=gpflow.inducing_variables.InducingPoints(time_points),
#     )
#
#     natgrad_rep = 10
#     lr = 0.9
#     # one step of natgrads for tsvgp
#     for _ in range(natgrad_rep):
#         print(_)
#         tsvgp_full.natgrad_step(lr=lr)
#
#
#     return tsvgp, qsvgp, input_data
#
