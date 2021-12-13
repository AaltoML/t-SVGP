import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import to_default_float

from src.models.tsvgp_white import t_SVGP_white

LENGTH_SCALE = 1.0
VARIANCE = 1.0
NUM_DATA = 8
EXTRA = 2
NOISE_VARIANCE = 0.3

rng = np.random.RandomState(123)
tf.random.set_seed(42)


def setup():

    N, M = 10, 3
    kernel = gpflow.kernels.RBF(lengthscales=LENGTH_SCALE, variance=VARIANCE)
    X = to_default_float(np.random.randn(N, 1))
    Y = to_default_float(np.random.randn(N, 1))
    Z = to_default_float(np.random.randn(M, 1))

    sgpr = gpflow.models.SGPR(
        data=(X, Y), kernel=kernel, inducing_variable=Z, noise_variance=NOISE_VARIANCE
    )

    tsgpr = t_SVGP_white(
        likelihood=gpflow.likelihoods.Gaussian(variance=NOISE_VARIANCE),
        kernel=kernel,
        inducing_variable=Z,
    )

    return (X, Y), sgpr, tsgpr


def setup_extra():

    N, M = 10, 3
    kernel = gpflow.kernels.RBF(lengthscales=LENGTH_SCALE, variance=VARIANCE)

    X = to_default_float(np.random.randn(N, 1))
    Y = to_default_float(np.random.randn(N, 1))
    Z = to_default_float(np.random.randn(M, 1))

    X_extra = to_default_float(np.random.randn(N, 1))
    Y_extra = to_default_float(np.random.randn(N, 1))

    X_concat = tf.concat([X, X_extra], axis=0)
    Y_concat = tf.concat([Y, Y_extra], axis=0)

    tsgpr = t_SVGP_white(
        likelihood=gpflow.likelihoods.Gaussian(variance=NOISE_VARIANCE),
        kernel=kernel,
        inducing_variable=Z,
    )

    tsgpr_ = t_SVGP_white(
        likelihood=gpflow.likelihoods.Gaussian(variance=NOISE_VARIANCE),
        kernel=kernel,
        inducing_variable=Z,
    )

    return (X, Y), (X_extra, Y_extra), (X_concat, Y_concat), tsgpr, tsgpr_


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

    tsgpr.natgrad_step((X, Y), lr=1.0)
    X = data[1]
    means, vars = sgpr.predict_f(X)
    means_, vars_ = tsgpr.predict_f(X)

    np.testing.assert_array_almost_equal(means, means_, decimal=4)
    np.testing.assert_array_almost_equal(vars, vars_, decimal=4)


def test_predict_extra_conditioning_tsgpr():
    """
    Testing prediction with extra conditioning
    """
    data, data_extra, data_concat, tsgpr, tsgpr_ = setup_extra()

    # model trained on data + data_extra
    tsgpr.natgrad_step(data_concat, lr=1.0)
    # model trained on data
    tsgpr_.natgrad_step(data, lr=1.0)

    X = data[1]
    # prediction on data
    means, vars = tsgpr.predict_f(X)
    # prediction on data with conditioning on extra data
    means_, vars_ = tsgpr_.predict_f_extra_data(X, extra_data=data_extra, jitter=0.0)

    np.testing.assert_array_almost_equal(means, means_, decimal=4)
    np.testing.assert_array_almost_equal(vars, vars_, decimal=4)
