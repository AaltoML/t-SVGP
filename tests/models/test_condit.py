import tensorflow as tf
import gpflow
import numpy as np
from gpflow.utilities import to_default_float
from src.models.tsvgp_white import t_SVGP_white

def setup():

    N, M = 10, 3
    kernel = gpflow.kernels.RBF()
    X = to_default_float(np.random.randn(N, 1))
    Y = to_default_float(np.random.randn(N, 1))
    Z = to_default_float(np.random.randn(M, 1))

    sgpr = gpflow.models.SGPR(
        data=(X, Y), kernel=kernel, inducing_variable=Z
    )

    tsgpr = t_SVGP_white(
        likelihood=gpflow.likelihoods.Gaussian(), kernel=kernel, inducing_variable=Z
    )

    return (X, Y), sgpr, tsgpr

def setup_extra():

    N, M = 10, 3
    kernel = gpflow.kernels.RBF()
    X = to_default_float(np.random.randn(N, 1))
    Y = to_default_float(np.random.randn(N, 1))
    Z = to_default_float(np.random.randn(M, 1))
    X_extra = to_default_float(np.random.randn(N, 1))
    Y_extra = to_default_float(np.random.randn(N, 1))

    X_concat = tf.concat([X, X_extra], axis=0)
    Y_concat = tf.concat([Y, Y_extra], axis=0)

    sgpr = gpflow.models.SGPR(
        data=(X_concat, Y_concat), kernel=kernel, inducing_variable=Z
    )

    tsgpr = t_SVGP_white(
        likelihood=gpflow.likelihoods.Gaussian(), kernel=kernel, inducing_variable=Z
    )

    return (X, Y), (X_extra, Y_extra), sgpr, tsgpr

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


def test_tsgpr_predict_extra_cond_against_gpflow_sgpr():
    """
    Testing marginal predictions of tsgpr
    """

    data, extra_data, sgpr, tsgpr = setup_extra()

    X, Y = data
    X_extra, Y_extra = extra_data

    means, vars = sgpr.predict_f(X)

    tsgpr.natgrad_step(X, Y, lr=1.0)
    means_, vars_ = tsgpr.predict_f_extra_data(X, extra_data=(X_extra, Y_extra))

    np.testing.assert_array_almost_equal(means, means_, decimal=4)
    np.testing.assert_array_almost_equal(vars, vars_, decimal=4)
