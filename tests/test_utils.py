"""Module containing the unit tests for the util modules"""
import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow.conditionals import conditional

from src.sites import DiagSites
from src.util import (
    conditional_from_precision_sites,
    conditional_from_precision_sites_white,
    kl_from_precision_sites_white,
    posterior_from_dense_site_white,
    project_diag_sites,
)
from tests.tools import mean_cov_from_precision_site

LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 3
NUM_INDUCING = 2

rng = np.random.RandomState(123)
tf.random.set_seed(42)


def _setup(num_latent_gps=1):

    """Covariance and sites"""
    input_points = rng.rand(NUM_DATA, 1) * 2 - 1  # X values
    inducing_points = rng.rand(NUM_INDUCING, 1) * 2 - 1  # X values
    kernel = gpflow.kernels.SquaredExponential(lengthscales=LENGTH_SCALE, variance=VARIANCE)

    Kuu = kernel.K(inducing_points)
    Kuf = kernel.K(inducing_points, input_points)

    lambda_1 = tf.constant(np.random.randn(NUM_DATA, num_latent_gps))
    lambda_2 = tf.ones_like(np.random.randn(NUM_DATA, num_latent_gps) ** 2)
    sites = DiagSites(lambda_1, lambda_2)

    return Kuu, Kuf, kernel, sites, input_points, inducing_points


@pytest.mark.parametrize("num_latent_gps", [1, 2])
def test_site_conditionals(num_latent_gps):
    """Test the conditional using sites versus gpflow's conditional"""

    Kuu, Kuf, kernel, sites, input_points, inducing_points = _setup(num_latent_gps)
    l, L = project_diag_sites(Kuf, sites.lambda_1, sites.lambda_2, Kuu=None)
    l_white, L_white = project_diag_sites(Kuf, sites.lambda_1, sites.lambda_2, Kuu=Kuu)

    q_mu, cov = mean_cov_from_precision_site(Kuu, l, L)
    q_sqrt = tf.linalg.cholesky(cov)

    # gpflow conditional
    mu, var = conditional(
        input_points,
        inducing_points,
        kernel,
        q_mu,
        q_sqrt=q_sqrt,
        white=False,
    )

    # conditional for sites
    Kff = kernel.K_diag(input_points)[..., None]
    mu2, var2 = conditional_from_precision_sites_white(Kuu, Kff, Kuf, l, L=L)

    # conditional for projected sites
    Kff = kernel.K_diag(input_points)[..., None]
    mu3, var3 = conditional_from_precision_sites(Kuu, Kff, Kuf, l_white, L=L_white)

    # comparison
    np.testing.assert_array_almost_equal(mu, mu2, decimal=3)
    np.testing.assert_array_almost_equal(var, var2, decimal=3)
    np.testing.assert_array_almost_equal(mu, mu3, decimal=3)
    np.testing.assert_array_almost_equal(var, var3, decimal=3)


def test_mean_cov_from_precision_site():
    """Testing whether the direct inverse of A⁻¹ + LLᵀ matches the alternative derivation"""

    Kuu, Kuf, kernel, sites, _, _ = _setup()
    m = tf.shape(Kuu)[-1]
    l, L = project_diag_sites(Kuf, sites.lambda_1, sites.lambda_2, cholesky=True)

    # inverse of A^-1 + B, naive way
    Luu = tf.linalg.cholesky(Kuu)
    iKuu = tf.linalg.cholesky_solve(Luu, np.eye(m))
    iKuuLB = tf.linalg.cholesky_solve(Luu, L)
    B = tf.matmul(iKuuLB, iKuuLB, transpose_b=True)
    prec = iKuu + B
    Lprec = tf.linalg.cholesky(prec)
    inv2 = tf.linalg.cholesky_solve(Lprec, np.eye(m))
    mean2 = tf.linalg.cholesky_solve(Lprec[0], tf.matmul(iKuu, l))

    # inverse with woodbury
    mean1, inv1 = mean_cov_from_precision_site(Kuu, l, L)

    # comparison
    np.testing.assert_array_almost_equal(inv1, inv2, decimal=5)
    np.testing.assert_array_almost_equal(mean1, mean2, decimal=5)


def test_kl_from_precision_sites():
    """Testing that gpflow and the kl from sites match"""

    Kuu, Kuf, kernel, sites, input_points, inducing_points = _setup()
    l, L = project_diag_sites(Kuf, sites.lambda_1, sites.lambda_2, Kuu=None)
    q_mu, q_cov = mean_cov_from_precision_site(Kuu, l, L)
    q_sqrt = tf.linalg.cholesky(q_cov)

    # Gpflow KL
    KL_gpflow = gpflow.kullback_leiblers.gauss_kl(q_mu, q_sqrt, K=Kuu)

    # KL from sites
    KL = kl_from_precision_sites_white(Kuu, l, L=L)
    np.testing.assert_almost_equal(KL, KL_gpflow)


from src.models.tsvgp import posterior_from_dense_site


def test_posteriors_from_dense_sites():

    Kuu, Kuf, kernel, sites, input_points, inducing_points = _setup()

    l, L = project_diag_sites(Kuf, sites.lambda_1, sites.lambda_2, Kuu=Kuu)
    l_white, L_white = project_diag_sites(Kuf, sites.lambda_1, sites.lambda_2, Kuu=None)

    m, v = posterior_from_dense_site(Kuu, l, L)
    m_white, v_white = posterior_from_dense_site_white(
        Kuu, l_white, L_white @ tf.linalg.matrix_transpose(L_white)
    )

    np.testing.assert_array_almost_equal(m, m_white, decimal=3)
    np.testing.assert_array_almost_equal(v, v_white, decimal=3)
