"""
Module for the t-SVGP model with whitened parameterization
"""
import numpy as np
import tensorflow as tf
from gpflow import default_float, default_jitter
from gpflow.covariances import Kuf, Kuu
from gpflow.models import GPModel
from gpflow.models.model import RegressionData
from gpflow.models.training_mixins import InputData
from gpflow.models.util import inducingpoint_wrapper
from gpflow.types import MeanAndVariance

from src.sites import DenseSites
from src.util import (
    conditional_from_precision_sites_white,
    gradient_transformation_mean_var_to_expectation,
    kl_from_precision_sites_white,
    posterior_from_dense_site_white,
)


class t_SVGP_white(GPModel):
    """
    Class for the t-SVGP model with whitened paramterization
    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        lambda_1=None,
        lambda_2=None,
        num_data=None,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        GPModel.__init__(self, kernel, likelihood, mean_function, num_latent_gps)

        self.num_data = num_data
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        self.num_inducing = self.inducing_variable.num_inducing

        self._init_variational_parameters(self.num_inducing, lambda_1, lambda_2)

    def _init_variational_parameters(self, num_inducing, lambda_1, lambda_2):
        """
        Constructs the site parameters λ₁, Λ₂.
        for site t(u) = exp(uᵀλ₁ - ½ uᵀΛ₂u)

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically referred to as M.
        :param lambda_1: np.array or None
            First order natural parameter of the variational site.
        :param lambda_2_sqrt: np.array or None
            Second order natural parameter of the variational site.
        """

        lambda_1 = np.zeros((num_inducing, self.num_latent_gps)) if lambda_1 is None else lambda_1

        if lambda_2 is None:
            lambda_2 = [
                tf.eye(num_inducing, dtype=default_float()) * 1e-10
                for _ in range(self.num_latent_gps)
            ]
            lambda_2 = np.array(lambda_2)
        else:
            assert lambda_2.ndim == 3
            self.num_latent_gps = lambda_2.shape[0]

        self.sites = DenseSites(lambda_1=lambda_1, lambda_2=lambda_2)

    @property
    def lambda_1(self):
        return self.sites.lambda_1

    @property
    def lambda_2(self):
        return self.sites.lambda_2

    def get_mean_chol_cov_inducing_posterior(self):
        """
        Computes the mean and cholesky factor of the posterior
        on the inducing variables q(u) = 𝓝(u; m, S)
        S = (K⁻¹ + Λ₂)⁻¹ = (K⁻¹ + L₂L₂ᵀ)⁻¹ = K - KL₂W⁻¹L₂ᵀK , W = (I + L₂ᵀKL₂)⁻¹
        m = S λ₁
        """
        K_uu = Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [P, M, M] or [M, M]
        return posterior_from_dense_site_white(K_uu, self.lambda_1, self.lambda_2)


    @property
    def cache_statistics(self):
        return self.cache_statistics_from_data(self.data)

    def prior_kl(self) -> tf.Tensor:
        K_uu = Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [P, M, M] or [M, M]
        return kl_from_precision_sites_white(K_uu, self.lambda_1, L2=self.lambda_2)

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        K_uu = Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [P, M, M] or [M, M]
        K_uf = Kuf(self.inducing_variable, self.kernel, Xnew)  # [P, M, M] or [M, M]
        K_ff = self.kernel.K_diag(Xnew)[..., None]
        mu, var = conditional_from_precision_sites_white(
            K_uu, K_ff, K_uf, self.lambda_1, L2=self.lambda_2
        )
        tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var

    def predict_f_extra_data(
            self, Xnew: InputData, extra_data=RegressionData, iter=1, lr=1.0) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """

        grad_mu = self.compute_data_natural_params(extra_data)

        lambda_1 = self.lambda_1
        lambda_2 = -0.5 * self.lambda_2

        K_uu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        lambda_1c = lambda_1 + K_uu @ grad_mu[0]
        lambda_2c = -2*(lambda_2 + K_uu @ grad_mu[1] @ K_uu)

        # predicting at new inputs
        K_uf = Kuf(self.inducing_variable, self.kernel, Xnew)
        K_ff = self.kernel.K_diag(Xnew)[..., None]

        mu, var = conditional_from_precision_sites_white(
                    K_uu, K_ff, K_uf, lambda_1c, L2=lambda_2c)

        return mu + self.mean_function(Xnew), var

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """The variational lower bound"""
        return self.elbo()

    def compute_data_natural_params(self, data, jitter=1e-9, nat_params=None):
        X, Y = data
        mean, var = self.predict_f(X)
        meanZ, varZ = self.predict_f(self.inducing_variable.Z)

        with tf.GradientTape() as g:
            g.watch([mean, var])
            ve = self.likelihood.variational_expectations(mean, var, Y)
        grads = g.gradient(ve, [mean, var])

        Id = tf.eye(self.num_inducing, dtype=tf.float64)

        # Compute the projection matrix A from prior information
        K_uu = Kuu(self.inducing_variable, self.kernel)
        K_uf = Kuf(self.inducing_variable, self.kernel, X)  # [P, M, M] or [M, M]
        chol_Kuu = tf.linalg.cholesky(K_uu + Id * jitter)
        A = tf.transpose(tf.linalg.cholesky_solve(chol_Kuu, K_uf))

        A = tf.tile(A[..., None], [1, 1, self.num_latent_gps])

        grads = [
            tf.einsum("nml,nl->ml", A, grads[0]),
            tf.einsum("nml,nol,nl->lmo", A, A, grads[1]),
        ]

        # chain rule at f
        grad_mu = gradient_transformation_mean_var_to_expectation(meanZ, grads)

        return grad_mu


    def natgrad_step(self, dataset, lr=0.1, jitter=1e-9):
        """Takes natural gradient step in Variational parameters in the local parameters
        λₜ = rₜ▽[Var_exp] + (1-rₜ)λₜ₋₁

        Input:
        :param: X : N x D
        :param: Y:  N x 1
        :param: lr: Scalar

        Output:
        Updates the params
        """

        X, Y = dataset

        # chain rule at f
        grad_mu = self.compute_data_natural_params((X, Y))
        K_uu = Kuu(self.inducing_variable, self.kernel)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, dtype=tf.float64)
            minibatch_size = tf.cast(tf.shape(X)[0], dtype=tf.float64)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, dtype=tf.float64)

        lambda_1 = self.lambda_1
        lambda_2 = -0.5 * self.lambda_2

        # compute update in natural form
        lambda_1 = (1.0 - lr) * lambda_1 + lr * scale * K_uu @ grad_mu[0]
        lambda_2 = (1.0 - lr) * lambda_2 + lr * scale * K_uu @ grad_mu[1] @ K_uu

        self.lambda_1.assign(lambda_1)
        self.lambda_2.assign(-2.0 * lambda_2)
