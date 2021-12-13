"""
Utilities for the model classes
"""

from typing import Optional

import tensorflow as tf
from gpflow import default_float


def conditional_from_precision_sites_white(
    Kuu: tf.Tensor,
    Kff: tf.Tensor,
    Kuf: tf.Tensor,
    l: tf.Tensor,
    L: tf.Tensor = None,
    L2=None,
    jitter=1e-9,
):
    """
    Given a g₁ and g2, and distribution p and q such that
      p(g₂) = N(g₂; 0, Kuu)

      p(g₁) = N(g₁; 0, Kff)
      p(g₁ | g₂) = N(g₁; Kfu (Kuu⁻¹) g₂, Kff - Kfu (Kuu⁻¹) Kuf)

    And  q(g₂) = N(g₂; μ, Σ) such that
        Σ⁻¹  = Kuu⁻¹  + Kuu⁻¹LLᵀKuu⁻¹
        Σ⁻¹μ = Kuu⁻¹l

    This method computes the mean and (co)variance of
      q(g₁) = ∫ q(g₂) p(g₁ | g₂) dg₂ = N(g₂; μ*, Σ**)

    with
    Σ** = k** - kfu Kuu⁻¹ kuf - kfu Kuu⁻¹ Σ Kuu⁻¹ kuf
        = k** - kfu Kuu⁻¹kuf + kfu (Kuu + LLᵀ)⁻¹ kuf
    μ* = k*u Kuu⁻¹ m
       = k*u Kuu⁻¹ Λ⁻¹ Kuu⁻¹ l
       = k*u (Kuu + LLᵀ)⁻¹ l

    Inputs:
    :param Kuu: tensor M x M
    :param Kff: tensor N x 1
    :param Kuf: tensor M x N
    :param L: tensor L x M x M
    :param l: tensor M x L
    """

    shape_constraints = [
        (Kuu, ["M", "M"]),
        (Kuf, ["M", "N"]),
        (Kff, ["N", 1]),
        (l, ["M", "L"]),
    ]

    if L2 is not None:
        shape_constraints.append(
            (L2, ["L", "M", "M"]),
        )
    if L is not None:
        shape_constraints.append(
            (L, ["L", "M", "M"]),
        )
    tf.debugging.assert_shapes(
        shape_constraints,
        message="conditional_from_precision_sites() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    if L2 is None:
        L2 = L @ tf.linalg.matrix_transpose(L)

    m = Kuu.shape[-1]
    I = tf.eye(m, dtype=tf.float64)
    R = L2 + Kuu + I * jitter
    LR = tf.linalg.cholesky(R)
    LA = tf.linalg.cholesky(Kuu)[None]

    tmp1 = tf.linalg.triangular_solve(LR, Kuf)
    tmp2 = tf.linalg.triangular_solve(LA, Kuf)

    cov = Kff - tf.linalg.matrix_transpose(
        tf.reduce_sum(tf.square(tmp2), axis=-2) - tf.reduce_sum(tf.square(tmp1), axis=-2)
    )
    mean = tf.matmul(Kuf, tf.linalg.cholesky_solve(LR, l), transpose_a=True)[0]
    return mean, cov


def conditional_from_precision_sites(
    Kuu: tf.Tensor,
    Kff: tf.Tensor,
    Kuf: tf.Tensor,
    l: tf.Tensor,
    L: tf.Tensor = None,
    L2=None,
):
    """
    Given a g₁ and g2, and distribution p and q such that
      p(g₂) = N(g₂; 0, Kuu)

      p(g₁) = N(g₁; 0, Kff)
      p(g₁ | g₂) = N(g₁; Kfu (Kuu⁻¹) g₂, Kff - Kfu (Kuu⁻¹) Kuf)

    And  q(g₂) = N(g₂; μ, Σ) such that
        Σ⁻¹  = Kuu⁻¹  + LLᵀ
        Σ⁻¹μ = l

    This method computes the mean and (co)variance of
      q(g₁) = ∫ q(g₂) p(g₁ | g₂) dg₂ = N(g₂; μ*, Σ**)

    with
    Σ** = k** - kfu Kuu⁻¹ kuf - kfu Kuu⁻¹ Σ Kuu⁻¹ kuf
        = k** - kfu Kuu⁻¹kuf + kfu Kuu⁻¹ (Kuu⁻¹  + LLᵀ)⁻¹ Kuu⁻¹ kuf
        = k** - kfu Kuu⁻¹kuf + kfu Kuu⁻¹ (Kuu - Kuu L(I + LᵀKuuL)⁻¹Lᵀ Kuu) Kuu⁻¹ kuf
        = k** - kfu L(I + LᵀKuuL)⁻¹Lᵀ kuf
        = k** - kfu LW⁻¹Lᵀ kuf
        = k** - kfu L Lw⁻ᵀLw⁻¹ Lᵀ kuf
        = k** - (Lw⁻¹ Lᵀ kuf)ᵀ Lw⁻¹ Lᵀ kuf
        = k** - (D kuf)ᵀ D kuf

    μ* = k*u Kuu⁻¹ m
       = k*u Kuu⁻¹ Σ l
       = k*u Kuu⁻¹ (Kuu⁻¹  + LLᵀ)⁻¹ l
       = k*u Kuu⁻¹ (Kuu - Kuu L(I + LᵀKuuL)⁻¹Lᵀ Kuu) l
       = k*u l - k*u L(I + LᵀKuuL)⁻¹Lᵀ Kuu l
       = k*u l - k*u LW⁻¹Lᵀ Kuu l
       = k*u l - (Lw⁻¹ Lᵀ kuf)ᵀ Lw⁻¹ Lᵀ Kuu l
       = k*u l - (D kuf)ᵀ D Kuu l

    Inputs:
    :param Kuu: tensor M x M
    :param Kff: tensor N x 1
    :param Kuf: tensor M x N
    :param L: tensor L x M x M
    :param l: tensor M x L
    """

    shape_constraints = [
        (Kuu, ["M", "M"]),
        (Kuf, ["M", "N"]),
        (Kff, ["N", 1]),
        (l, ["M", "L"]),
    ]

    if L2 is not None:
        shape_constraints.append(
            (L2, ["L", "M", "M"]),
        )
    if L is not None:
        shape_constraints.append(
            (L, ["L", "M", "M"]),
        )
    tf.debugging.assert_shapes(
        shape_constraints,
        message="conditional_from_precision_sites() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    if L is None:
        L = tf.linalg.cholesky(L2)

    m = Kuu.shape[-1]
    Id = tf.eye(m, dtype=default_float())
    C = tf.linalg.cholesky(Kuu)

    # W = I + Lₜᵀ Lₚ Lₚᵀ Lₜ, chol(W)
    CtL = tf.matmul(C, L, transpose_a=True)
    W = Id + tf.matmul(CtL, CtL, transpose_a=True)
    chol_W = tf.linalg.cholesky(W)

    D = tf.linalg.triangular_solve(chol_W, tf.linalg.matrix_transpose(L))
    tmp = tf.matmul(D, Kuf)

    mean = tf.matmul(Kuf, l, transpose_a=True) - tf.linalg.matrix_transpose(
        tf.reduce_sum(
            tf.matmul(D, tf.matmul(Kuu, tf.linalg.matrix_transpose(l)[..., None])) * tmp, axis=-2
        )
    )

    cov = Kff - tf.linalg.matrix_transpose(tf.reduce_sum(tf.square(tmp), axis=-2))
    return mean, cov


def project_diag_sites(
    Kuf: tf.Tensor,
    lambda_1: tf.Tensor,
    lambda_2: tf.Tensor,
    Kuu: Optional[tf.Tensor] = None,
    cholesky=True,
):
    """
    From  Kuu, Kuf, λ₁, λ₂, computes statistics
    L = Kuu⁻¹ (Σₙ Kufₙ λ₂ₙ Kfₙu) Kuu⁻¹
    l = Kuu⁻¹ (Σₙ Kufₙ λ₁ₙ)

    :param Kuf: L x M x N
    :param lambda_2: N x L
    :param lambda_1: N x L
    :param Kuu: L x M x M

    return:
    l : M x L
    L : L x M x M
    """

    shape_constraints = [
        (Kuf, [..., "M", "N"]),
        (lambda_1, ["N", "L"]),
        (lambda_2, ["N", "L"]),
    ]
    if Kuu is not None:
        shape_constraints.append((Kuu, [..., "M", "M"]))
    tf.debugging.assert_shapes(
        shape_constraints,
        message="conditional_from_precision_sites() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    num_latent = lambda_1.shape[-1]
    P = tf.tile(Kuf[None], [num_latent, 1, 1]) if tf.rank(Kuf) == 2 else Kuf
    if Kuu is not None:
        Kuu = Kuu[None] if tf.rank(Kuu) == 2 else Kuu
        Luu = tf.linalg.cholesky(Kuu)
        P = tf.linalg.cholesky_solve(Luu, P)

    l = tf.einsum("lmn,nl->ml", P, lambda_1)
    L = tf.einsum("lmn,lon,nl->lmo", P, P, lambda_2)
    if cholesky:
        L = tf.linalg.cholesky(L)
    return l, L


def kl_from_precision_sites_white(A, l, L=None, L2=None):
    """
    Computes the kl divergence KL[q(f)|p(f)]

    where:
    q(f) = N(μ,Σ) and covariance of a Gaussian with natural parameters
    with
        Σ = Λ⁻¹  = (A⁻¹  + A⁻¹LLᵀA⁻¹)⁻¹  = A(A + LLᵀ)⁻¹A
        μ = Λ⁻¹ A⁻¹l = Σ A⁻¹l

    p(f) = N(0,A)
    """

    shape_constraints = [
        (A, ["M", "M"]),
        (l, ["N", "L"]),
    ]
    if L is not None:
        shape_constraints.append((L, [..., "M", "M"]))
    if L2 is not None:
        shape_constraints.append((L2, [..., "M", "M"]))

    tf.debugging.assert_shapes(
        shape_constraints,
        message="kl_from_prec_precond() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    if L2 is None:
        L2 = L @ tf.linalg.matrix_transpose(L)
    m = tf.shape(L2)[-2]

    R = L2 + A
    LR = tf.linalg.cholesky(R)
    LA = tf.linalg.cholesky(A)

    # log det term
    log_det = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(LR)))) - tf.reduce_sum(
        tf.math.log(tf.square(tf.linalg.diag_part(LA)))
    )

    # trace term
    tmp = tf.linalg.triangular_solve(LR, LA)
    trace_plus_const = tf.reduce_sum(tf.square(tmp)) - tf.cast(m, A.dtype)

    # mahalanobis term
    mahalanobis = tf.reduce_sum(
        tf.square(tf.matmul(LA, tf.linalg.cholesky_solve(LR, l), transpose_a=True))
    )

    return 0.5 * (log_det + trace_plus_const + mahalanobis)


def kl_from_precision_sites(A, l, L=None, L2=None):
    """
    Computes the kl divergence KL[q(f)|p(f)]

    where:
    q(f) = N(μ,Σ) and covariance of a Gaussian with natural parameters
    with
        Σ = Λ⁻¹  = (A⁻¹  + A⁻¹LLᵀA⁻¹)⁻¹  = A(A + LLᵀ)⁻¹A
        μ = Λ⁻¹ A⁻¹l = Σ A⁻¹l

    p(f) = N(0,A)
    """

    shape_constraints = [
        (A, ["M", "M"]),
        (l, ["N", "L"]),
    ]
    if L is not None:
        shape_constraints.append((L, [..., "M", "M"]))
    if L2 is not None:
        shape_constraints.append((L2, [..., "M", "M"]))

    tf.debugging.assert_shapes(
        shape_constraints,
        message="kl_from_prec_precond() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    if L2 is None:
        L2 = L @ tf.linalg.matrix_transpose(L)
    m = tf.shape(L2)[-2]

    R = L2 + A
    LR = tf.linalg.cholesky(R)
    LA = tf.linalg.cholesky(A)

    # log det term
    log_det = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(LR)))) - tf.reduce_sum(
        tf.math.log(tf.square(tf.linalg.diag_part(LA)))
    )

    # trace term
    tmp = tf.linalg.triangular_solve(LR, LA)
    trace_plus_const = tf.reduce_sum(tf.square(tmp)) - tf.cast(m, A.dtype)

    # mahalanobis term
    mahalanobis = tf.reduce_sum(
        tf.square(tf.matmul(LA, tf.linalg.cholesky_solve(LR, l), transpose_a=True))
    )

    return 0.5 * (log_det + trace_plus_const + mahalanobis)


def posterior_from_dense_site(K, lambda_1, lambda_2_sqrt):
    """
    Returns the mean and cholesky factor of the density q(u) = p(u)t(u) = 𝓝(u; m, S)
    where p(u) = 𝓝(u; 0, K) and t(u) = exp(uᵀλ₁ - ½ uᵀΛ₂u)

    P = -2Λ₂

    S = (K⁻¹ + P)⁻¹ = (K⁻¹ + LLᵀ)⁻¹ = K - KLW⁻¹LᵀK , W = (I + LᵀKL)
    m = S λ₁

    Input:
    :param: K : M x M
    :param: lambda_1: M x P
    :param: lambda_2: P x M x M

    Output:
    m: M x P
    chol_S: P x M x M
    """
    shape_constraints = [
        (K, [..., "M", "M"]),
        (lambda_1, ["N", "L"]),
        (lambda_2_sqrt, ["L", "M", "M"]),
    ]
    tf.debugging.assert_shapes(
        shape_constraints,
        message="posterior_from_dense_site() arguments ",
    )

    L = lambda_2_sqrt  # L - likelihood precision square root
    m = K.shape[-1]
    Id = tf.eye(m, dtype=default_float())
    C = tf.linalg.cholesky(K)

    # W = I + Lₜᵀ Lₚ Lₚᵀ Lₜ, chol(W)
    CtL = tf.matmul(C, L, transpose_a=True)
    W = Id + tf.matmul(CtL, CtL, transpose_a=True)
    chol_W = tf.linalg.cholesky(W)

    # S_q = K - K W⁻¹K = K - [Lw⁻¹ K]ᵀ[Lw⁻¹ K]
    LtK = tf.matmul(L, K, transpose_a=True)
    iwLtK = tf.linalg.triangular_solve(chol_W, LtK, lower=True, adjoint=False)
    S_q = K - tf.matmul(iwLtK, iwLtK, transpose_a=True)
    chol_S_q = tf.linalg.cholesky(S_q)
    m_q = tf.einsum("lmn,nl->ml", S_q, lambda_1)

    return m_q, chol_S_q


def posterior_from_dense_site_white(K, lambda_1, lambda_2, jitter=1e-9):
    """
    Returns the mean and cholesky factor of the density q(u) = p(u)t(u) = 𝓝(u; m, S)
    where p(u) = 𝓝(u; 0, K) and t(u) = exp(uᵀλ₁ - ½ uᵀΛ₂u)

    S = Λ₂⁻¹ = (K⁻¹ + K⁻¹PK⁻¹)⁻¹ = = [K⁻¹(K + P)K⁻¹]⁻¹  = K(K + P)⁻¹K
    m = S K⁻¹λ₁ = K(K + LLᵀ)⁻¹λ₁

    Input:
    :param: K : M x M
    :param: lambda_1: M x P
    :param: lambda_2: P x M x M

    Output:
    m: M x P
    chol_S: P x M x M
    """

    shape_constraints = [(K, ["M", "M"]), (lambda_1, ["N", "L"]), (lambda_2, ["L", "M", "M"])]
    tf.debugging.assert_shapes(
        shape_constraints,
        message="posterior_from_dense_site_white() arguments ",
    )
    P = lambda_2  # L - likelihood precision square root
    m = K.shape[-1]
    Id = tf.eye(m, dtype=tf.float64)
    R = K + P
    LR = tf.linalg.cholesky(R + Id * jitter)
    iLRK = tf.linalg.triangular_solve(LR, K)
    S_q = tf.matmul(iLRK, iLRK, transpose_a=True)
    chol_S_q = tf.linalg.cholesky(S_q)
    m_q = (K @ tf.linalg.cholesky_solve(LR, lambda_1))[0]
    return m_q, chol_S_q


def gradient_transformation_mean_var_to_expectation(inputs, grads):
    """
    Transforms gradient 𝐠 of a function wrt [μ, σ²]
    into its gradients wrt to [μ, σ² + μ²]
    :param inputs: [μ, σ²]
    :param grads: 𝐠
    Output:
    ▽μ
    """
    return grads[0] - 2.0 * tf.einsum("lmo,ol->ml", grads[1], inputs), grads[1]
