import tensorflow as tf


def mean_cov_from_precision_site(A: tf.Tensor, l: tf.Tensor, L: tf.Tensor):
    """
    Computes the mean and covariance of a Gaussian with natural parameters
    Λ₂ = A⁻¹ + A⁻¹LLᵀA⁻¹
    λ₁ = A⁻¹ l

    The mean and covariance are computed as follows
    Σ = Λ⁻¹  = (A⁻¹  + A⁻¹LLᵀA⁻¹)⁻¹  = A(A + LLᵀ)⁻¹A
    μ = Λ⁻¹ A⁻¹l = Σ A⁻¹l
    = A(A + LLᵀ)⁻¹ l

    Inputs:
    :param A: tensor M x M
    :param l: tensor M x L
    :param L: tensor L x M x M
    """

    shape_constraints = [
        (A, ["M", "M"]),
        (L, ["L", "M", "M"]),
        (l, ["M", "L"]),
    ]
    tf.debugging.assert_shapes(
        shape_constraints,
        message="mean_cov_from_precision_site() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    R = L @ tf.linalg.matrix_transpose(L) + A
    LR = tf.linalg.cholesky(R)

    tmp = tf.linalg.triangular_solve(LR, A)
    cov = tf.matmul(tmp, tmp, transpose_a=True)
    mean = tf.matmul(tmp, tf.linalg.triangular_solve(LR, l), transpose_a=True)[0]
    return mean, cov
