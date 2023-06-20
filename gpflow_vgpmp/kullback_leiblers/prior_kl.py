import sys

import tensorflow as tf
from gpflow.base import TensorLike
from gpflow.covariances import Kuu
from gpflow.inducing_variables import SharedIndependentInducingVariables
from gpflow.kernels import SeparateIndependent, SharedIndependent
from gpflow.utilities import Dispatcher
from gpflow.config import default_jitter
from gpflow.kullback_leiblers import gauss_kl


prior_kl = Dispatcher("prior_kl")


@prior_kl.register(SharedIndependentInducingVariables, SeparateIndependent, TensorLike, TensorLike, TensorLike)
def prior_kl_separateindependent(inducing_variable, kernel, q_mu, q_sqrt, query_states):
    """
    Computes the KL divergence between the variational posterior and the prior.
    Shift the mean of the variational posterior to the mean of the prior.
    For SeparateIndependent kernel.
    """

    n = 4
    K = Kuu(inducing_variable, kernel, jitter=default_jitter())
    L = tf.linalg.cholesky(K)
    # Subtract prior mean from q_mu, then whiten
    p_mu = tf.linalg.cholesky_solve(L[..., :n, :n], tf.transpose(query_states)[..., None])
    p_mu = tf.matmul(K[..., :n], p_mu)
    q_mu = tf.concat([query_states, q_mu], axis=0)

    whitened_diff = tf.linalg.triangular_solve(L, tf.transpose(q_mu)[..., None] - p_mu)
    whitened_diff = tf.transpose(tf.squeeze(whitened_diff))[n:, ...]

    return gauss_kl(whitened_diff, q_sqrt)


@prior_kl.register(SharedIndependentInducingVariables, SharedIndependent, TensorLike, TensorLike, TensorLike)
def prior_kl_sharedindependent(inducing_variable, kernel, q_mu, q_sqrt, query_states):
    """
    Functionally equivalent to prior_kl_separateindependent, but for SharedIndependent kernel
    """
    n = len(inducing_variable.inducing_variable.ny)
    K = Kuu(inducing_variable.inducing_variable, kernel.kernel, jitter=default_jitter())
    L = tf.linalg.cholesky(K)

    # Subtract prior mean from q_mu, then whiten
    p_mu = K[..., :n] @ tf.linalg.cholesky_solve(L[..., :n, :n], query_states)

    q_mu = tf.concat([query_states, q_mu], axis=0)

    whitened_diff = tf.linalg.triangular_solve(L, q_mu - p_mu)[n:, ...]
    return gauss_kl(whitened_diff, q_sqrt)
