import sys
from abc import ABC
from typing import Callable, List, Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import kullback_leiblers, default_jitter
from gpflow.base import Parameter, TensorLike
from gpflow.config import default_float
from gpflow.covariances import Kuf
from gpflow.inducing_variables.multioutput import SharedIndependentInducingVariables
from gpflow.kernels import (Kernel, SeparateIndependent, SquaredExponential, SharedIndependent, Matern52, RBF, Matern12)
# from gpflow.kullback_leiblers import prior_kl, gauss_kl
from gpflow_vgpmp.kullback_leiblers.prior_kl import prior_kl
from gpflow.utilities import triangular, positive
from gpflow_sampling.models import PathwiseSVGP
from gpflow_vgpmp.inducing_variables.inducing_variables import VariableInducingPoints, DerivativeInducingPoints
from gpflow_vgpmp.likelihoods.likelihood import VariationalMonteCarloLikelihood
from gpflow_vgpmp.utils.ops import initialize_Z, bounded_param, timing
from gpflow_vgpmp.covariances.multioutput.Kuus import Kuu

# <------ Exports
__all__ = "vgpmp"

from gpflow_vgpmp.utils.sdf_utils import SignedDensityField


# ============================================
# ---------------vGPMP Planner----------------
# ============================================


class VGPMP(PathwiseSVGP, ABC):
    def __init__(self, *args, prior: Callable = None, alpha, query_states, num_samples, num_bases, num_inducing,
                 parameters,
                 learning_rate, **kwargs):
        super(PathwiseSVGP, self).__init__(*args, **kwargs)
        self._velocities = tf.zeros(shape=(2, self.num_latent_gps), dtype=default_float()) + \
                          tf.random.normal((2, self.num_latent_gps), mean=0.0 + 1e-5, stddev=1e-5,
                                           dtype=default_float())
        self._query_states = self.likelihood.joint_sigmoid.inverse(
            tf.constant(query_states, dtype=default_float(), shape=(2, self.num_latent_gps)))
        self.optimizer = self.initialize_optimizer(learning_rate)
        self.num_samples = num_samples
        self.num_bases = num_bases
        self.num_inducing = num_inducing
        self.prior = prior
        self.planner_parameters = parameters
        self.alpha = Parameter(alpha, transform=positive(1e-4))

    def reinitialize(self):
        """
        Since optimization is stochastic it can happen that the optimization gets stuck in a local minimum.
        This function reinitializes the mean and variance of the variational distribution.
        """
        self._q_mu.assign(self.cached_q_mu)
        self._q_sqrt.assign(self.cached_q_sqrt)
        self.optimizer = self.initialize_optimizer(self.optimizer.learning_rate)

    @classmethod
    def initialize(cls,
                   sigma_obs: float = 0.05,
                   alpha: float = 1.0,
                   variance: float = 0.1,
                   lengthscales: List[float] = None,
                   rs: List[float] = None,
                   learning_rate: float = 0.1,
                   num_data=None,
                   num_output_dims=None,
                   num_inducing: int = 14,
                   num_samples: int = 51,
                   num_bases: int = 1024,
                   num_spheres: int = None,
                   offset: List = None,
                   query_states: List[np.array] = None,
                   joint_constraints: List = None,
                   velocity_constraints: List = None,
                   sdf: Optional['SignedDensityField'] = None,
                   robot: Optional['Robot'] = None,
                   kernels: List[Kernel] = None,
                   num_latent_gps: int = None,
                   parameters: dict = None,
                   train_sigma: bool = True,
                   no_frames_for_spheres: int = 7,
                   robot_name: str = None,
                   epsilon: float = 0.05,
                   q_mu: TensorLike = None,
                   **kwargs):

        if parameters is None:
            parameters = {}

        if num_latent_gps is None:
            num_latent_gps = num_output_dims

        if num_spheres is None:
            print(
                "Number of spheres has not been set. The simulator will now exit...")
            sys.exit()

        if lengthscales is None:
            print("Lengthscales have not been set. The simulator will now exit...")
            sys.exit()

        if offset is None:
            offset = [0, 0, 0]

        assert num_output_dims == num_latent_gps
        assert query_states is not None
        assert joint_constraints is not None
        assert len(lengthscales) == num_latent_gps

        Z = initialize_Z(num_latent_gps, num_inducing)

        if kernels is None:
            kernels = []

            for i in range(num_latent_gps):
                kern = Matern52(lengthscales=lengthscales[i], variance=variance)
                kernels.append(kern)
        kernel = SeparateIndependent(kernels)

        # Original inducing points
        _Z = DerivativeInducingPoints(Z=Z, dof=num_output_dims)

        likelihood = VariationalMonteCarloLikelihood(sigma_obs,
                                                     num_spheres,
                                                     robot,
                                                     parameters,
                                                     robot_name,
                                                     sdf,
                                                     rs,
                                                     offset,
                                                     joint_constraints,
                                                     velocity_constraints,
                                                     train_sigma,
                                                     no_frames_for_spheres,
                                                     epsilon)

        # handle q_mu here
        if q_mu is None:
            q_mu = likelihood.joint_sigmoid(np.zeros(shape=(num_inducing, num_latent_gps), dtype=default_float()))
        else:
            assert isinstance(q_mu, np.ndarray)
            if len(q_mu.shape) == 1:
                q_mu = tf.repeat(q_mu, num_inducing, axis=0)
            else:
                assert q_mu.shape == (num_inducing, num_latent_gps)

        return cls(kernel=kernel,
                   likelihood=likelihood,
                   inducing_variable=SharedIndependentInducingVariables(_Z),
                   num_latent_gps=num_latent_gps,
                   num_samples=num_samples,
                   num_bases=num_bases,
                   num_data=num_data,
                   query_states=query_states,
                   num_inducing=num_inducing,
                   parameters=parameters,
                   learning_rate=learning_rate,
                   alpha=alpha,
                   q_mu=q_mu,
                   **kwargs)

    @property
    def q_mu(self):
        return tf.concat([self.query_states, self._q_mu], axis=0)

    @property
    def query_states(self):
        return tf.concat([self._velocities, self._query_states], axis=0)

    @property
    def q_sqrt(self):
        """
        Manually whiten covariance matrix
        """
        # K = Kuu(self.inducing_variable.inducing_variable, self.kernel.kernel, jitter=default_jitter()) # sharedIndependent
        K = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(K)
        return L @ tf.pad(self._q_sqrt, [[0, 0], [4, 0], [4, 0]]) + \
            1e-6 * tf.pad(tf.eye(4, dtype=default_float()), [[0, self.num_inducing], [0, self.num_inducing]])

    def _init_variational_parameters(
            self,
            num_inducing: int,
            q_mu: Optional[np.ndarray],
            q_sqrt: Optional[np.ndarray],
            q_diag: bool,
    ) -> None:
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the
        routine initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with
        P, number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing:
            Number of inducing variables, typically refered to as M.
        :param q_mu:
            Mean of the variational Gaussian posterior. If None the function initializes
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt:
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function initializes `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag:
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """

        q_mu = self.likelihood.joint_sigmoid.inverse(q_mu)
        self._q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]
        self.cached_q_mu = q_mu
        np_q_sqrt: np.ndarray = np.array(
            [
                np.eye(num_inducing, dtype=default_float())
                for _ in range(self.num_latent_gps)
            ]
        )
        self._q_sqrt = Parameter(np_q_sqrt, transform=triangular())  # [P, M, M]
        self.cached_q_sqrt = np_q_sqrt


    @tf.function
    def elbo(self, data: tf.Tensor) -> tf.Tensor:
        r"""
        Estimate the evidence lower bound on the log marginal likelihood of the model
        by using decoupled sampling to construct a Monte Carlo integral.

        predict_f_samples produces S x N x D samples. The MC routine evaluates the mean
        log likelihood of S samples over N x D joint configurations. The log likelihood is
        handled differently depending on what type of objective is being optimized
        (collision proximity or soft constraints)

        Args:
            data (tf.Tensor): (2 x dof)
        Returns:
            ELBO (float): ELBO for the current posterior
        """

        with self.temporary_paths(num_samples=self.num_samples, num_bases=self.num_bases):
            f = self.predict_f_samples(data)  # S x N x D
        g = self.likelihood.joint_sigmoid(f)

        # func = tf.range(g.shape[0])
        # grads = tf.map_fn(lambda i: tf.gradients(g[i], data)[0], func, dtype=default_float())
        kl = prior_kl(self.inducing_variable, self.kernel, self._q_mu, self._q_sqrt, self.query_states)

        likelihood_obs = tf.reduce_mean(self.likelihood.log_prob(g), axis=0)  # log_prob produces S x N
        # likelihood_vel = tf.reduce_mean(-0.5 * tf.reduce_sum(tf.square(grads - 3.), axis=-1), axis=0)  # log_prob produces S x N
        return tf.reduce_sum(
            likelihood_obs) * self.alpha + - kl  # + tf.reduce_sum(( 1 / tf.squeeze(self.likelihood.variance) ) ** 2)

    @tf.function
    def debug_likelihood(self, data) -> tf.Tensor:
        r"""
        Method for debugging the likelihood function. This method is used in the
        environment loop to check the effects of e.g. sigma_obs on the overall
        value of the likelihood.

        Args:
            data (tf.Tensor): (1 x dof) tensor of joint positions
        Returns:
            Likelihood (float): likelihood of the current configuration
        """
        likelihood_obs = tf.reduce_mean(self.likelihood.log_prob(data), axis=0)  # log_prob produces S x N
        return tf.reduce_sum(likelihood_obs)

    # @timing
    # def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
    #     objective = self.elbo(data)
    #     if self.prior is not None:
    #         objective += self.prior(self)
    #     return objective
    @tf.autograph.experimental.do_not_convert
    def sample_from_posterior(self, X, robot, compute_uncertainty=False):

        mu, sigma = map(tf.squeeze, self.posterior().predict_f(X))
        mu = self.likelihood.joint_sigmoid(mu)
        with self.temporary_paths(num_samples=150, num_bases=self.num_bases):
            f = tf.squeeze(self.predict_f_samples(X))
        samples = self.likelihood.joint_sigmoid(f)

        if compute_uncertainty:
            uncertainty = tf.stack(
                [tf.stack([robot.compute_joint_positions(tf.reshape(tf.constant(time, dtype=default_float()), shape=[-1, 1]),
                                                         self.likelihood.sampler.craig_dh_convention)[0][-1] for time in
                           sample]) for sample in samples])
            uncertainty = tfp.stats.variance(uncertainty, sample_axis=0)
        else:
            uncertainty = tf.constant(1.0, dtype=default_float())

        return mu, samples[self.get_best_sample(samples)], samples[:7], 2 * tf.math.sqrt(uncertainty)

    def initialize_optimizer(self, learning_rate):
        return tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.8, beta_2=0.95)

    @tf.autograph.experimental.do_not_convert
    def get_best_sample(self, samples):
        cost = tf.reduce_sum(self.likelihood.log_prob(samples), axis=-1)
        # tf.print(self.likelihood.log_prob(samples)[tf.math.argmax(cost)], summarize=-1)
        return tf.math.argmax(cost)
