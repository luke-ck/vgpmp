import sys
from typing import Callable, List, Optional

import numpy as np
import tensorflow as tf
from gpflow.base import Parameter, AnyNDArray
from gpflow.config import default_float
from gpflow.inducing_variables.multioutput import SharedIndependentInducingVariables
from gpflow.kernels import (Kernel, SeparateIndependent, SquaredExponential)
from gpflow.utilities import triangular
from gpflow_sampling.models import PathwiseSVGP

from gpflow_vgpmp.inducing_variables.inducing_variables import VariableInducingPoints
from gpflow_vgpmp.likelihoods.likelihood import VariationalMonteCarloLikelihood
from gpflow_vgpmp.utils.miscellaneous import timing
from gpflow_vgpmp.utils.ops import initialize_Z
from gpflow_vgpmp.utils.sampler import Sampler

# <------ Exports
__all__ = ("vgpmp")


# =========================================
# ---------------GP Planner----------------
# =========================================


class VGPMP(PathwiseSVGP):
    def __init__(self, *args, prior: Callable = None, query_states, num_samples, num_bases, num_inducing, parameters,
                 learning_rate, **kwargs):
        super(PathwiseSVGP, self).__init__(*args, **kwargs)
        self.velocities = tf.zeros(shape=(2, self.num_latent_gps), dtype=default_float()) + \
                          tf.random.normal((2, self.num_latent_gps), mean=0.0, stddev=1e-5, dtype=default_float())
        self.query_states = self.likelihood.joint_sigmoid.inverse(
            tf.constant(query_states, dtype=default_float(), shape=(2, self.num_latent_gps)))
        self.optimizer = self.initialize_optimizer(learning_rate)
        self.num_samples = num_samples
        self.num_bases = num_bases
        self.num_inducing = num_inducing
        self.prior = prior
        self.planner_parameters = parameters

    @classmethod
    def initialize(cls,
                   num_data=None,
                   num_output_dims=None,
                   sigma_obs=0.05,
                   rs=List[float],
                   alpha=1,
                   learning_rate: float = 0.1,
                   num_inducing: int = 14,
                   num_samples: int = 16,
                   num_bases: int = 1024,
                   num_spheres=None,
                   lengthscale: float = 0.35,
                   offset: List = None,
                   query_states: List[np.array] = None,
                   joint_constraints: List = None,
                   velocity_constraints: List = None,
                   sdf=None,
                   robot=None,
                   kernels: List[Kernel] = None,
                   num_latent_gps: int = None,
                   parameters: dict = {},
                   **kwargs):

        if num_latent_gps is None:
            num_latent_gps = num_output_dims

        if num_spheres is None:
            print(
                "Number of spheres has not been set. The simulator will now exit...")
            sys.exit()

        if offset is None:
            offset = [0, 0, 0]

        assert num_output_dims == num_latent_gps
        assert query_states is not None
        assert joint_constraints is not None

        Z = initialize_Z(num_latent_gps, num_inducing)
        Sigma_obs = tf.constant(sigma_obs, shape=(1, num_spheres), dtype=default_float())

        if kernels is None:
            kernels = []
            for _ in range(num_latent_gps):
                kern = SquaredExponential(lengthscales=lengthscale)
                kernels.append(kern)

        # kernel = SharedIndependent(SquaredExponential(lengthscales=1), output_dim=num_latent_gps)
        kernel = SeparateIndependent(kernels)

        # Original inducing points
        _Z = VariableInducingPoints(Z=Z, dof=num_output_dims)

        sampler = Sampler(robot, parameters)
        likelihood = VariationalMonteCarloLikelihood(Sigma_obs, sampler, sdf, alpha, rs, offset, joint_constraints,
                                                     velocity_constraints)

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
                   **kwargs)

    @property
    def q_mu(self):
        return tf.concat([self.velocities, self.query_states, self._q_mu], axis=0)

    @property
    def q_sqrt(self):
        return tf.pad(self._q_sqrt, [[0, 0], [4, 0], [4, 0]]) + \
               1e-6 * tf.pad(tf.eye(4, dtype=default_float()), [[0, self.num_inducing], [0, self.num_inducing]])[
                   None, ...]

    def _init_variational_parameters(
            self,
            num_inducing: int,
            q_mu: Optional[AnyNDArray],
            q_sqrt: Optional[AnyNDArray],
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
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt:
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag:
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self._q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        np_q_sqrt: AnyNDArray = np.array(
            [
                np.eye(num_inducing, dtype=default_float())
                for _ in range(self.num_latent_gps)
            ]
        )
        self._q_sqrt = Parameter(np_q_sqrt, transform=triangular())  # [P, M, M]

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
            num_samples (int): number of samples to draw
            num_bases (int): num bases for VFF prior
        Returns:
            ELBO (float): ELBO for the current posterior
        """

        with self.temporary_paths(num_samples=self.num_samples, num_bases=self.num_bases):
            f = self.predict_f_samples(data)  # S x N x D
        g = self.likelihood.joint_sigmoid(f)

        kl = self.prior_kl()

        likelihood_obs = tf.reduce_mean(self.likelihood._log_prob(g), axis=0)  # log_prob produces S x N
        # tf.print(tf.reduce_sum(likelihood_obs), kl, summarize=-1)
        return self.likelihood.alpha * tf.reduce_sum(likelihood_obs) - kl

    @timing
    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        objective = self.elbo(data)
        if self.prior is not None:
            objective += self.prior(self)
        return objective

    def sample_from_posterior(self, X):
        mu, sigma2 = self.predict_f(X, full_cov=True)

        mu = self.likelihood.joint_sigmoid(mu)
        return mu

    def initialize_optimizer(self, learning_rate):
        return tf.optimizers.Adam(learning_rate=learning_rate)
