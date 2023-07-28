#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
from gpflow import kernels
from gpflow.base import TensorLike
from gpflow_sampling.covariances.Kfus import Kfu

from gpflow_vgpmp.covariances.Kufs import Kuf as Kuf_dispatch

from gpflow.inducing_variables.multioutput.inducing_variables import SharedIndependentInducingVariables, \
    SeparateIndependentInducingVariables
from gpflow_vgpmp.covariances.Kufs import Kuf
from gpflow_vgpmp.kernels.kernels import FirstOrderKernelDerivativeSeparateIndependent, \
    VanillaConditioningSeparateIndependent


# ==============================================
#                     Kfus
# ==============================================


@Kfu.register(SharedIndependentInducingVariables, FirstOrderKernelDerivativeSeparateIndependent, TensorLike)
def _kfu_fallback_shared_derivative(Z, kern, X, **kwargs):
    kuf = Kuf_dispatch(Z, kern, X, **kwargs)
    perm = get_kfu_axes(X, Kuf)
    len_ny = len(Z.inducing_variable.ny)
    kuf = tf.concat([-kuf[:, :len_ny], kuf[:, len_ny:]], axis=1)
    return tf.transpose(kuf, perm)


@Kfu.register((SharedIndependentInducingVariables, SeparateIndependentInducingVariables),
              VanillaConditioningSeparateIndependent, TensorLike)
def _kfu_fallback_shared_vanilla(Z, kern, X, **kwargs):
    kuf = Kuf_dispatch(Z, kern, X, **kwargs)
    # Assume features of x and z are 1-dimensional
    perm = get_kfu_axes(X, kuf)
    return tf.transpose(kuf, perm)


@Kfu.register(SharedIndependentInducingVariables,
              FirstOrderKernelDerivativeSeparateIndependent, TensorLike)
def _kfu_fallback_separate_derivative(Z, kern, X, **kwargs):
    kuf = Kuf_dispatch(Z, kern, X, **kwargs)
    # Assume features of x and z are 1-dimensional
    perm = get_kfu_axes(X, kuf)
    len_ny = len(Z.inducing_variable.ny)
    kuf = tf.concat([-kuf[:, :len_ny], kuf[:, len_ny:]], axis=1)
    return tf.transpose(kuf, perm)


@Kfu.register(SeparateIndependentInducingVariables,
              FirstOrderKernelDerivativeSeparateIndependent, TensorLike)
def _kfu_fallback_separate_derivative(Z, kern, X, **kwargs):
    kuf = Kuf_dispatch(Z, kern, X, **kwargs)
    # Assume features of x and z are 1-dimensional
    perm = get_kfu_axes(X, kuf)
    len_ny = len(Z.inducing_variable_list[0].ny)
    kuf = tf.concat([-kuf[:, :len_ny], kuf[:, len_ny:]], axis=1)
    return tf.transpose(kuf, perm)


@tf.function
def get_kfu_axes(X, kuf):
    ndims_x = X.shape.ndims - 1  # assume x lives in 1d space
    ndims_z = 1  # shared Z live in 1d space, separate Z are 2d but 1-to-1 with L
    assert ndims_x + ndims_z == kuf.shape.ndims - 1
    # Swap the batch axes of x and z
    axes = list(range(1, ndims_x + ndims_z + 1))  # keep L output-features first
    perm = [0] + axes[ndims_z: ndims_z + ndims_x] + axes[:ndims_z]
    return perm
