"""Multiple RHS version of `tf.linalg.experimental.conjugate_gradient`."""

import collections
import typing as tp

import tensorflow as tf


def multi_conjugate_gradient(
    operator: tf.linalg.LinearOperator,
    rhs: tf.Tensor,
    preconditioner: tp.Optional[tf.linalg.LinearOperator] = None,
    x: tp.Optional[tf.Tensor] = None,
    tol: float = 1e-5,
    max_iter: int = 20,
    name: str = "conjugate_gradient",
):
    r"""Conjugate gradient solver.

    Solves a linear system of equations `A*x = rhs` for self-adjoint, positive
    definite matrix `A` and right-hand side matrix `rhs`, using an iterative,
    matrix-free algorithm where the action of the matrix A is represented by
    `operator`. The iteration terminates when either the number of iterations
    exceeds `max_iter` or when the residual norm has been reduced to `tol`
    times its initial value, i.e. \\(||rhs - A x_k|| <= tol ||rhs||\\).

    This is equivalent to mapping
    `lambda rhs_vec: tf.linalg.experimental.conjugate_gradient(operator, rhs_vec, ...)`
    over the final dimension of `rhs`.

    Args:
      operator: A `LinearOperator` that is self-adjoint and positive definite.
      rhs: A possibly batched vector of shape `[..., N, n_rhs]` containing the
        right-hand size matrix.
      preconditioner: A `LinearOperator` that approximates the inverse of `A`.
        An efficient preconditioner could dramatically improve the rate of
        convergence. If `preconditioner` represents matrix `M`(`M` approximates
        `A^{-1}`), the algorithm uses `preconditioner.apply(x)` to estimate
        `A^{-1}x`. For this to be useful, the cost of applying `M` should be
        much lower than computing `A^{-1}` directly.
      x: A possibly batched vector of shape `[..., N]` containing the initial
        guess for the solution.
      tol: A float scalar convergence tolerance.
      max_iter: An integer giving the maximum number of iterations.
      name: A name scope for the operation.

    Returns:
      output: A namedtuple representing the final state with fields:
        - i: A scalar `int32` `Tensor`. Number of iterations executed.
        - x: A rank-2 `Tensor` of shape `[..., N, n_rhs]` containing the computed
            solution.
        - r: A rank-2 `Tensor` of shape `[.., M, n_rhs]` containing the residual vector.
        - p: A rank-2 `Tensor` of shape `[..., N, n_rhs]`. `A`-conjugate basis vector.
        - gamma: \\(r \dot M \dot r\\), equivalent to  \\(||r||_2^2\\) when
          `preconditioner=None`.
        - unconverged: [..., n_rhs] `bool` tensor flagging unconverged solutions.
    """
    if not (operator.is_self_adjoint and operator.is_positive_definite):
        raise ValueError("Expected a self-adjoint, positive definite operator.")

    cg_state = collections.namedtuple(
        "CGState", ["i", "x", "r", "p", "gamma", "unconverged"]
    )

    def stopping_criterion(i, state):
        return tf.logical_and(i < max_iter, tf.reduce_any(state.unconverged))

    def dot(x, y):
        return tf.reduce_sum(x * y, axis=-2)

    def cg_step(i, state):  # pylint: disable=missing-docstring
        z = tf.linalg.matmul(operator, state.p)
        alpha = state.gamma / dot(state.p, z)
        alpha = tf.expand_dims(alpha, axis=-2)
        x = state.x + alpha * state.p
        r = state.r - alpha * z
        if preconditioner is None:
            q = r
        else:
            q = preconditioner.matmul(r)
        gamma = dot(r, q)
        beta = gamma / state.gamma
        p = q + beta[..., None, :] * state.p
        unconverged = tf.linalg.norm(state.r, axis=-2) > tol
        gamma = tf.where(unconverged, gamma, state.gamma)
        unconverged_ = tf.expand_dims(unconverged, axis=-2)
        x = tf.where(unconverged_, x, state.x)
        r = tf.where(unconverged_, r, state.r)
        p = tf.where(unconverged_, p, state.p)
        i = i + 1
        return i, cg_state(i, x, r, p, gamma, unconverged)

    # We now broadcast initial shapes so that we have fixed shapes per iteration.

    with tf.name_scope(name):
        broadcast_shape = tf.broadcast_dynamic_shape(
            tf.shape(rhs)[:-2], operator.batch_shape_tensor()
        )
        if preconditioner is not None:
            broadcast_shape = tf.broadcast_dynamic_shape(
                broadcast_shape, preconditioner.batch_shape_tensor()
            )
        broadcast_rhs_shape = tf.concat([broadcast_shape, tf.shape(rhs)[-2:]], axis=-1)
        r0 = tf.broadcast_to(rhs, broadcast_rhs_shape)
        tol *= tf.linalg.norm(r0, axis=-2)

        if x is None:
            x = tf.zeros(broadcast_rhs_shape, dtype=rhs.dtype.base_dtype)
        else:
            r0 = rhs - tf.linalg.matmul(operator, x)
        if preconditioner is None:
            p0 = r0
        else:
            p0 = tf.linalg.matmul(preconditioner, r0)
        gamma0 = dot(r0, p0)
        i = tf.zeros((), dtype=tf.int32)
        unconverged = tf.ones_like(tol, dtype=tf.bool)
        state = cg_state(i=i, x=x, r=r0, p=p0, gamma=gamma0, unconverged=unconverged)
        _, state = tf.while_loop(stopping_criterion, cg_step, [i, state])
        return cg_state(
            state.i,
            x=state.x,
            r=state.r,
            p=state.p,
            gamma=state.gamma,
            unconverged=state.unconverged,
        )
