"""Multiple RHS version of `tf.linalg.experimental.conjugate_gradient`."""

import typing as tp

import tensorflow as tf


class CGState(tp.NamedTuple):
    i: tf.Tensor  # [n_rhs]
    x: tf.Tensor  # [*b, n, n_rhs]
    r: tf.Tensor  # [*b, n, n_rhs]
    p: tf.Tensor  # [*b, n, n_rhs]
    gamma: tf.Tensor  # [*b, n, n_rhs]
    converged: tf.Tensor  # [*b, n_rhs]


def multi_conjugate_gradient(
    operator: tf.linalg.LinearOperator,
    rhs: tf.Tensor,
    preconditioner: tp.Optional[tf.linalg.LinearOperator] = None,
    x: tp.Optional[tf.Tensor] = None,
    tol: float = 1e-5,
    atol: float = 1e-7,
    max_iter: int = 20,
    name: str = "conjugate_gradient",
) -> CGState:
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
      tol: Relative tolerance. Actual tolerance used is tol * norm(r0) + atol.
      atol: Absolute tolerance. Actual tolerance used is tol * norm(r0) + atol.
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
        - converged: [..., n_rhs] `bool` tensor flagging unconverged solutions.
    """
    if not (operator.is_self_adjoint and operator.is_positive_definite):
        raise ValueError("Expected a self-adjoint, positive definite operator.")

    def dot(x, y):
        return tf.einsum("...ij,...ij->...j", tf.math.conj(x), y)

    def stopping_criterion(i, state: CGState):
        return tf.logical_and(
            i < max_iter, tf.logical_not(tf.reduce_all(state.converged))
        )

    def cg_step(i, state: CGState):  # pylint: disable=missing-docstring
        z = tf.linalg.matmul(operator, state.p)
        alpha = state.gamma / dot(state.p, z)
        tf.debugging.assert_all_finite(
            tf.where(state.converged, tf.zeros_like(alpha), alpha),
            "alpha must be finite",
        )
        x = state.x + alpha[..., None, :] * state.p
        r = state.r - alpha[..., None, :] * z
        if preconditioner is None:
            q = r
        else:
            q = preconditioner.matmul(r)
        gamma = dot(r, q)
        beta = gamma / state.gamma
        tf.debugging.assert_all_finite(
            tf.where(state.converged, tf.zeros_like(beta), beta), "beta must be finite"
        )
        p = q + beta[..., None, :] * state.p

        converged = tf.linalg.norm(r, axis=-2) <= tol
        # only update those that haven't converged.
        old_converged = state.converged  # [..., n_rhs]
        old_converged_ = tf.expand_dims(old_converged, axis=-2)  # [..., 1, n_rhs]
        all_converged = tf.reduce_all(
            old_converged, axis=range(x.shape.ndims - 2)
        )  # [n_rhs]

        return i + 1, CGState(
            tf.where(all_converged, state.i, state.i + 1),
            tf.where(old_converged_, state.x, x),
            tf.where(old_converged_, state.r, r),
            tf.where(old_converged_, state.p, p),
            tf.where(old_converged, state.gamma, gamma),
            tf.where(old_converged, old_converged, converged),
        )

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
        tol = tf.linalg.norm(r0, axis=-2) * tol + atol
        tf.debugging.assert_positive(tol)

        if x is None:
            x = tf.zeros(broadcast_rhs_shape, dtype=rhs.dtype.base_dtype)
        else:
            r0 = rhs - tf.linalg.matmul(operator, x)
        if preconditioner is None:
            p0 = r0
        else:
            p0 = tf.linalg.matmul(preconditioner, r0)
        state = CGState(
            i=tf.zeros(tf.shape(x)[-1], dtype=tf.int32),
            x=x,
            r=r0,
            p=p0,
            gamma=dot(r0, p0),
            converged=tf.linalg.norm(r0, axis=-2) <= tol,
        )
        _, state = tf.while_loop(stopping_criterion, cg_step, (0, state))
        return state


class SingleCGState(tp.NamedTuple):
    i: tf.Tensor  # [n_rhs]
    x: tf.Tensor  # [*b, n, n_rhs]
    r: tf.Tensor  # [*b, n, n_rhs]
    p: tf.Tensor  # [*b, n, n_rhs]
    gamma: tf.Tensor  # [*b, n, n_rhs]
    converged: tf.Tensor  # [*b, n_rhs]


def single_conjugate_gradient(
    operator: tf.linalg.LinearOperator,
    rhs: tf.Tensor,
    preconditioner: tp.Optional[tf.linalg.LinearOperator] = None,
    x: tp.Optional[tf.Tensor] = None,
    tol: float = 1e-5,
    atol: float = 1e-7,
    max_iter: int = 20,
    name: str = "conjugate_gradient",
):
    """Conjugate gradient solver.

    Solves a linear system of equations `A*x = rhs` for self-adjoint, positive
    definite matrix `A` and right-hand side vector `rhs`, using an iterative,
    matrix-free algorithm where the action of the matrix A is represented by
    `operator`. The iteration terminates when either the number of iterations
    exceeds `max_iter` or when the residual norm has been reduced to `tol`
    times its initial value, i.e. \\(||rhs - A x_k|| <= tol ||rhs||\\).

    This is identical to `tf.linalg.experimental.conjugate_gradient` except
    it uses `tol * norm(rhs) + atol` rather than `tol * norm(rhs)`.

    Args:
      operator: A `LinearOperator` that is self-adjoint and positive definite.
      rhs: A possibly batched vector of shape `[..., N]` containing the right-hand
        size vector.
      preconditioner: A `LinearOperator` that approximates the inverse of `A`.
        An efficient preconditioner could dramatically improve the rate of
        convergence. If `preconditioner` represents matrix `M`(`M` approximates
        `A^{-1}`), the algorithm uses `preconditioner.apply(x)` to estimate
        `A^{-1}x`. For this to be useful, the cost of applying `M` should be
        much lower than computing `A^{-1}` directly.
      x: A possibly batched vector of shape `[..., N]` containing the initial
        guess for the solution.
      tol: Relative tolerance. Convergence occurs at
          `norm(r0) <= tol * norm(rhs) + atol`.
      atol: Absolute tolerance. Use `atol == 0` to get same performance as
           `tf.linalg.experimental.conjugate_gradient`.
      max_iter: An integer giving the maximum number of iterations.
      name: A name scope for the operation.

    Returns:
      output: A namedtuple representing the final state with fields:
        - i: A scalar `int32` `Tensor`. Number of iterations executed.
        - x: A rank-1 `Tensor` of shape `[..., N]` containing the computed
            solution.
        - r: A rank-1 `Tensor` of shape `[.., M]` containing the residual vector.
        - p: A rank-1 `Tensor` of shape `[..., N]`. `A`-conjugate basis vector.
        - gamma: \\(r \dot M \dot r\\), equivalent to  \\(||r||_2^2\\) when
          `preconditioner=None`.
    """
    if not (operator.is_self_adjoint and operator.is_positive_definite):
        raise ValueError("Expected a self-adjoint, positive definite operator.")

    def stopping_criterion(i, state):
        return tf.logical_and(
            i < max_iter, tf.reduce_any(tf.logical_not(state.converged))
        )

    def dot(x, y):
        return tf.squeeze(tf.linalg.matvec(x[..., None], y, adjoint_a=True), axis=-1)

    def cg_step(i, state):  # pylint: disable=missing-docstring
        z = tf.linalg.matvec(operator, state.p)
        alpha = state.gamma / dot(state.p, z)
        x = state.x + alpha[..., None] * state.p
        r = state.r - alpha[..., None] * z
        if preconditioner is None:
            q = r
        else:
            q = preconditioner.matvec(r)
        gamma = dot(r, q)
        beta = gamma / state.gamma
        p = q + beta[..., None] * state.p
        converged = tf.linalg.norm(r, axis=-1) <= tol
        return i + 1, CGState(i + 1, x, r, p, gamma, converged)

    # We now broadcast initial shapes so that we have fixed shapes per iteration.

    with tf.name_scope(name):
        broadcast_shape = tf.broadcast_dynamic_shape(
            tf.shape(rhs)[:-1], operator.batch_shape_tensor()
        )
        if preconditioner is not None:
            broadcast_shape = tf.broadcast_dynamic_shape(
                broadcast_shape, preconditioner.batch_shape_tensor()
            )
        broadcast_rhs_shape = tf.concat([broadcast_shape, [tf.shape(rhs)[-1]]], axis=-1)
        r0 = tf.broadcast_to(rhs, broadcast_rhs_shape)
        tol = tol * tf.linalg.norm(r0, axis=-1) + atol

        if x is None:
            x = tf.zeros(broadcast_rhs_shape, dtype=rhs.dtype.base_dtype)
        else:
            r0 = rhs - tf.linalg.matvec(operator, x)
        if preconditioner is None:
            p0 = r0
        else:
            p0 = tf.linalg.matvec(preconditioner, r0)
        gamma0 = dot(r0, p0)
        i = tf.zeros((), dtype=tf.int32)
        state = CGState(
            i=i,
            x=x,
            r=r0,
            p=p0,
            gamma=gamma0,
            converged=tf.linalg.norm(r0, axis=-1) <= tol,
        )
        _, state = tf.while_loop(stopping_criterion, cg_step, [i, state])
        return CGState(
            state.i,
            x=state.x,
            r=state.r,
            p=state.p,
            gamma=state.gamma,
            converged=state.converged,
        )
