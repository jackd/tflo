import functools

import tensorflow as tf
import tqdm

from tflo.extras import delegate


def _matmul(
    operator: tf.linalg.LinearOperator, x: tf.Tensor, adjoint: bool, adjoint_arg: bool
) -> tf.Tensor:
    assert tf.executing_eagerly()
    vecs = tf.unstack(x, axis=-2 if adjoint_arg else -1)
    vecs = tqdm.tqdm(vecs)
    out = [tf.linalg.matvec(operator, v, adjoint_a=adjoint) for v in vecs]
    return tf.stack(out, axis=-1)


def _solve(
    operator: tf.linalg.LinearOperator, rhs: tf.Tensor, adjoint: bool, adjoint_arg: bool
):
    vecs = tf.unstack(rhs, axis=-2 if adjoint_arg else -1)
    vecs = tqdm.tqdm(vecs)
    out = [operator.solvevec(v, adjoint=adjoint) for v in vecs]
    return tf.stack(out, axis=-1)


class LinearOperatorProg(delegate.LinearOperatorDelegate):
    """`LinearOperatorDelegate` that shows a progress bar during matmul/solve."""

    def __init__(
        self, operator: tf.linalg.LinearOperator, name: str = "LinearOperatorProg"
    ):
        super().__init__(operator=operator, name=name)

    def _compute_matmul_shape(
        self, arg_shape: tf.TensorShape, adjoint: bool, adjoint_arg: bool
    ) -> tf.TensorShape:
        batch_shape = tf.broadcast_static_shape(
            self.operator.batch_shape, arg_shape[:-2]
        )
        range_dim = self.operator.shape[-1 if adjoint else -2]
        domain_dim = arg_shape[-2 if adjoint_arg else -1]
        return tf.TensorShape((*batch_shape, range_dim, domain_dim))

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        matmul = functools.partial(
            _matmul,
            adjoint=adjoint,
            adjoint_arg=adjoint_arg,
        )
        args = (self.operator, x)

        if tf.executing_eagerly():
            return matmul(*args)

        out = tf.py_function(matmul, args, x.dtype)
        out.set_shape(self._compute_matmul_shape(x.shape, adjoint, adjoint_arg))
        return out

    def _solve(self, rhs, adjoint: bool = False, adjoint_arg: bool = False):
        solve = functools.partial(_solve, adjoint=adjoint, adjoint_arg=adjoint_arg)
        args = (self.operator, rhs)

        if tf.executing_eagerly():
            return solve(*args)
        out = tf.py_function(solve, args, rhs.dtype)
        out.set_shape(self._compute_matmul_shape(rhs.shape, adjoint, adjoint_arg))
        return out
