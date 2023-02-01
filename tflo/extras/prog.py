import tensorflow as tf
import tqdm

from tflo.extras import delegate


class LinearOperatorProg(delegate.LinearOperatorDelegate):
    """`LinearOperatorDelegate` that shows a progress bar during matmul/solve."""

    def __init__(
        self, operator: tf.linalg.LinearOperator, name: str = "LinearOperatorProg"
    ):
        super().__init__(operator=operator, name=name)

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        # if tf.executing_eagerly():
        #     vecs = tf.unstack(x, axis=-2 if adjoint_arg else -1)
        #     out = [self._matvec(v, adjoint=adjoint) for v in tqdm.tqdm(vecs)]
        #     return tf.stack(out, axis=-1)
        # return super()._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
        vecs = tf.unstack(x, axis=-2 if adjoint_arg else -1)
        if tf.executing_eagerly():
            vecs = tqdm.tqdm(vecs)

        out = [self._matvec(v, adjoint=adjoint) for v in vecs]
        return tf.stack(out, axis=-1)

    def _solve(self, rhs, adjoint: bool = False, adjoint_arg: bool = False):
        # if tf.executing_eagerly():
        #     vecs = tf.unstack(rhs, axis=-2 if adjoint_arg else -1)
        #     out = [self._solvevec(v, adjoint=adjoint) for v in tqdm.tqdm(vecs)]
        #     return tf.stack(out, axis=-1)
        # return super()._solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
        vecs = tf.unstack(rhs, axis=-2 if adjoint_arg else -1)
        if tf.executing_eagerly():
            vecs = tqdm.tqdm(vecs)
        out = [self._solvevec(v, adjoint=adjoint) for v in vecs]
        return tf.stack(out, axis=-1)
