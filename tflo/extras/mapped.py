import typing as tp

import tensorflow as tf

from tflo.extras import delegate


class LinearOperatorMapped(delegate.LinearOperatorDelegate):
    """
    `LinearOperatorDelegate` that uses `tf.map_fn` to vectorize operations.

    Operations defined in terms of `tf.map_fn`:
        - _matmul (wrapping _matvec)
        - _solve (wrapping _solvevec)
    """

    def __init__(
        self,
        operator: tf.linalg.LinearOperator,
        parallel_iterations: tp.Optional[int] = None,
        name: str = "LinearOperatorProg",
    ):
        self._parallel_iterations = parallel_iterations
        super().__init__(
            operator=operator,
            name=name,
            parameters=dict(parallel_iterations=parallel_iterations),
        )

    @tf.function
    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        if adjoint_arg:
            x = tf.math.conj(x)
        else:
            x = tf.transpose(x)

        x = tf.map_fn(
            lambda xi: self._matvec(xi, adjoint=adjoint),
            x,
            parallel_iterations=self._parallel_iterations,
        )
        return tf.transpose(x)

    @tf.function
    def _solve(self, rhs, adjoint: bool = False, adjoint_arg: bool = False):
        if adjoint_arg:
            rhs = tf.math.conj(rhs)
        else:
            rhs = tf.transpose(rhs)

        sol = tf.map_fn(
            lambda r: self._solvevec(r, adjoint=adjoint),
            rhs,
            num_parallel_iterations=self._num_parallel_iterations,
        )
        return tf.transpose(sol)
