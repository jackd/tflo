import tensorflow as tf


class LinearOperatorCGSolver(tf.linalg.LinearOperator):
    def __init__(
        self,
        operator: tf.linalg.LinearOperator,
        tol: float = 1e-5,
        max_iter: int = 20,
        name="LinearOperatorCGSolver",
    ):
        assert operator.is_self_adjoint
        assert operator.is_positive_definite
        self._operator = operator
        self._tol = tol
        self._max_iter = max_iter
        super().__init__(
            dtype=operator.dtype,
            is_positive_definite=True,
            is_self_adjoint=True,
            name=name,
            parameters=dict(operator=operator, tol=tol, max_iter=max_iter),
        )

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False) -> tf.Tensor:
        del adjoint  # necessarily self-adjoint
        if adjoint_arg:
            axis = -2
            x = tf.math.conj(x)
        else:
            axis = -1
        xs = tf.unstack(x, axis=axis)

        return tf.stack([self._matvec(xi) for xi in xs], axis=-1)

    def _matvec(self, x, adjoint: bool = False) -> tf.Tensor:
        del adjoint  # necessarily self-adjoint
        return tf.linalg.experimental.conjugate_gradient(
            self._operator, x, tol=self.tol, max_iter=self.max_iter
        ).x

    def _shape(self) -> tf.TensorShape:
        return self._operator.shape

    @property
    def tol(self) -> float:
        return self._tol

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def _log_abs_determinant(self):
        return -self._operator._log_abs_determinant()

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        del adjoint  # necessarily self-adjoint
        return self._operator._matmul(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _solvevec(self, rhs, adjoint=False):
        del adjoint  # necessarily self-adjoint
        return self._operator._matvec(rhs)

    @property
    def _composite_tensor_fields(self):
        return ("operator",)
