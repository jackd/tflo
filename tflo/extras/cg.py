import typing as tp

import tensorflow as tf

from tflo.ops import multi_conjugate_gradient, single_conjugate_gradient


class LinearOperatorCGSolver(tf.linalg.LinearOperator):
    def __init__(
        self,
        operator: tf.linalg.LinearOperator,
        preconditioner: tp.Optional[tf.linalg.LinearOperator] = None,
        x0: tp.Optional[tf.Tensor] = None,
        tol: float = 1e-5,
        atol: float = 1e-7,
        max_iter: int = 20,
        name="LinearOperatorCGSolver",
        *,
        is_non_singular: tp.Optional[bool] = None,
        is_positive_definite: tp.Optional[bool] = None,
        is_square: tp.Optional[bool] = None,
        is_self_adjoint: tp.Optional[bool] = None,
    ):
        assert is_non_singular in (None, True)
        assert is_square in (None, True)
        assert is_self_adjoint in (None, True)
        assert is_positive_definite in (None, True)
        assert operator.is_self_adjoint
        assert operator.is_positive_definite
        self._operator = operator
        self._preconditioner = preconditioner
        self._tol = tol
        self._atol = atol
        self._max_iter = max_iter
        self._x0 = x0
        super().__init__(
            dtype=operator.dtype,
            is_positive_definite=True,
            is_self_adjoint=True,
            name=name,
            parameters=dict(
                operator=operator,
                tol=tol,
                atol=atol,
                max_iter=max_iter,
                preconditioner=self._preconditioner,
                x0=x0,
            ),
        )

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False) -> tf.Tensor:
        del adjoint  # necessarily self-adjoint
        if adjoint_arg:
            x = tf.linalg.adjoint(x)
        x0 = self._x0
        if x0 is not None and x0.shape.ndims:
            x0 = tf.expand_dims(x0, axis=-1)
            x0 = tf.tile(x0, (1,) * (x0.shape.ndims - 1) + (tf.shape(x)[-1],))
        cg = multi_conjugate_gradient(
            self._operator,
            x,
            tol=self._tol,
            atol=self._atol,
            max_iter=self._max_iter,
            preconditioner=self._preconditioner,
            x=x0,
        )
        x = cg.x
        tf.debugging.assert_all_finite(
            x, "conjugate_gradient did not return finite values"
        )
        return x

    def _matvec(self, x, adjoint: bool = False) -> tf.Tensor:
        del adjoint  # necessarily self-adjoint
        return single_conjugate_gradient(
            self._operator,
            x,
            tol=self._tol,
            atol=self._atol,
            max_iter=self.max_iter,
            x=self._x0,
        ).x

    def _shape(self) -> tf.TensorShape:
        return self._operator.shape

    @property
    def tol(self) -> float:
        return self._tol

    @property
    def atol(self) -> float:
        return self._atol

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

    def _adjoint(self):
        return self

    @property
    def _composite_tensor_fields(self):
        return ("operator", "preconditioner", "x0")
