import typing as tp

import tensorflow as tf


class LinearOperatorStaticPowerSeries(tf.linalg.LinearOperator):
    def __init__(
        self,
        operator: tf.linalg.LinearOperator,
        coeffs: tp.Iterable[tp.Union[int, float]],
        is_non_singular: tp.Optional[bool] = None,
        is_self_adjoint: tp.Optional[bool] = None,
        is_positive_definite: tp.Optional[bool] = None,
        name="LinearOperatorStaticPowerSeries",
    ):
        self._coeffs = coeffs
        self._operator = operator
        coeffs = list(coeffs)

        super().__init__(
            dtype=operator.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=True,
            parameters=dict(
                operator=operator,
                coeffs=coeffs,
                is_non_singular=is_non_singular,
                is_self_adjoint=is_self_adjoint,
                is_positive_definite=is_positive_definite,
                name=name,
            ),
            name=name,
        )

    def _shape(self):
        return self._operator.shape

    def _mul(self, mul_fn, x, adjoint_arg=False):
        if adjoint_arg:
            x = tf.linalg.adjoint(x)
        px = x  # powers of x
        acc = self._coeffs[0] * x
        for c in self._coeffs[1:]:
            px = mul_fn(px)
            acc = acc + c * px
        return acc

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        return self._mul(
            lambda x: self._operator._matmul(x, adjoint=adjoint),
            x,
            adjoint_arg=adjoint_arg,
        )

    def _matvec(self, x, adjoint: bool = False):
        return self._mul(lambda x: self._operator._matvec(x, adjoint=adjoint), x)
