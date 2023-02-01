import scipy.sparse.linalg as la
import tensorflow as tf

from tflo.scipy_wrapper import to_scipy


class LinearOperatorExponential(tf.linalg.LinearOperator):
    def __init__(
        self, operator: tf.linalg.LinearOperator, name="LinearOperatorExponential"
    ):
        assert len(operator.shape) == 2, operator.shape
        assert operator.shape[0] == operator.shape[1], operator.shape
        self.operator = operator
        super().__init__(
            dtype=self.operator.dtype,
            is_self_adjoint=operator.is_self_adjoint,
            is_square=True,
            is_positive_definite=None if operator.dtype.is_complex else True,
            parameters=dict(operator=operator),
            name=name,
        )

    def _shape(self):
        return self.operator._shape()

    def _shape_tensor(self):
        return self.operator._shape_tensor()

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        if adjoint_arg:
            x = tf.math.conj(tf.transpose(x))
        op = self.operator
        if adjoint:
            op = op.adjoint()
        return la.expm_multiply(to_scipy(op), x.numpy())

    def _matvec(self, x, adjoint: bool = False):
        op = to_scipy(self.operator)
        if adjoint:
            op = op.adjoint()
        return la.expm_multiply(op, x.numpy())

    @property
    def _composite_tensor_fields(self):
        return ("operator",)

    def _adjoint(self) -> "LinearOperatorExponential":
        return LinearOperatorExponential(self.operator.adjoint())
