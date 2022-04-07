import typing as tp

import tensorflow as tf

from tflo import extras, utils
from tflo.matrix.core import Matrix, register_matrix_cls


@register_matrix_cls(extras.LinearOperatorCGSolver)
class CGSolverMatrix(Matrix):
    operator: Matrix
    tol: float = 1e-5
    max_iter: int = 20
    name: str = "CGSolverMatrix"

    @property
    def is_square(self) -> bool:
        return True

    @property
    def is_self_adjoint(self) -> bool:
        return True

    @property
    def is_positive_definite(self) -> bool:
        return True

    @property
    def is_non_singular(self) -> bool:
        return True

    class Spec:
        @property
        def shape(self) -> tf.TensorShape:
            return self.operator.shape

        @property
        def dtype(self):
            return self.operator.dtype

        @property
        def is_square(self) -> bool:
            return True

        @property
        def is_self_adjoint(self) -> bool:
            return True

        @property
        def is_positive_definite(self) -> bool:
            return True

        @property
        def is_non_singular(self) -> bool:
            return True


@register_matrix_cls(extras.LinearOperatorSparseMatrix)
class SparseMatrix(Matrix):
    matrix: tf.SparseTensor
    is_self_adjoint: tp.Optional[bool] = None
    is_non_singular: tp.Optional[bool] = None
    is_positive_definite: tp.Optional[bool] = None
    is_square: tp.Optional[bool] = None
    name: str = "SparseMatrix"

    class Spec:
        @property
        def shape(self) -> tf.TensorShape:
            return self.matrix.shape

        @property
        def dtype(self):
            return self.matrix.dtype


@register_matrix_cls(extras.LinearOperatorHStacked)
class HStackedMatrix(Matrix):
    operators: tp.Tuple[Matrix, ...]
    is_self_adjoint: tp.Optional[bool] = None
    is_non_singular: tp.Optional[bool] = None
    is_positive_definite: tp.Optional[bool] = None
    is_square: tp.Optional[bool] = None
    name: str = "HStackedMatrix"

    class Spec:
        @property
        def shape(self) -> tf.TensorShape:
            return utils.hstacked_shape([op.shape for op in self.operators])

        @property
        def dtype(self) -> tf.DType:
            return self.operators[0].dtype


@register_matrix_cls(extras.LinearOperatorVStacked)
class VStackedMatrix(Matrix):
    operators: tp.Tuple[Matrix, ...]
    is_self_adjoint: tp.Optional[bool] = None
    is_non_singular: tp.Optional[bool] = None
    is_positive_definite: tp.Optional[bool] = None
    is_square: tp.Optional[bool] = None
    name: str = "VStackedMatrix"

    class Spec:
        @property
        def shape(self) -> tf.TensorShape:
            return utils.vstacked_shape([op.shape for op in self.operators])

        @property
        def dtype(self) -> tf.DType:
            return self.operators[0].dtype


@register_matrix_cls(extras.LinearOperatorStaticPowerSeries)
class StaticPowerSeriesMatrix(Matrix):
    operator: Matrix
    coeffs: tp.Tuple[tp.Union[int, float], ...]
    is_self_adjoint: tp.Optional[bool] = None
    is_non_singular: tp.Optional[bool] = None
    is_positive_definite: tp.Optional[bool] = None
    name: str = "StaticPowerSeriesMatrix"

    @property
    def is_square(self) -> bool:
        return True

    class Spec:
        @property
        def shape(self):
            return self.operator.shape

        @property
        def dtype(self):
            return self.operator.dtype

        @property
        def is_square(self):
            return True
