import abc
import typing as tp

import keras
import tensorflow as tf
from keras.layers.core.tf_op_layer import _delegate_method

_methods = (
    "batch_shape_tensor",
    "cholesky",
    "cond",
    "determinant",
    "diag_part",
    "domain_dimension_tensor",
    "eigvals",
    "inverse",
    "log_abs_determinant",
    "matmul",
    "matvec",
    "range_dimension_tensor",
    "shape_tensor",
    "solve",
    "solve",
    "tensor_rank_tensor",
    "to_dense",
    "trace",
)


def copy_docs(matrix_cls: type, operator_cls: type):
    assert isinstance(matrix_cls, type)
    assert isinstance(operator_cls, type)
    matrix_cls.__doc__ = (
        f"Matrix wrapper around {operator_cls.__name__}.\n\n" + operator_cls.__doc__
    )
    for method in _methods:
        getattr(matrix_cls, method).__doc__ = getattr(operator_cls, method)


def wrap_docs(operator_cls: type):
    def decorator(matrix_cls: type):
        copy_docs(matrix_cls, operator_cls)
        return matrix_cls

    return decorator


_matrix_cls = {}
_operator_cls = {}


def register_matrix_cls(operator_cls: type):
    assert issubclass(operator_cls, tf.linalg.LinearOperator)
    if operator_cls in _matrix_cls:
        raise TypeError(f"operator_cls {operator_cls} already registered")

    def decorator(matrix_cls: type):
        assert issubclass(matrix_cls, Matrix)
        if matrix_cls in _operator_cls:
            raise TypeError(f"matrix_cls {matrix_cls} already registered")
        _matrix_cls[operator_cls] = matrix_cls
        _operator_cls[matrix_cls] = operator_cls
        return matrix_cls

    return decorator


def wrap_and_register(operator_cls: type):
    wrap = wrap_docs(operator_cls)
    register = register_matrix_cls(operator_cls)

    def decorator(matrix_cls):
        wrap(matrix_cls)
        register(matrix_cls)
        return matrix_cls

    return decorator


def from_operator(struct):
    operators = tf.nest.map_structure(
        lambda p: isinstance(p, tf.linalg.LinearOperator),
        struct,
        expand_composites=False,
    )
    return tf.nest.map_structure(
        lambda p, is_lo: Matrix.from_operator(p) if is_lo else p,
        struct,
        operators,
        expand_composites=False,
    )

    # no idea why the below doesn't work - something to do with composites?
    # return tf.nest.map_structure(
    #     lambda p: Matrix.from_operator(p)
    #     if isinstance(operator, tf.linalg.LinearOperator)
    #     else p,
    #     operator.params,
    #     expand_composites=False,
    # )


class Matrix(tf.experimental.BatchableExtensionType, metaclass=abc.ABCMeta):
    @property
    def parameters(self) -> tp.Mapping[str, tp.Any]:
        return {a: getattr(self, a) for a in type(self).__annotations__}

    def to_operator(self) -> tf.linalg.LinearOperator:
        operator_cls = _operator_cls[type(self)]
        params = tf.nest.map_structure(
            lambda p: p.to_operator() if isinstance(p, Matrix) else p, self.parameters
        )
        return operator_cls(**params)

    @classmethod
    def from_operator(cls, operator: tf.linalg.LinearOperator) -> "Matrix":
        registered_cls = _matrix_cls[type(operator)]
        assert issubclass(registered_cls, cls)

        params = from_operator(operator.parameters)

        return registered_cls(**params)

    @property
    def shape(self):
        return self.to_operator().shape

    @property
    def batch_shape(self):
        return self.to_operator().batch_shape

    @property
    def domain_dimension(self) -> int:
        return int(self.to_operator().domain_dimension)

    @property
    def range_dimension(self) -> int:
        return int(self.to_operator().range_dimension)

    @property
    def dtype(self):
        return self.to_operator().dtype

    @property
    def tensor_rank(self):
        return self.to_operator().tensor_rank

    def assert_positive_definite(self):
        return self.to_operator().assert_positive_definite()

    def assert_self_adjoint(self):
        return self.to_operator().assert_self_adjoint()

    def batch_shape_tensor(self, name: str = "batch_shape_tensor"):
        return self.to_operator().batch_shape_tensor(name=name)

    def cholesky(self, name: str = "cholesky"):
        return self.to_operator().cholesky(name=name)

    def cond(self, name: str = "cond"):
        return self.to_operator().cond(name=name)

    def determinant(self, name: str = "det"):
        return self.to_operator().determinant(name=name)

    def diag_part(self, name: str = "diag_part"):
        return self.to_operator().diag_part(name=name)

    def domain_dimension_tensor(self, name: str = "domain_dimension_tensor"):
        return self.to_operator().domain_dimension_tensor(name=name)

    def eigvals(self, name: str = "eigvals"):
        return self.to_operator().eigvals(name=name)

    def inverse(self, name: str = "inverse"):
        return self.to_operator().inverse(name=name)

    def log_abs_determinant(self, name: str = "log_abs_det"):
        return self.to_operator().log_abs_determinant(name=name)

    def matmul(
        self,
        x: tf.Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
        name: str = "matmul",
    ):
        return self.to_operator().matmul(
            x, adjoint=adjoint, adjoint_arg=adjoint_arg, name=name
        )

    def matvec(self, x: tf.Tensor, adjoint: bool = False, name: str = "matvec"):
        return self.to_operator().matvec(x, adjoint=adjoint, name=name)

    def range_dimension_tensor(self, name: str = "range_dimension_tensor"):
        return self.to_operator().range_dimension_tensor(name=name)

    def shape_tensor(self, name: str = "shape_tensor"):
        return self.to_operator().shape_tensor(name=name)

    def solve(
        self,
        rhs: tf.Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
        name: str = "solve",
    ):
        return self.to_operator().solve(
            rhs, adjoint=adjoint, adjoint_arg=adjoint_arg, name=name
        )

    def solvevec(self, rhs: tf.Tensor, adjoint: bool = False, name: str = "solve"):
        return self.to_operator().solvevec(rhs, adjoint=adjoint, name=name)

    def tensor_rank_tensor(self, name: str = "tensor_rank_tensor"):
        return self.to_operator().tensor_rank_tensor(name=name)

    def to_dense(self, name: str = "to_dense"):
        return self.to_operator().to_dense(name=name)

    def trace(self, name: str = "trace"):
        return self.to_operator().trace(name=name)

    def adjoint(self, name: str = "adjoint") -> "Matrix":
        return Matrix.from_operator(self.to_operator().adjoint(name=name))

    def __matmul__(self, other):
        return tf.linalg.matmul(self, other)

    def __add__(self, other):
        return tf.math.add(self, other)

    def __sub__(self, other):
        return tf.math.subtract(self, other)

    def __neg__(self):
        return tf.math.negative(self)


copy_docs(Matrix, tf.linalg.LinearOperator)


class KerasMatrix(keras.engine.keras_tensor.KerasTensor):
    @property
    def batch_shape(self):
        return self.shape[:-2]

    @property
    def domain_dimension(self):
        return self.shape[-1]

    @property
    def range_dimension(self):
        return self.shape[-2]

    @property
    def tensor_rank(self) -> int:
        return self.shape.ndims


keras.engine.keras_tensor.register_keras_tensor_specialization(Matrix, KerasMatrix)
for method in _methods:
    _delegate_method(KerasMatrix, method)


@register_matrix_cls(tf.linalg.LinearOperatorAdjoint)
class AdjointMatrix(Matrix):
    operator: Matrix
    is_non_singular: tp.Optional[bool] = None
    is_self_adjoint: tp.Optional[bool] = None
    is_positive_definite: tp.Optional[bool] = None
    is_square: tp.Optional[bool] = None
    name: tp.Optional[str] = "AdjointMatrix"

    class Spec:
        @property
        def shape(self):
            shape = self.operator.shape
            return tf.TensorShape((*shape[:-2], shape[-1], shape[-2]))

        @property
        def dtype(self):
            return self.operator.dtype


@tf.experimental.dispatch_for_api(tf.cast, {"x": AdjointMatrix})
def _cast(x, dtype: tf.DType, name=None):
    return AdjointMatrix(
        tf.cast(x.operator, dtype, name=name),
        is_non_singular=x.is_non_singular,
        is_self_adjoint=x.is_self_adjoint,
        is_positive_definite=x.is_positive_definite,
        is_square=x.is_square,
    )


@register_matrix_cls(tf.linalg.LinearOperatorComposition)
class CompositionMatrix(Matrix):
    operators: tp.Tuple[Matrix, ...]
    is_self_adjoint: tp.Optional[bool] = None
    is_non_singular: tp.Optional[bool] = None
    is_positive_definite: tp.Optional[bool] = None
    is_square: tp.Optional[bool] = None
    name: tp.Optional[str] = "CompositionMatrix"

    class Spec:
        @property
        def dtype(self):
            return self.operators[0].dtype

        @property
        def shape(self):
            return tf.TensorShape(
                (*self.operators[0].shape[:-1], self.operators[-1].shape[-1])
            )


@tf.experimental.dispatch_for_api(tf.cast, {"x": CompositionMatrix})
def _cast(x, dtype: tf.DType, name=None):
    return CompositionMatrix(
        tuple(tf.cast(op, dtype, name=name) for op in x.operators),
        is_self_adjoint=x.is_self_adjoint,
        is_non_singular=x.is_non_singular,
        is_positive_definite=x.is_positive_definite,
        is_square=x.is_square,
    )


@register_matrix_cls(tf.linalg.LinearOperatorDiag)
class DiagMatrix(Matrix):
    diag: tf.Tensor
    is_self_adjoint: tp.Optional[bool] = None
    is_non_singular: tp.Optional[bool] = None
    is_positive_definite: tp.Optional[bool] = None
    is_square: tp.Optional[bool] = None
    name: str = "DiagMatrix"

    class Spec:
        @property
        def shape(self):
            shape = self.diag.shape
            n = shape[-1]
            return tf.TensorShape((*shape[:-1], n, n))

        @property
        def dtype(self):
            return self.diag.dtype


@register_matrix_cls(tf.linalg.LinearOperatorFullMatrix)
class FullMatrix(Matrix):
    matrix: tf.Tensor
    is_self_adjoint: tp.Optional[bool] = None
    is_non_singular: tp.Optional[bool] = None
    is_positive_definite: tp.Optional[bool] = None
    is_square: tp.Optional[bool] = None
    name: str = "FullMatrix"

    class Spec:
        @property
        def shape(self) -> tf.TensorShape:
            return self.matrix.shape

        @property
        def dtype(self) -> tf.DType:
            return self.matrix.dtype


@tf.experimental.dispatch_for_api(tf.cast, {"x": FullMatrix})
def _cast(x, dtype: tf.DType, name=None):
    return FullMatrix(
        tf.cast(x.matrix, dtype, name=name),
        is_self_adjoint=x.is_self_adjoint,
        is_non_singular=x.is_non_singular,
        is_positive_definite=x.is_positive_definite,
        is_square=x.is_square,
    )


@register_matrix_cls(tf.linalg.LinearOperatorScaledIdentity)
class ScaledIdentityMatrix(Matrix):
    num_rows: int
    multiplier: tf.Tensor
    is_self_adjoint: tp.Optional[bool] = True
    is_non_singular: tp.Optional[bool] = None
    is_positive_definite: tp.Optional[bool] = None
    is_square: tp.Optional[bool] = True
    assert_proper_shapes: bool = False
    name: str = "ScaledIdentityMatrix"

    class Spec:
        @property
        def shape(self):
            return tf.TensorShape(
                (*self.multiplier.shape, self.num_rows, self.num_rows)
            )

        @property
        def dtype(self):
            return self.multiplier.dtype


def composition_matrix(*args: tp.Iterable[Matrix], **kwargs) -> Matrix:
    operators = []
    for arg in args:
        if isinstance(arg, CompositionMatrix):
            operators.extend(arg.operators)
        else:
            operators.append(arg)
    if len(operators) == 1:
        assert not kwargs, kwargs
        return operators[0]
    if len(operators) == 0:
        raise ValueError("Requires at least one operator.")
    return CompositionMatrix(operators, **kwargs)
