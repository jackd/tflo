import typing as tp

import tensorflow as tf


class LinearOperatorSparseMatrix(tf.linalg.LinearOperator):
    def __init__(
        self,
        matrix: tf.SparseTensor,
        is_non_singular: tp.Optional[bool] = None,
        is_self_adjoint: tp.Optional[bool] = None,
        is_positive_definite: tp.Optional[bool] = None,
        is_square: tp.Optional[bool] = None,
        name="LinearOperatorSparseMatrix",
    ):
        r"""Initialize a `LinearOperatorSparseMatrix`.

        Args:
          matrix:  Shape `[M, N]`.
          is_non_singular:  Expect that this operator is non-singular.
          is_self_adjoint:  Expect that this operator is equal to its hermitian
            transpose.
          is_positive_definite:  Expect that this operator is positive definite,
            meaning the quadratic form `x^H A x` has positive real part for all
            nonzero `x`.  Note that we do not require the operator to be
            self-adjoint to be positive-definite.  See:
            https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
          is_square:  Expect that this operator acts like square [batch] matrices.
          name: A name for this `LinearOperator`.

        Raises:
          NotImplementedError: matrix shape is more than rank 2.
        """
        if matrix.shape.ndims > 2:
            raise NotImplementedError("Batched sparse tensors not currently supported")
        parameters = dict(
            matrix=matrix,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
        )
        self._matrix = matrix
        assert isinstance(matrix, tf.SparseTensor)
        super().__init__(
            dtype=self._matrix.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            parameters=parameters,
            name=name,
        )

    @property
    def matrix(self) -> tf.SparseTensor:
        return self._matrix

    def _shape(self):
        return self._matrix.shape

    def _shape_tensor(self):
        return self._matrix.dense_shape

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        return tf.sparse.sparse_dense_matmul(
            self._matrix, x, adjoint_a=adjoint, adjoint_b=adjoint_arg
        )

    def _to_dense(self):
        return tf.sparse.to_dense(self._matrix)

    @property
    def _composite_tensor_fields(self):
        return ("matrix",)
