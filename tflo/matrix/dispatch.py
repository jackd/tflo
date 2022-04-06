import tensorflow as tf

from tflo.matrix.core import Matrix, from_operator


def dispatch_for_matrix_usage(api, *signatures):
    @tf.experimental.dispatch_for_api(api, *signatures)
    def fn(*args, **kwargs):
        args, kwargs = tf.nest.map_structure(
            lambda m: m.to_operator() if isinstance(m, Matrix) else m, (args, kwargs)
        )
        out = api(*args, **kwargs)
        return from_operator(out)

    return fn


dispatch_for_matrix_usage(tf.linalg.adjoint, {"matrix": Matrix})
dispatch_for_matrix_usage(tf.linalg.cholesky, {"input": Matrix})
dispatch_for_matrix_usage(tf.linalg.diag_part, {"input": Matrix})
dispatch_for_matrix_usage(tf.linalg.logdet, {"matrix": Matrix})
dispatch_for_matrix_usage(tf.linalg.matmul, {"a": Matrix}, {"b": Matrix})
dispatch_for_matrix_usage(tf.linalg.matvec, {"a": Matrix})
dispatch_for_matrix_usage(tf.linalg.solve, {"matrix": Matrix})
dispatch_for_matrix_usage(tf.linalg.trace, {"x": Matrix})


# @tf.experimental.dispatch_for_api(tf.linalg.adjoint, {"matrix": Matrix})
# def adjoint(matrix: Matrix, name: tp.Optional[str] = None) -> Matrix:
#     return Matrix.from_operator(tf.linalg.adjoint(matrix.to_operator(), name=name))
