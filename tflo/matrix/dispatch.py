import tensorflow as tf

from tflo.matrix import core, extras

Matrix = core.Matrix


def dispatch_for_matrix_usage(api, *signatures):
    @tf.experimental.dispatch_for_api(api, *signatures)
    def fn(*args, **kwargs):
        args, kwargs = tf.nest.map_structure(
            lambda m: m.to_operator() if isinstance(m, Matrix) else m, (args, kwargs)
        )
        out = api(*args, **kwargs)
        return core.from_operator(out)

    return fn


dispatch_for_matrix_usage(tf.linalg.adjoint, {"matrix": Matrix})
dispatch_for_matrix_usage(tf.linalg.cholesky, {"input": Matrix})
dispatch_for_matrix_usage(tf.linalg.diag_part, {"input": Matrix})
dispatch_for_matrix_usage(tf.linalg.logdet, {"matrix": Matrix})
dispatch_for_matrix_usage(tf.linalg.matmul, {"a": Matrix}, {"b": Matrix})
dispatch_for_matrix_usage(tf.linalg.matvec, {"a": Matrix})
dispatch_for_matrix_usage(tf.linalg.solve, {"matrix": Matrix})
dispatch_for_matrix_usage(tf.linalg.trace, {"x": Matrix})


@tf.experimental.dispatch_for_api(tf.math.add, {"x": Matrix, "y": Matrix})
def _add(x: Matrix, y: Matrix, name=None):
    x_ops = [x.operators if isinstance(x, extras.SumMatrix) else x]
    y_ops = [y.operators if isinstance(y, extras.SumMatrix) else y]
    return extras.SumMatrix(x_ops + y_ops, name=name or "add")


@tf.experimental.dispatch_for_api(tf.math.negative, {"x": Matrix})
def _negative(x: Matrix, name=None):
    return extras.NegativeMatrix(x, name=name or "negative")


@tf.experimental.dispatch_for_api(tf.math.subtract, {"x": Matrix, "y": Matrix})
def _subtract(x: Matrix, y: Matrix, name=None):
    with tf.name_scope(name or "subtract"):
        return tf.math.add(x, -y)
