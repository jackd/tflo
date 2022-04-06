import typing as tp

import tensorflow as tf


def merged_dimension(dims: tp.Sequence[tp.Optional[int]]) -> tp.Optional[int]:
    shapes = [tf.TensorShape((d,)) for d in dims]
    shape = shapes[0]
    for shape in shapes[1:]:
        shape = shape.merge_with(shape)
    return shape[0]


def concatenated_dimension(dims: tp.Sequence[tp.Optional[int]]) -> tp.Optional[int]:
    return None if any(d is None for d in dims) else sum(dims)


def vstacked_shape(shapes: tp.Sequence[tf.TensorShape]) -> tf.TensorShape:
    batch_shapes = [shape[:-2].concatenate((shape[-1],)) for shape in shapes]
    batch_shape: tf.TensorShape = batch_shapes[0]
    for s in batch_shapes[1:]:
        batch_shape = batch_shape.merge_with(s)
    dim = concatenated_dimension([shape[-2] for shape in shapes])
    return tf.TensorShape((*batch_shape[:-1], dim, batch_shape[-1]))


def hstacked_shape(shapes: tp.Sequence[tf.TensorShape]) -> tf.TensorShape:
    batch_shapes = [shape[:-1] for shape in shapes]
    batch_shape: tf.TensorShape = batch_shapes[0]
    for s in batch_shapes[1:]:
        batch_shape = batch_shape.merge_with(s)
    dim = concatenated_dimension([shape[-1] for shape in shapes])
    return batch_shape.concatenate((dim,))


def vstack(tensors):
    return tf.concat(tensors, axis=-2)


def hstack(tensors):
    return tf.concat(tensors, axis=-1)


def get_random_st(
    rng: tf.random.Generator,
    shape: tp.Tuple[int, ...],
    sparsity: tp.Union[int, float] = 0.1,
) -> tf.SparseTensor:
    size = tf.reduce_prod(shape).numpy()
    i = tf.argsort(rng.uniform_full_int((size,)))
    nnz = sparsity if isinstance(sparsity, int) else int(size * sparsity)
    i = tf.sort(i[:nnz])
    indices = tf.unravel_index(i, shape)
    indices = tf.transpose(indices)
    return tf.SparseTensor(tf.cast(indices, tf.int64), rng.normal((nnz,)), shape)
