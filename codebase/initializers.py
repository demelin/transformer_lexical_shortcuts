import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer


class OrthoWeight(Initializer):
    def __init__(self, dtype=tf.float32):
        self.dtype = tf.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        random_sample = tf.random_normal(shape=[shape[0], shape[0]], dtype=dtype)
        _, initial_weights, _ = tf.svd(random_sample, compute_uv=True)
        return initial_weights

    def get_config(self):
        return {'dtype': self.dtype.name}


class NormWeight(Initializer):
    def __init__(self, scale=0.01, ortho=True, dtype=tf.float32):
        self.scale = scale
        self.ortho = ortho
        self.dtype = tf.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        n_in = shape[0]
        n_out = shape[1]

        if n_out is None:
            n_out = n_in
        if n_out == n_in and self.ortho:
            ortho_initializer = OrthoWeight()
            initial_weights = ortho_initializer([n_in])
        else:
            initial_weights = self.scale * tf.random_normal(shape=[n_in, n_out], dtype=dtype)
        return initial_weights

    def get_config(self):
        return {'scale': self.scale,
                'ortho': self.ortho,
                'dtype': self.dtype.name}
