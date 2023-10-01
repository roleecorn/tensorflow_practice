import tensorflow as tf
# import keras
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras

class GCN_layer(keras.layers.Layer):
    def __init__(self, fea_dim: int, out_dim: int, activation=None):
        super().__init__()
        self.fea_dim = fea_dim
        self.out_dim = out_dim
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.wei = self.add_weight(name='wei', 
                                   shape=[self.fea_dim, self.out_dim],
                                   initializer='glorot_uniform')  # 使用Glorot均匀初始化

    def call(self, inputs):
        features, support = inputs
        features = tf.cast(features, dtype=tf.float32)
        support = tf.cast(support, dtype=tf.float32)
        H_t = tf.matmul(support, features)
        output = tf.matmul(H_t, self.wei)

        if self.activation is not None:
            output = self.activation(output)
        
        return output
