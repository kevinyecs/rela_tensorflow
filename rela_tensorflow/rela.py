import tensorflow as tf
from keras.layers import Layer

from keras.layers import (
    Dropout,
    Linear
)

'''
Sparse Attention with Linear Units 

OOP Tensorflow implementation

Paper (https://arxiv.org/pdf/2104.07012.pdf)
Tensorflow implementation of ReLA (https://github.com/bzhangGo/zero/blob/master/modules/rela.py)

'''


class RectifiedLinearAttention(Layer):
    def __init__(self, query, memory, mem_mask, hidden_size, 
                 ln=False, num_heads=1, cache=None, dropout=None, out_map=True, scope=None, **kwargs):
        super(RectifiedLinearAttention, self).__init__(**kwargs)
        self.query = query
        self.memory = memory
        self.mem_mask = mem_mask
        self.hidden_size = hidden_size
        self.ln = ln
        self.num_heads = num_heads
        self.cache = cache
        self.dropout = dropout
        self.out_map = out_map
    
    def call(self, x):
        if self.memory is None:
            # suppose self-attention from queries alone
            h = Linear(self.hidden_size * 3 , use_bias=False)(self.query)
            q, k, v = tf.split(h, 3, axis=-1)

            if self.cache is not None:
                k = tf.concat([self.cache['k'], k], axis=1)
                v = tf.concat([self.cache['v'], v], axis=1)
                cache = {'k': k, 'v': v}
            
        else:
            q = Linear(self.hidden_size, use_bias=False)(self.query)
            if self.cache is not None and ('mk' in cache and 'mv' in cache):
                k, v = cache['mk'], cache['mv']
            else:
                k = Linear(self.hidden_size, use_bias=False)(self.memory)
                v = Linear(self.hidden_size, use_bias=False)(self.memory)
                if self.cache is not None:
                    cache['mk'], cache['mv'] = k, v

        q = SplitHeadsLayer(self.num_heads)(q)
        k = SplitHeadsLayer(self.num_heads)(k)
        v = SplitHeadsLayer(self.num_heads)(v)

        q *= (self.hidden_size // self.num_heads) ** -0.5

        # q * k => attention weights
        logits = tf.matmul(q, k, transpose_b=True)

        if self.mem_mask is not None:
            zero_one_mask = tf.to_float(tf.equal(self.mem_mask, 0.0))
            logits *= zero_one_mask
        
        # Rectified Magic
        weights = tf.nn.relu(logits)

        if self.dropout is not None:
            dweights = Dropout(self.dropout)(weights)
        
        # weights * v => attention vector
        o = tf.matmul(dweights, v)
        o = CombineHeadsLayer()(o)

        #performs RMSNorm to stabilize running
        o = GatedRMSNormLayer()(o)

        if self.out_map:
            o = Linear(self.hidden_size)(o)
        
        return o


class SplitHeadsLayer(Layer):
    def __init__(self, num_heads, **kwargs):
        super(SplitHeadsLayer, self).__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        super(SplitHeadsLayer, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        n = self.num_heads
        old_shape = x.get_shape().as_list()

        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        return tf.transpose(ret, [0, 2, 1, 3])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + [self.num_heads, input_shape[-1] // self.num_heads]
            
class CombineHeadsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CombineHeadsLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CombineHeadsLayer, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        x = tf.transpose(x, [0, 2, 1, 3])
        old_shape = x.get_shape().as_list()
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        x.set_shape(new_shape)
        return x

    def compute_output_shape(self, input_shape):
        a, b = input_shape[-2:]
        return input_shape[:-2] + [a * b if a and b else None]
    

class GatedRMSNormLayer(Layer):
    def __init__(self, eps=None, **kwargs):
        super(GatedRMSNormLayer, self).__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        super(GatedRMSNormLayer, self).build(input_shape)
        layer_size = input_shape[-1]

        self.scale = self.add_weight("scale", shape=(layer_size,),
                                    initializer=tf.ones_initializer(),
                                    trainable=True)
        self.gate = self.add_weight("gate", shape=(layer_size,),
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)

    def call(self, inputs):
        x = inputs
        if self.eps is None:
            self.eps = tf.keras.backend.epsilon()

        ms = tf.reduce_mean(x ** 2, axis=-1, keepdims=True)
        gated_x = tf.nn.sigmoid(self.gate * x)

        return self.scale * x * tf.math.rsqrt(ms + self.eps) * gated_x

    def compute_output_shape(self, input_shape):
        return input_shape
    
