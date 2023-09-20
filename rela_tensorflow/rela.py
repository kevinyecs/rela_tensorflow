import tensorflow as tf

from keras.layers import Layer
from keras import Model, Sequential
from keras.layers import (
    Dropout,
    Dense,
    LayerNormalization,
    Embedding
)

'''
Sparse Attention with Linear Units 

Tensorflow implementation of Rectified Linear Attention

Paper (https://arxiv.org/pdf/2104.07012.pdf)
Paper implementation of ReLA (https://github.com/bzhangGo/zero/blob/master/modules/rela.py)
Pytorch implementation of ReLA (https://github.com/lucidrains/rela-transformer)
'''

class GatedRMSNorm(Layer):
    '''
    Gated Root Mean Square Layer Normalization (GatedRMSNorm)
    https://arxiv.org/pdf/2104.07012.pdf

    They used RMSNorm instead of the classic LayerNorm because it avoids the
    re-centering constraint, being more flexible and computationally simpler.

    References:
    https://arxiv.org/pdf/1910.07467.pdf (RMSNorm)
    https://github.com/lucidrains/rela-transformer (logic)
    https://github.com/bzhangGo/rmsnorm (logic)
    https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/rms_norm.py (logic)
    '''

    def __init__(self,
                 *,
                 eps : float = 1e-08,
                 use_bias : bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.use_bias = use_bias

    def build(self, x_shape):
        d = x_shape[-1]

        ## gain parameter (g)
        self.scale = self.add_weight(
            name = 'scale',
            shape = (1, d),
            initializer = tf.initializers.glorot_uniform())

        self.w = self.add_weight(
            name = 'w',
            shape = (1, d))

        self.offset = self.add_weight(
            name = 'offset',
            shape = (1, d) if self.use_bias else (1,),
            initializer = tf.zeros_initializer())

        self.built = True

    def call(self, x):
        ms = tf.reduce_mean(tf.math.square(x), axis = -1, keepdims = True)
        rms = self.scale * x * tf.math.rsqrt(ms + self.eps) + self.offset
        return tf.nn.sigmoid(x * self.w) * rms
    
class ReLA(Layer):
    '''
    Rectified Linear Attention (ReLA)
    https://arxiv.org/pdf/2104.07012.pdf



    References:
    https://arxiv.org/pdf/1907.01470.pdf (memory)
    '''
    
    def __init__(self,
                 causal : bool = True,
                 qkv_dim : int = 64,
                 n_heads : int = 8,
                 relu_squared : bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.causal = causal
        self.qkv_dim = qkv_dim
        self.n_heads = n_heads
        self.relu_squared = relu_squared

    def build(self, x_shape):
        d = x_shape[-1]

        self.norm = LayerNormalization()
        self.to_qkv = Dense(self.qkv_dim * 3, use_bias = False)
        self.norm_v = GatedRMSNorm()
        self.to_out = Dense(d)

        self.built = True

    def call(self, x):
        b, n, d = x.shape

        x = self.norm(x)

        qkv = tf.reshape(self.to_qkv(x), [3, b, self.n_heads, n, self.qkv_dim // self.n_heads])
        q, k, v = tf.unstack(qkv, axis = 1)

        a = tf.einsum('...nd, ...md -> ...nm', q, k) / n
        a = tf.nn.relu(a)

        if self.relu_squared:
            a = tf.math.square(a)

        if self.causal:
            mask = tf.cast(tf.linalg.band_part(tf.ones([n, n]), -1, 0), tf.bool)
            a = tf.where(mask, a, 0.0)

        out = tf.einsum('...hnm, ...hmd -> ...hnd', a, v)
        out = self.norm_v(out)

        x = tf.reshape(out, [b, n, self.qkv_dim])
        return self.to_out(x)

class ReLATransformer(Model):
    def __init__(self,
                 *,
                 emb_dim : int,
                 n_tokens : int,
                 max_seq_len : int,
                 depth : int = 4,
                 causal : bool = True,
                 n_heads : int = 8,
                 qkv_dim : int = 64,
                 expansion_factor : int = 4,
                 relu_squared : bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.depth = depth

        self.token_emb = Embedding(n_tokens, emb_dim, name = 'embeddings')
        self.pos_emb = Embedding(max_seq_len, emb_dim, name = 'embeddings')

        self.blocks = Sequential([
            Sequential([
                ReLA(
                    causal = causal,
                    qkv_dim = qkv_dim,
                    n_heads = n_heads,
                    relu_squared = relu_squared,
                    name = f'rela{i}'),
                Sequential([
                    LayerNormalization(),
                    Dense(emb_dim * expansion_factor, 'gelu'),
                    Dense(emb_dim)
                ], name = f'ffn{i}')
            ]) for i in range(depth)], name = 'blocks')

        self.to_logits = Sequential([
            LayerNormalization(),
            Dense(n_tokens)
        ], name = 'logits')

    def call(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(tf.range(x.shape[-2]))[tf.newaxis, ...] + x
        x = self.blocks(x)
        return self.to_logits(x)
