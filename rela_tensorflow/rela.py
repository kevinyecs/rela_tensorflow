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
    """
    Gated Root Mean Square Layer Normalization (GatedRMSNorm)
    https://arxiv.org/pdf/2104.07012.pdf

    They used RMSNorm instead of the classic LayerNorm because it avoids the
    re-centering constraint, being more flexible and computationally simpler.

    References:
    https://arxiv.org/pdf/1910.07467.pdf (RMSNorm)
    https://github.com/lucidrains/rela-transformer (logic)
    https://github.com/bzhangGo/rmsnorm (logic)
    https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/rms_norm.py (logic)
    """

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
        scale_init = tf.math.sqrt(3 / d)
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
    """
    Rectified Linear Attention (ReLA)
    https://arxiv.org/pdf/2104.07012.pdf



    References:
    https://arxiv.org/pdf/1907.01470.pdf (memory)
    """
    def __init__(self,
                 causal : bool = True,
                 qkv_dim : int = 64,
                 n_heads : int = 8,
                 relu_squared : bool = True,
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
        q, k, v = tf.map_fn(
            fn = lambda x : tf.reshape(x, [b, self.n_heads, n, self.qkv_dim // self.n_heads]),
            elems = qkv)

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
class PositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.

  Example:
  ```python
  position_embedding = PositionEmbedding(max_length=100)
  inputs = tf.keras.Input((100, 32), dtype=tf.float32)
  outputs = position_embedding(inputs)
  ```


  Args:
    max_length: The maximum size of the dynamic sequence.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    seq_axis: The axis of the input tensor where we add the embeddings.

  Reference: This layer creates a positional embedding as described in
  [BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding](https://arxiv.org/abs/1810.04805).
  """

  def __init__(self,
               max_length,
               initializer="glorot_uniform",
               seq_axis=1,
               **kwargs):

    super().__init__(**kwargs)
    if max_length is None:
      raise ValueError(
          "`max_length` must be an Integer, not `None`."
      )
    self._max_length = max_length
    self._initializer = tf.keras.initializers.get(initializer)
    self._seq_axis = seq_axis

  def build(self, input_shape):
    dimension_list = input_shape
    width = dimension_list[-1]
    weight_sequence_length = self._max_length

    self._position_embeddings = self.add_weight(
        "embeddings",
        shape=[weight_sequence_length, width],
        initializer=self._initializer)

    super().build(input_shape)

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    actual_seq_len = input_shape[self._seq_axis]
    position_embeddings = self._position_embeddings[:actual_seq_len, :]
    new_shape = [1 for _ in inputs.get_shape().as_list()]
    new_shape[self._seq_axis] = actual_seq_len
    new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
    position_embeddings = tf.reshape(position_embeddings, new_shape)
    return tf.broadcast_to(position_embeddings, input_shape)
  
def FeedForward(dim, mult=4):
    ff_layer = Sequential()
    ff_layer.add(LayerNormalization())
    ff_layer.add(Dense(dim * mult, activation='gelu'))  # Add GELU activation
    ff_layer.add(Dense(dim))

    return ff_layer


class ReLATransformer(Layer):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        causal=True,
        heads=8,
        dim_head=64,
        num_memory_kv=0,
        no_ff=False,
        ff_mult=4,
        relu_squared=False
    ):

        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = Embedding(num_tokens, dim)
        self.pos_emb = PositionEmbedding(max_seq_len)

        layers = []
        for i in range(depth):
            layers.append(ReLA(qkv_dim=dim, relu_squared=relu_squared, n_heads=heads, causal=causal))
            if not no_ff:
                layers.append(FeedForward(dim=dim, mult=ff_mult))

        self.layers = Sequential(layers, name='layers')

        self.to_logits = Sequential([
            LayerNormalization(),
        Dense(num_tokens)  # Define the number of output tokens (n_tokens) here
        ], name='logits')

    def call(self, x, mask=None):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)
        x = self.pos_emb(x) + x
        x = self.layers(x)
        return self.to_logits(x)
