import tensorflow as tf
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, AveragePooling1D, Dense, AveragePooling2D
from keras.datasets import imdb # dont need
from keras.models import Sequential, Model
import numpy as np
import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class TransformerBlock(Layer):
    # def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
    def __init__(self, d_model, d_k, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()

        # outputs tensor of shape (max_obs,d_model)
        # key_dim determines size of W_q, W_k, and W_v to be (d_model, key_dim)
        # in "attention is all you need" key_dim was set to d_model/num_heads
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_k)

        # the outpu needs to be d_model
        # wide and shallow network ff_dim should be larger than d_model
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(d_model)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.supports_masking = True

    def call(self, inputs, training, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

"reduces by 2"
#(batch size, number of sentinels, number of observations)
#input_dim = (#num of sentinels)
class FunnelTransformerBlock(Layer):
    def __init__(self, d_model, d_k, num_heads, ff_dim, input_dim, rate=0.1, reduce=True):
        super(FunnelTransformerBlock, self).__init__()
        #self.pool = AveragePooling1D(pool_size=2, strides=2)
        if reduce == True:
            self.first_map = Dense(input_dim//2)
        elif reduce == False:
            self.first_map = Dense(input_dim[1]*2)
        # outputs tensor of shape (max_obs,d_model)
        # key_dim determines size of W_q, W_k, and W_v to be (d_model, key_dim)
        # in "attention is all you need" key_dim was set to d_model/num_heads
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_k)

        # the outpu needs to be d_model
        # wide and shallow network ff_dim should be larger than d_model
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(d_model)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        #query = self.pool(inputs)
        query = tf.transpose(inputs, perm=[0,2,1])
        query = self.first_map(query)
        query = tf.transpose(query, perm=[0,2,1])
        attn_output = self.att(query, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(query + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_mask(self, x, mask=None):
        @tf.function
        def pool_mask(mask):
            size = tf.shape(mask)[1]
            even_ind = tf.range(0, size, delta=2)
            odd_ind = tf.range(1, size, delta=2)
            mask = tf.math.logical_or(tf.gather(params=mask, indices=even_ind, axis=1),
                                      tf.gather(params=mask, indices=odd_ind, axis=1))
            return mask

        return pool_mask(mask)

class ReduceMeanNorm(keras.layers.Layer):
    def __init__(self, axis=1):
        super(ReduceMeanNorm, self).__init__()
        self.axis = axis

    def call(self, inputs):
        rm = tf.reduce_mean(inputs, axis=self.axis)
        return tf.math.l2_normalize(rm, axis=self.axis)

class TokenEmbedding(Layer):
    def __init__(self, vocab_size, d_model, mask_zero=False):
        super(TokenEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=d_model,
                                   mask_zero=mask_zero,
                                   embeddings_initializer=tf.keras.initializers.GlorotNormal(),
                                   )
        self.scale = tf.math.sqrt(tf.cast(d_model, tf.float32))
        self.mask_zero = mask_zero

    def call(self, x):
        return self.token_emb(x) / self.scale

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        return self.token_emb.compute_mask(x)

class PhyloEmbedding(Layer):
    def __init__(self, num_sentinel, d_model, mask_zero=False):
        super(PhyloEmbedding, self).__init__()
        # add one to account for relative abundance
        self.sentinel_emb = Embedding(input_dim=num_sentinel, output_dim=d_model,
                                      embeddings_initializer=tf.keras.initializers.GlorotNormal())
        self.sent_indicies = tf.range(0, num_sentinel)
        self.scale = tf.math.sqrt(tf.cast(num_sentinel, tf.float32))
        self.linear_proj = Dense(num_sentinel, use_bias=False)
        self.use_mask = mask_zero

    # input will be an (B, max_len, sentinel)
    def call(self, x):

        # outputs (B, sentinel, d_model)
        sent_pos = tf.transpose(self.sentinel_emb(self.sent_indicies)) / self.scale

        # outputs (B, d_model, sentinel)
        return self.linear_proj(x) + sent_pos[tf.newaxis, :]

    def compute_mask(self, inputs, mask=None):
        if not self.use_mask:
            return None
        return tf.math.greater(tf.gather(params=inputs, indices=0, axis=2), 0)
            