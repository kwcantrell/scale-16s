import tensorflow as tf
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from keras.datasets import imdb # dont need
from keras.models import Sequential, Model
import numpy as np
import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


    
class TokenEmbedding(Layer):
    def __init__(self, vocab_size, embed_dim, mask_zero=False):
        super(TokenEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=mask_zero)
        self.mask_zero = mask_zero

    def call(self, x):
        x = self.token_emb(x)
        return x

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        return self.token_emb.compute_mask(x)
            