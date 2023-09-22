import tensorflow as tf
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer, Conv1D
from keras.layers import Embedding, Input, AveragePooling1D, Dense, AveragePooling2D
from keras.datasets import imdb # dont need
from keras.models import Sequential, Model
import numpy as np
import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

@keras.saving.register_keras_serializable(package="Scale16s")
class BaseTransformerBlock(Layer):
    def __init__(self, **kwargs):
        super(BaseTransformerBlock, self).__init__()
        self.num_heads = kwargs['num_heads']
        self.num_sent = kwargs['num_sent']
        self.rate = kwargs['rate']
        self.use_causal_mask = False
        if 'att' in kwargs:
            self.att = kwargs['att']
        else:
            # outputs tensor of shape (fixed_len, num_sent)
            # key_dim determines size of W_q, W_k, and W_v to be (fixed_len, key_dim)
            # in "attention is all you need" key_dim was set to fixed_len/num_heads
            self.att = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=int(self.num_sent/self.num_heads),
                use_bias=True
            )
        # the output needs to be num_sent
        # wide and shallow network ff_dim should be larger than num_sent
        self.ffn = Sequential(
            [Dense(self.num_sent*2, activation="relu"), 
            Dense(self.num_sent)]
        )
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.supports_masking = True
    
    def _call_base_block(self, query, value, training=False, mask=None):
        attn_output = self.att(
            query, value,
            attention_mask=mask,
            use_causal_mask=self.use_causal_mask
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(query + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'att': tf.keras.layers.serialize(self.att),
            'num_heads': self.num_heads,
            'num_sent': self.num_sent,
            'rate': self.rate,
        })
        return config    

    @classmethod
    def from_config(cls, config):
        config['att'] = tf.keras.layers.deserialize(config['att'])
        return cls(**config)

@keras.saving.register_keras_serializable(package="Scale16s")
class TransformerBlock(BaseTransformerBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, training=True, mask=None):
        if mask is not None:
            mask_shape = tf.shape(mask)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.broadcast_to(mask, [mask_shape[0], mask_shape[1], mask_shape[1]])
        return self._call_base_block(query=inputs, value=inputs, training=training, mask=mask)

# @keras.saving.register_keras_serializable(package="Scale16s")
# class CrossAttnTransformerBlock(BaseTransformerBlock):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
    
#     def call(self, inputs, training):
#         (x, context) = inputs
#         return self._call_base_block(x, context, training)

# @keras.saving.register_keras_serializable(package="Scale16s")
# class CausalAttnTransformerBlock(BaseTransformerBlock):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
    
#     def call(self, inputs, training):
#         return self._call_base_block(inputs, inputs, training)


@keras.saving.register_keras_serializable(package="Scale16s")
class FunnelTransformerBlock(BaseTransformerBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = AveragePooling1D(pool_size=2, strides=2)
    
    def call(self, inputs, training=True, mask=None):
        if mask is not None:
            mask_shape = tf.shape(mask)
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.broadcast_to(mask, [mask_shape[0], int(mask_shape[1]/2), mask_shape[1]])
        query = self.pool(inputs)
        return self._call_base_block(query=query, value=inputs, training=training, mask=mask)
    
    def compute_mask(self, x, mask=None):
        if mask is None:
           return None
        size = tf.shape(mask)[1]
        even_ind = tf.range(0, size, delta=2)
        odd_ind = tf.range(1, size, delta=2)
        mask = tf.math.logical_or(tf.gather(params=mask, indices=even_ind, axis=1),
                                    tf.gather(params=mask, indices=odd_ind, axis=1))
        return mask

@keras.saving.register_keras_serializable(package="Scale16s")
class EncodingLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.num_heads = kwargs['num_heads']
        self.num_sent = kwargs['num_sent']
        self.rate = kwargs['rate']
      
        self.trans_block = TransformerBlock(**kwargs)
        self.funnel_block = FunnelTransformerBlock(**kwargs)
    
    def call(self, inputs, training=True, mask=None):
        trans_out = self.trans_block(inputs, training=training, mask=mask)
        return self.funnel_block(trans_out, training=training, mask=mask)
    
    def compute_mask(self, inputs, mask=None):
       return self.funnel_block.compute_mask(inputs, mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'num_sent': self.num_sent,
            'rate': self.rate,
        })
        return config
    
@keras.saving.register_keras_serializable(package="Scale16s")
class QuadReduceTransformer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.num_heads = kwargs['num_heads']
        self.num_sent = kwargs['num_sent']
        self.rate = kwargs['rate']
        self.max_obs = int(kwargs['max_obs']/8)

        self.blocks = []
        for _ in range(8):
            self.blocks.append(TransformerBlock(**kwargs))
        self.supports_masking=True
    
    def call(self, inputs, training=True, mask=None):
        start = 0
        end = self.max_obs
        x = inputs[:, start:end]
        m = mask[:, start:end]
        context = self.blocks[0](x, training=training, mask=m)

        mask = tf.expand_dims(mask, axis=-1)
        for block in self.blocks[1:]:
            start += self.max_obs
            end += self.max_obs
            x = inputs[:, start:end]
            m = mask[:, start:end]
            context = block._call_base_block(query=x,
                                            value=context,
                                            training=training,
                                            mask=m)
        return context
    
    def compute_mask(self, x, mask=None):
        if mask is None:
           return None
        return mask[:, :self.max_obs]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'num_sent': self.num_sent,
            'rate': self.rate,
            'max_obs': self.max_obs
        })
        return config

@keras.saving.register_keras_serializable(package="Scale16s")
class DecodingLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.num_heads = kwargs['num_heads']
        self.num_sent = kwargs['num_sent']
        self.rate = kwargs['rate']
      
        self.trans_block = TransformerBlock(**kwargs)
        self.funnel_block = FunnelTransformerBlock(**kwargs)
    
    def call(self, inputs, training, mask=None):
        trans_out = self.trans_block(inputs, training, mask)
        self.output_attention = self.funnel_block(trans_out, training, mask)
        return self.output_attention
    
    def compute_mask(self, inputs, mask=None):
       return self.funnel_block.compute_mask(inputs, mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'num_sent': self.num_sent,
            'rate': self.rate,
        })
        return config


@keras.saving.register_keras_serializable(package="Scale16s")
class ReduceMeanNorm(keras.layers.Layer):
    def __init__(self, axis=1):
        super(ReduceMeanNorm, self).__init__()
        self.axis = axis

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=self.axis)
        return x
    
@keras.saving.register_keras_serializable(package="Scale16s")
class SentinelEmbedding(Layer):
    def __init__(self, **kwargs):
        super(SentinelEmbedding, self).__init__()
        self.max_obs = kwargs['max_obs']
        self.fixed_len = kwargs['fixed_len']
        self.rate = kwargs['rate']
        self.sent_embeddings = Sequential([
            Dense(self.fixed_len, 
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    use_bias=True,
            )]
        )
        # self.dropout = Dropout(self.rate)
        self.supports_masking = True
    
    # @tf.function
    def call(self, inputs, training=True, mask=None):
        x = tf.transpose(inputs, perm=[0,2,1])
        # emb = self.sent_embeddings(self.dropout(x, training=training))
        emb = self.sent_embeddings(x)
        return tf.transpose(emb, perm=[0,2,1])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_obs': self.max_obs, 
            'fixed_len': self.fixed_len, 
            'rate': self.rate, 
        })
        return config   
    
    def compute_mask(self, inputs, mask=None):
        if mask is None:
           return None
        return mask[:, :self.fixed_len]

    
# @keras.saving.register_keras_serializable(package="Scale16s")
# class TokenEmbedding(Layer):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.total_obs = kwargs['total_obs']
#         self.max_obs = kwargs['max_obs']
#         self.num_sent = kwargs['num_sent']
#         self.fixed_len = kwargs['fixed_len']
#         self.embedding = tf.keras.layers.Embedding(
#             self.total_obs, self.num_sent, mask_zero=True,
#             input_length=self.max_obs
#         )
#         self.mask_zero=True
#         self.supports_masking=True

#     def compute_mask(self, x, mask=None):
#         if not self.mask_zero:
#             return None
#         return self.embedding.compute_mask(x)[:, :self.fixed_len]
    
#     def call(self, x, training=False, mask=None):
#         return self.embedding(x)
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'total_obs': self.total_obs,
#             'max_obs': self.max_obs,
#             'num_sent': self.num_sent,
#             'fixed_len': self.fixed_len,
#         })
#         return config

@keras.saving.register_keras_serializable(package="Scale16s")
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()
    self.d_model = d_model
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
@keras.saving.register_keras_serializable(package="Scale16s")
def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


@keras.saving.register_keras_serializable(package="Scale16s")
def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

# @keras.saving.register_keras_serializable(package="Scale16s")
# class TokenEmbedding(Layer):
#     def __init__(self, vocab_size, d_model, mask_zero=False):
#         super(TokenEmbedding, self).__init__()
#         self.token_emb = Embedding(input_dim=vocab_size, output_dim=d_model,
#                                    mask_zero=mask_zero,
#                                    embeddings_initializer=tf.keras.initializers.GlorotNormal(),
#                                    )
#         self.scale = tf.math.sqrt(tf.cast(d_model, tf.float32))
#         self.mask_zero = mask_zero

#     def call(self, x):
#         return self.token_emb(x) / self.scale

#     def compute_mask(self, x, mask=None):
#         if not self.mask_zero:
#             return None
#         return self.token_emb.compute_mask(x)

# @keras.saving.register_keras_serializable(package="Scale16s")
# class PhyloEmbedding(Layer):
#     def __init__(self, num_sentinel, d_model, mask_zero=False):
#         super(PhyloEmbedding, self).__init__()
#         # add one to account for relative abundance
#         self.sentinel_emb = Embedding(input_dim=num_sentinel, output_dim=d_model,
#                                       embeddings_initializer=tf.keras.initializers.GlorotNormal())
#         self.sent_indicies = tf.range(0, num_sentinel)
#         self.scale = tf.math.sqrt(tf.cast(num_sentinel, tf.float32))
#         self.linear_proj = Dense(num_sentinel, use_bias=True)
#         self.use_mask = mask_zero

#     # input will be an (B, max_len, sentinel)
#     def call(self, x):

#         # outputs (B, sentinel, d_model)
#         sent_pos = tf.transpose(self.sentinel_emb(self.sent_indicies)) / self.scale

#         # outputs (B, d_model, sentinel)
#         return self.linear_proj(x) + sent_pos[tf.newaxis, :]

#     def compute_mask(self, inputs, mask=None):
#         if not self.use_mask:
#             return None
#         return tf.math.greater(tf.gather(params=inputs, indices=0, axis=2), 0)
            