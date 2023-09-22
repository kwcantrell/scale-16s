import tensorflow as tf
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer, Conv1D
from keras.layers import Embedding, Input, AveragePooling1D, Dense, AveragePooling2D
from keras.datasets import imdb # dont need
from keras.models import Sequential, Model
import numpy as np
import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

@keras.saving.register_keras_serializable(package="custom_layer")
class TransformerBlock(Layer):
    def __init__(self, num_heads, num_sent, rate, include_ffn=True, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.num_sent = num_sent
        self.rate = rate
        self.use_causal_mask = False
        self.mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=int(self.num_sent/self.num_heads),
            use_bias=True
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)

        if include_ffn:
            self.ffn = Sequential(
                [Dense(self.num_sent*2, activation="relu"), 
                Dense(self.num_sent)]
            )
            self.dropout = Dropout(self.rate)

            self.layernorm2 = LayerNormalization(epsilon=1e-6)
        else:
            self.ffn = None
        self.supports_masking = True
    
    def bulld(self, inputs):
        self.mha._build_from_signature(inputs)

    def call(self, query, value, key=None, training=False, mask=None):
        if mask is not None:
            attention_mask = tf.expand_dims(mask, -1)
        else:
            attention_mask = None
        mha_out = self.mha(query, value, key=key, attention_mask=attention_mask, training=training)
        mha_out_norm = self.layernorm1(query + mha_out)

        if self.ffn is not None:
            ffn_out = self.ffn(mha_out_norm)
            ffn_out_drop = self.dropout(ffn_out + mha_out_norm)
            return self.layernorm2(ffn_out_drop)
        else:
            return mha_out_norm


@keras.saving.register_keras_serializable(package="Scale16s")
class ReductionTransformer(tf.keras.layers.Layer):
    def __init__(self, num_heads, num_sent, rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_sent = num_sent
        self.rate = rate

        self.left_att_head = TransformerBlock(num_heads, num_sent, rate, include_ffn=False)
        self.right_att_head = TransformerBlock(num_heads, num_sent, rate, include_ffn=False)
        self.merge_head = TransformerBlock(num_heads, num_sent, rate)
        self.supports_masking = True
    
    def build(self, input_shape):
        self.input_dim = tf.TensorShape(input_shape).as_list()[1]
        self.output_dim = int(self.input_dim/2)

    def call(self, input, training=False, mask=None):
        if mask is not None:
            left_mask = mask[:, :self.output_dim]
            right_mask = mask[:, self.output_dim:]
        else:
            left_mask = None
            right_mask = None

        left_input = input[:, :self.output_dim]
        left_att_out = self.left_att_head(
            left_input, left_input,
            training=training, mask=left_mask
        )

        right_input = input[:, self.output_dim:]
        right_att_out = self.right_att_head(
            right_input, right_input,
            training=training, mask=right_mask
        )

        merge_mask = tf.logical_and(left_mask, right_mask)
        return self.merge_head(
            left_att_out, left_att_out + right_att_out, 
            key=right_att_out, training=training, mask=merge_mask
        )

    def compute_mask(self, x, mask=None):
        if mask is None:
           return None
        left_mask = mask[:, :self.output_dim]
        right_mask = mask[:, self.output_dim:]
        return tf.logical_and(left_mask, right_mask)

@keras.saving.register_keras_serializable(package="Scale16s")
class QuadReductionTransformer(tf.keras.layers.Layer):
    def __init__(self, num_heads, num_sent, rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_sent = num_sent
        self.rate = rate

        self.small_blocks =[
            TransformerBlock(self.num_heads, self.num_sent, self.rate)
            for _ in range(8)
        ]
        self.supports_masking = True


    def build(self, input_shape):
        self.input_dim = tf.TensorShape(input_shape).as_list()[1]
        self.block_size = int(self.input_dim/8)

    # @tf.function(input_signature=[tf.TensorSpec(dtype=tf.float32, shape=[None, None, None])])
    def call(self, input, training=False, mask=None):
        
        def _reduce_with_mask(blocks, input, mask, training):
            ma = tf.TensorArray(tf.bool, size=8, dynamic_size=False,
                                clear_after_read=True)#, element_shape=[32, None]) 
            ma = ma.unstack(tf.reshape(mask, [32, 8, -1]))
            inpt = tf.TensorArray(tf.float32, size=8, dynamic_size=False,
                                 clear_after_read=True)#, element_shape=[32, None, 512])
            inpt = inpt.unstack(tf.reshape(input, [32, 8, -1, 512]))
            if not hasattr(self, 'inp_i'):
                inp_i, msk_i = tf.Variable(inpt.read(0), trainable=False), tf.Variable(ma.read(0), trainable=False)
                context_i = tf.Variable(blocks[0](inp_i, inp_i, training=training, attention_mask=msk_i), trainable=False)
            else:
                inp_i, msk_i = inp_i.assign(inpt.read(0)), msk_i.assign(ma.read(0))
                context_i = context_i.assign(blocks[0](inp_i, inp_i, training=training, attention_mask=msk_i))
                
            for i in tf.range(1, 8):
                inp_i, msk_i = inp_i.assign(inpt.read(i)), msk_i.assign(ma.read(i))
                context_i = context_i.assign(blocks[i](context_i, inpt.read(i), training=training, attention_mask=msk_i))
            return context_i

        if mask is not None:
            return _reduce_with_mask(self.small_blocks, input, mask, training)
            
    def compute_mask(self, x, mask=None):
        if mask is None:
           return None
        bi_mask = mask[:, :self.block_size]
        for i in range(7):
            bi_mask = tf.logical_and(bi_mask, mask[:, self.block_size*i:self.block_size*(i+1)])
        return bi_mask

@keras.saving.register_keras_serializable(package="Scale16s")
class EncodingLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, num_sent, rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_sent = num_sent
        self.rate = rate
      
        self.trans_block = TransformerBlock(self.num_heads, self.num_sent, self.rate)
        self.reduc_block = ReductionTransformer(self.num_heads, self.num_sent, self.rate)
        self.supports_masking=True
    
    def call(self, inputs, training=True, mask=None):
        trans_out = self.trans_block(inputs, inputs, training=training, mask=mask)
        return self.reduc_block(trans_out, training=training, mask=mask)
    
    def compute_mask(self, inputs, mask=None):
       return self.reduc_block.compute_mask(inputs, mask)

@keras.saving.register_keras_serializable(package="Scale16s")
class TokenEmbedding(Layer):
    def __init__(self, total_obs, output_size, max_per_sample, **kwargs):
        super().__init__(**kwargs)
        self.total_obs = total_obs
        self.output_size = output_size
        self.max_per_sample = max_per_sample
        self.embedding = tf.keras.layers.Embedding(
            self.total_obs, self.output_size, mask_zero=True,
            input_length=self.max_per_sample
        )
        self.supports_masking=True

    def compute_mask(self, x, mask=None):
        return self.embedding.compute_mask(x)
    
    def call(self, x, training=False, mask=None):
        return self.embedding(x)
    
@keras.saving.register_keras_serializable(package="Scale16s")
class ReduceMeanNorm(keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(ReduceMeanNorm, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=self.axis)
        return x

    
@keras.saving.register_keras_serializable(package="Scale16s")
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=tf.constant(4000, dtype=tf.float32)):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        tf_step = tf.constant(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(tf_step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
