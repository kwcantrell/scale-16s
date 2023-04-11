import pandas as pd
import numpy as np
import tensorflow as tf
from simple_transformer import TransformerBlock, TokenEmbedding
from biom import load_table
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from keras.models import Sequential, Model


# Save table #
table_path = 'dataset/training/data/16s-full-train.biom'
table = load_table(table_path)
vocab = table.shape[0]
MAX_OBS_PER_SAMPLE=int(np.max(table.pa(inplace=False).sum(axis='sample')))
BATCH_SIZE=32
def process_input(line):
    defs = [tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs, field_delim='\t')

    defs = [tf.constant([], dtype=tf.int32)]*MAX_OBS_PER_SAMPLE
    obs = tf.io.decode_csv(fields[1], record_defaults=defs, field_delim=",") 
    counts = tf.io.decode_csv(fields[2], record_defaults=defs, field_delim=",")
    sample_depth = tf.reduce_sum(counts, 0, keepdims=True)
    counts = tf.math.divide(counts, sample_depth)
    dist = fields[-1:]

    return (tf.stack(obs),tf.stack(counts)), tf.stack(dist)

def csv_reader_dataset(file_path):
    ds = tf.data.Dataset.list_files(file_path, shuffle=False).shuffle(16).repeat().interleave(
        lambda file_path: tf.data.TextLineDataset(file_path), cycle_length=1)
    ds = ds.shuffle(10000)
    ds = ds.map(process_input)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(1000)

ds = csv_reader_dataset('dataset/training/input/*')
v_ds = csv_reader_dataset('dataset/training/input/*')
# for x in ds.take(32):
#     # process_input(file)
#     # print(ds.map(process_input(line)))
#     print(x)



# step 3 create model
embed_dim=128
num_heads=5
num_layers=4
ff_dim=256
max_obs=MAX_OBS_PER_SAMPLE

def build_model(mask_zero=False):
  input_obs = Input(shape=(max_obs), batch_size=BATCH_SIZE)
  input_counts = Input(shape=(max_obs), batch_size=BATCH_SIZE)
  
  embedding_layer = TokenEmbedding(vocab, embed_dim, mask_zero=mask_zero)
  x_obs = embedding_layer(input_obs)
  transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) 
                        for _ in range(num_layers)]
  for i in range(num_layers):
    x_obs = transformer_blocks[i](x_obs)
  x_obs = tf.reshape(x_obs, [x_obs.shape[0], -1])
  x_obs = tf.keras.layers.Concatenate(axis=1)([x_obs, input_counts])
  x_obs = Dense(max_obs, activation="relu")(x_obs)
  x_obs = Dense(max_obs/2, activation="relu")(x_obs)
  outputs = Dense(1)(x_obs)
  model = Model(inputs=(input_obs, input_counts), outputs=outputs)

  learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(0.001, 1000, alpha=0.00001, m_mul=0.5)
  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,
                                      epsilon=1e-8)
  model.compile(loss="mse", optimizer=optimizer)

  return model

model = build_model(mask_zero=True)
cp_callback = tf.keras.callbacks.ModelCheckpoint('dataset/checkpoint/cp.ckpt',
                                                save_weights_only=True,
                                                verbose=1)
model.fit(ds,steps_per_epoch=1000, epochs=5, validation_steps=75,callbacks=[cp_callback])
# model.load_weights('dataset/checkpoint/cp.ckpt')
# print(model.predict(v_ds , steps=1))
for x, y in ds.take(1):
  pred_y = model.call(x, training=False)
  print(y, pred_y)
  # print(y)
