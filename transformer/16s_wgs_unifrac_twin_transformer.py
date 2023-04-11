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
table_path = 'unifrac-dataset/training/data/filtered-merged.biom'
table = load_table(table_path)
vocab = table.shape[0]
MAX_OBS_PER_SAMPLE=int(np.max(table.pa(inplace=False).sum(axis='sample')))
print(MAX_OBS_PER_SAMPLE)
BATCH_SIZE=16
def process_input(line):
    # load data from files
    defs = [float()] + [str()]*4
    fields = tf.io.decode_csv(line, record_defaults=defs, field_delim='\t')
    distance = fields[0]
    table_info = tf.strings.split(fields[1:], sep=',')
    table_info = tf.strings.to_number(table_info, out_type=tf.dtypes.int32)

    # make batch index come first
    table_info = table_info.to_tensor(shape=(4, BATCH_SIZE, MAX_OBS_PER_SAMPLE))
    table_info = tf.transpose(table_info, perm=[1,0,2])

    # normalize the table so that sample depths sum to 1
    s1_obs, s1_count, s2_obs, s2_count = tf.split(table_info, num_or_size_splits=4, axis=1)
    s1_depth = tf.math.reduce_sum(s1_count, axis=2, keepdims=True)
    s1_count = tf.math.divide(s1_count, s1_depth)
    s2_depth = tf.math.reduce_sum(s2_count, axis=2, keepdims=True)
    s2_count = tf.math.divide(s2_count, s2_depth)

    # remove dim
    s1_obs = tf.reshape(s1_obs,[BATCH_SIZE, -1])
    s1_count = tf.reshape(s1_count,[BATCH_SIZE, -1])
    s2_obs = tf.reshape(s2_obs,[BATCH_SIZE, -1])
    s2_count = tf.reshape(s2_count,[BATCH_SIZE, -1])

    return (s1_obs, s1_count, s2_obs, s2_count), distance

def csv_reader_dataset(file_path):
    ds = tf.data.Dataset.list_files(file_path, shuffle=False)
    ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(1000)
    ds = ds.interleave(
      lambda file_path: tf.data.TextLineDataset(file_path),
      cycle_length=tf.data.AUTOTUNE,
      num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10000)
    ds = ds.batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(process_input, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

ds = csv_reader_dataset('unifrac-dataset/training/input/*')
v_ds = csv_reader_dataset('unifrac-dataset/training/input/*')
# for (s1_obs, s1_count, s2_obs, s2_count), y in ds.take(1):
#     # process_input(file)
#     # print(ds.map(process_input(line)))
#     print(s2_obs,s2_count)



# step 3 create model
embed_dim=128
num_heads=5
num_layers=4
ff_dim=256
max_obs=MAX_OBS_PER_SAMPLE

def build_model(mask_zero=False):
  input_obs_16s = Input(shape=(max_obs), batch_size=BATCH_SIZE)
  input_counts_16s = Input(shape=(max_obs), batch_size=BATCH_SIZE)
  input_obs_wgs = Input(shape=(max_obs), batch_size=BATCH_SIZE)
  input_counts_wgs = Input(shape=(max_obs), batch_size=BATCH_SIZE)
  embedding_layer = TokenEmbedding(vocab, embed_dim, mask_zero=mask_zero)
  transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) 
                        for _ in range(num_layers)]
  trans_dense = Dense(max_obs, activation="relu")

  #################### 16s transformer twin #############################
  x_obs_16s = embedding_layer(input_obs_16s)
  for i in range(num_layers):
    x_obs_16s = transformer_blocks[i](x_obs_16s)
  x_obs_16s = tf.reshape(x_obs_16s, [x_obs_16s.shape[0], -1])
  x_obs_16s = tf.keras.layers.Concatenate(axis=1)([x_obs_16s, input_counts_16s])
  x_obs_16s = trans_dense(x_obs_16s)

  ################### wgs transformer twin ############################
  x_obs_wgs = embedding_layer(input_obs_wgs)
  for i in range(num_layers):
    x_obs_wgs = transformer_blocks[i](x_obs_wgs)
  x_obs_wgs = tf.reshape(x_obs_wgs, [x_obs_wgs.shape[0], -1])
  x_obs_wgs = tf.keras.layers.Concatenate(axis=1)([x_obs_wgs, input_counts_wgs])
  x_obs_wgs = trans_dense(x_obs_wgs)

  ################### combine 16s and wgs ############################
  x_obs = tf.keras.layers.Concatenate(axis=1)([x_obs_16s, x_obs_wgs])
  x_obs = Dense(max_obs, activation="relu")(x_obs)
  x_obs = Dense(max_obs/2, activation="relu")(x_obs)
  outputs = Dense(1)(x_obs)

  model = Model(inputs=(input_obs_16s, input_counts_16s, input_obs_wgs, input_counts_wgs), outputs=outputs)

  learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(0.001, 1000, alpha=0.00001, m_mul=0.98)
  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,
                                      epsilon=1e-8)
  model.compile(loss="mse", optimizer=optimizer)

  return model

model = build_model(mask_zero=True)
cp_callback = tf.keras.callbacks.ModelCheckpoint('unifrac-dataset/checkpoint/cp.ckpt',
                                                save_weights_only=True,
                                                verbose=1)
model.load_weights('unifrac-dataset/checkpoint/cp.ckpt')
# model.fit(ds,steps_per_epoch=1000, epochs=120, validation_steps=1000,callbacks=[cp_callback])
print(model.predict(v_ds , steps=1))
for x, y in ds.take(1):
  pred_y = model.call(x, training=False)
  print(y, pred_y)
  # print(y)
