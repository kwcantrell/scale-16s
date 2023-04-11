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

NUM_CLASSES=3
NUM_SAMPLES_PER_CLASS=4
MAX_FEATURES_PER_SAMPLE=3
BATCH_SIZE=32
def shuffle_class_order(line):
    defs = [int()]*MAX_FEATURES_PER_SAMPLE
    fields = tf.io.decode_csv(line, record_defaults=defs, field_delim=',')
    fields = tf.transpose(fields)
    return fields

def create_triplets(classes):
    classes = tf.random.shuffle(classes)
    positives = classes[0, :]
    negatives = tf.reshape(classes[1:, :], [-1, MAX_FEATURES_PER_SAMPLE])
    
    positives = tf.random.shuffle(positives)
    negatives = tf.random.shuffle(negatives)
    
    ancor = positives[tf.newaxis, 0, :]
    ancor = tf.tile(ancor, tf.constant([NUM_SAMPLES_PER_CLASS-1,1]))
    positives = positives[1:, :]
    negatives = negatives[:(NUM_SAMPLES_PER_CLASS-1)]
    return ancor, positives, negatives

def create_dataset(inputs_dir):
    dataset = tf.data.Dataset.list_files(inputs_dir, shuffle=False)
    dataset = dataset.interleave(
      lambda file_path: tf.data.TextLineDataset(file_path),
      cycle_length=tf.data.AUTOTUNE,
      num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(shuffle_class_order, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(NUM_SAMPLES_PER_CLASS) # batch size should be # examples per class
    dataset = dataset.batch(NUM_CLASSES) # batch size should be # class
    dataset = dataset.cache()
    dataset = dataset.map(create_triplets, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat(NUM_SAMPLES_PER_CLASS-1)
    dataset = dataset.repeat(NUM_SAMPLES_PER_CLASS)
    dataset = dataset.unbatch()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

dataset = create_dataset('test.inputs')
for i in range(2):
    for s in dataset:
        # print(o)
        print(s)
        # print(i)
        print('run!!!!')

# step 3 create model
embed_dim=128
num_heads=5
num_layers=2
ff_dim=256
max_obs=MAX_OBS_PER_SAMPLE

def call_twin():
  for i in range(num_layers):
    x_obs_16s = transformer_blocks[i](x_obs_16s)
  x_obs_16s = tf.reshape(x_obs_16s, [x_obs_16s.shape[0], -1])
  x_obs_16s = tf.keras.layers.Concatenate(axis=1)([x_obs_16s, input_counts_16s])
  x_obs_16s = trans_dense(x_obs_16s)


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
