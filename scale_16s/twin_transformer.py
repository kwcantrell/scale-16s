import os
import json
import gzip
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from simple_transformer import TransformerBlock, FunnelTransformerBlock, ReduceMeanNorm, TokenEmbedding
from project_embedding import project_embbeddings
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense, Flatten
from keras.models import Sequential, Model
import random
import psutil

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

model_prefix = 'emp500'
model_dir = f'models/{model_prefix}'

with open(os.path.join(model_dir, 'params.json')) as f:
    model_params = json.load(f)

with gzip.open(os.path.join(model_dir, 'class-data.gz'), 'rb') as f:
    data = json.loads(f.read().decode('utf-8'))['class_data']

## params ##
data_prefix = model_params['data_prefix']
total_features = model_params['total_features']
batch_size = model_params['batch_size']
items_per_class = model_params['items_per_class']
max_len = model_params['max_len' ]
d_model = model_params['d_model']
d_key = model_params['d_key']
num_heads = model_params['num_heads']
ff_dim = model_params['ff_dim']
mask_zero = model_params['mask_zero']
vocab_size = model_params['vocab_size']

n_samples=4
data = tf.ragged.constant(data)

total_epochs = 200
steps_per_epoch = 5
cycles=40
cycle_epochs = int(total_epochs/cycles)
cur_epoch = 0

enq = tf.keras.utils.OrderedEnqueuer(DataLoader(data, batch_size, items_per_class, max_len),  use_multiprocessing=False)
enq.start(workers=1, max_queue_size=1)
dataset = tf.data.Dataset.from_generator(
    enq.get,
    output_signature=(
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32),tf.TensorSpec(shape=[batch_size], dtype=tf.int32)))

for x in enq.get():
    print(x)
    break

# # tb_callback = tf.keras.callbacks.TensorBoard(
# #     log_dir='logs/datasets',
# #     write_graph=False,
# #     write_steps_per_second=True,
# #     profile_batch="1,20")

# model = tf.keras.Sequential()
# model.add(TokenEmbedding(vocab_size, d_model, mask_zero=mask_zero))
# model.add(TransformerBlock(d_model, d_key, num_heads, ff_dim))
# model.add(FunnelTransformerBlock(d_model, d_key, num_heads, ff_dim))
# model.add(FunnelTransformerBlock(d_model, d_key, num_heads, ff_dim))
# model.add(FunnelTransformerBlock(d_model, d_key, num_heads, ff_dim))
# model.add(FunnelTransformerBlock(d_model, d_key, num_heads, ff_dim))
# model.add(ReduceMeanNorm())
# model.add(tf.keras.layers.Flatten())


# ## initialize model ##
# model.compile(
#     optimizer=tf.keras.optimizers.Adafactor(0.001),
#     loss=tfa.losses.TripletSemiHardLoss())

# for c in range(cycles):
#     model.fit(
#         dataset,
#         epochs=(c+1)*cycle_epochs,
#         initial_epoch=c*cycle_epochs,
#         steps_per_epoch=steps_per_epoch,
#         workers=0)
# # # cur_epoch += cycles

# model.save(os.path.join(model_dir, 'encoder'))
