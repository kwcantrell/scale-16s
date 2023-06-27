import os
import json
import gzip
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from simple_transformer import TransformerBlock, FunnelTransformerBlock, ReduceMeanNorm, TokenEmbedding, PhyloEmbedding
from project_embedding import project_embbeddings
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense, Flatten
from keras.models import Sequential, Model
import random
import psutil
from dataset_utils import DatasetCreater

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, items_per_class, max_len, num_sentinels):
        self.data = data
        self.batch_size = 2
        self.items_per_class = 2
        self.maxlen = max_len
        # self.num_classes = tf.shape(self.data)[0]
        self.num_classes = self.data.num_unique_classes
        self.epoch_size = self.num_classes*self.items_per_class
        self.r_order = None
        self.batches = None
        self.classes = tf.repeat(tf.range(0, self.num_classes), self.items_per_class)
        self.num_batches = tf.cast(self.epoch_size / self.batch_size, dtype=tf.int32)
        self.num_sentinels =  num_sentinels
        self.is_initialzed = False
        self.__get_batches()


    @tf.function
    def __get_batches(self):
        if not self.is_initialzed:
            self.batches = tf.Variable(
                tf.zeros(shape=[self.epoch_size, self.maxlen, self.num_sentinels], dtype=tf.float32), trainable=False)
            self.r_order =tf.Variable(
                tf.random.shuffle(tf.range(0, self.epoch_size, dtype=tf.int32 )), trainable=False)
            self.is_initialzed = True

        #helper function
        def get_epoch_samples(samples):
            num_samps = tf.shape(samples)[0]
            if num_samps >= self.items_per_class:
                indices = tf.range(0, num_samps)
            else:
                rep = tf.cast(tf.math.ceil(self.items_per_class / num_samps), dtype=tf.int32)
                indices = tf.repeat(tf.range(0, num_samps), repeats=rep)
            
            samples = tf.gather(params=samples, indices=indices)
            samples = samples.to_tensor(shape=[tf.shape(samples)[0], self.maxlen, self.num_sentinels])
            samples = tf.random.shuffle(samples)
            return samples[:self.items_per_class]
        
        data = tf.map_fn(fn=lambda t: get_epoch_samples(t), elems=self.data,
                        fn_output_signature=tf.TensorSpec([None, self.maxlen, self.num_sentinels], dtype=tf.float32))
        data = tf.reshape(data, shape=[self.epoch_size, self.maxlen, self.num_sentinels])

        self.batches.assign(data, use_locking=False, read_value=False)
        self.r_order.assign(tf.random.shuffle(self.r_order), use_locking=False)

    def __len__(self):
        return self.num_batches

    def on_epoch_end(self):
        self.__get_batches()

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        if end <= self.epoch_size:
            indices = self.r_order[start:end]
        else:
            indices = tf.concat([self.r_order[start:], self.r_order[:(end-start)]], axis=0)
        return (tf.gather(params=self.batches, indices=indices),
                    tf.gather(params=self.classes, indices=indices))

model_prefix = 'emp500'
model_dir = f'models/{model_prefix}'

with open(os.path.join(model_dir, 'params.json')) as f:
    model_params = json.load(f)

data = DatasetCreater(model_params['data_prefix'])

## params ##
data_prefix = model_params['data_prefix']
# total_features = model_params['total_features']
batch_size = model_params['batch_size']
items_per_class = model_params['items_per_class']
max_len = model_params['max_len' ]
d_model = 512# model_params['d_model'] 
d_key = model_params['d_key']
num_heads = model_params['num_heads']
ff_dim = model_params['ff_dim']
mask_zero = model_params['mask_zero']
vocab_size = 512#model_params['vocab_size']
num_sentinels=512

# #                <fixed>     <varies>   <fixed>   
# # data will be (# samples, # feautues, sentiel)
# data = tf.ragged.constant([[
#     [
#         [1., 2., 3., 4.], 
#         [1., 2., 3., 4.],
#         [1., 2., 3., 4.],
#         [1., 2., 3., 4.],
#         [1., 2., 3., 4.],
#     ],
#     [
#         [1., 2., 3., 4.],
#         [1., 2., 3., 4.], 
#         [1., 2., 3., 4.],
#         [1., 2., 3., 4.],
#         [1., 2., 3., 4.],
#     ],
# ]])

# print(data[0])
# total_epochs = 300
# steps_per_epoch = 5
# cycles=60
# cycle_epochs = int(total_epochs/cycles)
# cur_epoch = 0

enq = tf.keras.utils.OrderedEnqueuer(DataLoader(data, batch_size, items_per_class, max_len, num_sentinels),  use_multiprocessing=False)
enq.start(workers=1, max_queue_size=1)
# dataset = tf.data.Dataset.from_generator(
#     enq.get,
#     output_signature=(
#         tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32),tf.TensorSpec(shape=[batch_size], dtype=tf.int32)))

for x in enq.get():
    print(x)
    break
# # tb_callback = tf.keras.callbacks.TensorBoard(
# #     log_dir='logs/datasets',
# #     write_graph=False,
# #     write_steps_per_second=True,
# #     profile_batch="1,20")

d_model=max_len
model = tf.keras.Sequential()
model.add(PhyloEmbedding(num_sentinels, d_model, mask_zero=True))
model.add(TransformerBlock(num_sentinels, d_key, num_heads, ff_dim))
model.add(TransformerBlock(num_sentinels, d_key, num_heads, ff_dim))
model.add(FunnelTransformerBlock(num_sentinels, d_key, num_heads, ff_dim))
model.add(FunnelTransformerBlock(num_sentinels, d_key, num_heads, ff_dim))
model.add(TransformerBlock(num_sentinels, d_key, num_heads, ff_dim))
model.add(FunnelTransformerBlock(num_sentinels, d_key, num_heads, ff_dim))
model.add(FunnelTransformerBlock(num_sentinels, d_key, num_heads, ff_dim))
model.add(ReduceMeanNorm())
model.add(tf.keras.layers.Flatten())


## initialize model ##
model.compile(
    optimizer=tf.keras.optimizers.Adafactor(0.001),
    loss=tfa.losses.TripletSemiHardLoss())

for (x, y) in enq.get():
    pred = model(x)
    print('pred', pred.shape, pred)
    break
print(model.summary())
# for c in range(cycles):
#     model.fit(
#         dataset,
#         epochs=(c+1)*cycle_epochs,
#         initial_epoch=c*cycle_epochs,
#         steps_per_epoch=steps_per_epoch,
#         workers=0)
# # # cur_epoch += cycles

# model.save(os.path.join(model_dir, 'encoder'))
