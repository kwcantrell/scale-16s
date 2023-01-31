import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pathlib
import time
import datetime
import math
from skbio.stats import subsample_counts
from biom import load_table
from matplotlib import pyplot as plt



REL_PATH='samples'
WGS_BASE_TABLE='wgs_table'
BASE_TABLE_16S='table_16s'
SAMPLE_TYPE='env_package'

# load biom tables
wgs_table = load_table('%s/wgs.biom' % REL_PATH)
wgs_df = pd.read_csv('%s/wgs-metadata.txt' % REL_PATH,sep='\t', index_col=0)
table_16s = load_table('%s/16s.biom' % REL_PATH)
df_16s = pd.read_csv('%s/16s-metadata.txt' % REL_PATH, sep='\t', index_col=0)

# def create_token(table):
# @tf.function
def create_token(table):
    obs = table.ids(axis='observation')
    obs = np.array([str(i) for i, _ in enumerate(obs)])
    vocab = np.concatenate((obs, ['[Z]', '[S]', '[E]']))
    
    max_features = vocab.size + 2
    max_len = obs.size +2

    count = table.shape[1]
    # table = tf.constant(table.matrix_data.todense())
    table = table.transpose().matrix_data.todense()
    # table = tf.transpose(table)

    feature_vecs = tf.greater(table, 0)
    feature_vecs = tf.where(feature_vecs, obs, [tf.constant('[Z]')])
    starts = tf.fill([count, 1], '[S]')
    ends = tf.fill([count, 1], '[E]')
    feature_vecs = tf.concat([starts, feature_vecs, ends], axis=1)
    feature_vecs = tf.strings.reduce_join(feature_vecs, separator=' ', axis=1)
    
    vocab = tf.data.Dataset.from_tensor_slices(vocab)
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=max_len,
    )
    vectorize_layer.adapt(vocab)

    tokenizer = tf.keras.models.Sequential()
    tokenizer.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    tokenizer.add(vectorize_layer)

    return tokenizer, feature_vecs


def filter_low_count(data, id, metadata):
    data[data <= 10] = 0.0
    return data

# extract gut samples
sample_ids = wgs_df.loc[wgs_df['env_package'] == 'human-gut'].index
sample_ids = sample_ids[:int(len(sample_ids)/2)]

# filter/rarefy tables
wgs_table.filter(sample_ids)
wgs_table.transform(filter_low_count)
wgs_table.remove_empty(axis='observation')
depth = int(np.percentile(wgs_table.sum(axis='sample'), 5))
print('wgs_table start', wgs_table.shape)
wgs_table = wgs_table.subsample(depth, axis='sample', with_replacement=False)
print('wgs_table end', wgs_table.shape)
wgs_ids = {i for i in wgs_table.ids(axis='sample')}
# wgs_table = wgs_table.head(10, 6)
wgs_tokenizer, wgs_feature_vecs = create_token(wgs_table)
wgs_tokenizer.save('transformer/tokenizers/wgs_tokenizer')
# print(wgs_table, wgs_tokenizer.predict(wgs_feature_vecs))
wgs_dataset = tf.data.Dataset.from_tensor_slices(wgs_feature_vecs)
wgs_dataset.save('transformer/datasets/wgs')

table_16s.filter(sample_ids)
table_16s.transform(filter_low_count)
table_16s.remove_empty(axis='observation')
depth = int(np.percentile(table_16s.sum(axis='sample'), 5))
print('16s_table start', table_16s.shape)
table_16s = table_16s.subsample(depth, axis='sample', with_replacement=False)
print('16s_table end', table_16s.shape)
ids_16s = {i for i in table_16s.ids(axis='sample')}

common_ids = wgs_ids.intersection(ids_16s)
wgs_table.filter(common_ids)
wgs_table.remove_empty(axis='observation')
table_16s.filter(common_ids)
table_16s.remove_empty(axis='observation')
print('final shapes', table_16s.shape, wgs_table.shape)
# wgs_table = wgs_table.head(10, 6)
# table_16s = table_16s.head(10, 6)

# get tokenizer and create datasets
wgs_tokenizer, wgs_feature_vecs = create_token(wgs_table)
wgs_tokenizer.save('transformer/tokenizers/wgs_tokenizer')
wgs_dataset = tf.data.Dataset.from_tensor_slices(wgs_feature_vecs)
wgs_dataset.save('transformer/datasets/wgs')

tokenizer_16s, feature_vecs_16s = create_token(table_16s)
tokenizer_16s.save('transformer/tokenizers/16s_tokenizer')
dataset_16s = tf.data.Dataset.from_tensor_slices(feature_vecs_16s)
dataset_16s.save('transformer/datasets/16s')

# dataset = tf.data.Dataset.load('transformer/datasets/wgs')
# token = tf.keras.models.load_model('transformer/tokenizers/wgs_tokenizer')
# print(token.layers[0])
# for elem in dataset:
#     # print(elem)
#     print(len(token.layers[0].get_vocabulary()))
#     break