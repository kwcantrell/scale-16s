import pandas as pd
import numpy as np
import tensorflow as tf
from biom import load_table
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from keras.models import Sequential, Model

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
