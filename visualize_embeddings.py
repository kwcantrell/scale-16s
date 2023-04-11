import pandas as pd
import numpy as np
import tensorflow as tf
from simple_transformer import TransformerBlock, TokenEmbedding
from biom import load_table
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from keras.models import Sequential, Model
from util import get_global_constant, get_observation_vocab
import os

# load data
obs_vocab = get_observation_vocab(get_global_constant("BASE_16S_TABLE"))

log_dir="logs/16s-sample-embeddings"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save observations separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
    for ob in obs_vocab:
        f.write('{}\n'.format(ob))
    
