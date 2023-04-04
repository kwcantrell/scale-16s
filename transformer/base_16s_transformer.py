import pandas as pd
import numpy as np
import tensorflow as tf
from simple_transformer import TransformerBlock, TokenEmbedding
from biom import load_table
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from keras.models import Sequential, Model
# import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# modle encoder transformer and append feature counts and feed into linear layer.

REL_PATH='samples'

# step 1 get human 16s samples
table_16s = load_table('%s/16s.biom' % REL_PATH)
df_16s = pd.read_csv('%s/16s-metadata.txt' % REL_PATH, sep='\t', index_col=0)
df_human = df_16s.loc[(df_16s.env_package.str.contains('human') & ~(df_16s.env_package.str.contains('associated')))]
human_sample_ids = df_human.index
table_human_16s = table_16s.filter(human_sample_ids)
table_human_16s.remove_empty(axis='observation', inplace=True)
num_cat = len(df_human.env_package.unique())
# step 2 create training data
max_obs = np.max(table_human_16s.pa(inplace=False).sum(axis='sample'))
# create an of rarefy table
seq_to_num = {o_id:i for i, o_id  in enumerate(table_human_16s.ids(axis='observation'))}
num_seq = len(seq_to_num)


# step 3 create model
embed_dim=8
num_heads=5
ff_dim=8

inputs = Input(shape=(max_obs,))
embedding_layer = TokenEmbedding(max_obs, num_seq, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)


# print(table_human_16s.shape)
# print(df_16s.loc[(df_16s.env_package.str.contains('human') & ~(df_16s.env_package.str.contains('associated')))].groupby('sample_type').count())
# sample_ids = wgs_df.loc[wgs_df['env_package'] == 'human-gut'].index
