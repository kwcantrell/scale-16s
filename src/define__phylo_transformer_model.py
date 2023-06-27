import os
import gzip
import json
import numpy as np
import pandas as pd
from dataset_utils import DatasetCreater

## params ##
model_params = {
    'data_prefix': 'phylo-emp500',
    'total_features': None, # total unique microbes
    'batch_size': 64,
    'items_per_class': 5,
    'max_len' : 2048, # maximum unique features in a sample
    'd_model': 128, # size of token embedding
    'd_key': 32, # SIZE OF KEY attention matrices
    'num_heads': 8, # attention head in first layer, each subsequent layer doubles
    'ff_dim': 128, # size of linear layer in transormer block
    'mask_zero': True, # The `0` token will be masked
    'total_epochs' : 5,
    'steps_per_epoch':10,
    'wgs_start_token': None,
    'num_sentinels': 512
}

prefix = model_params['data_prefix']
model_dir = f'models/{prefix}'
if not os.path.exists(model_dir):
        os.makedirs(model_dir)


# data = np.load('data/phylo-emp500-sentinel-tree-info.npz')
# print(data['sent_dist'].shape)


# fetch data 
class_data = DatasetCreater(model_params['data_prefix']).get_data()
print('?')

# print(pd.jsonnormaile({ 'a': data_creator.get_data()}))
# class_data = {'sent_dist': data_creator.get_data()}
# print('start gzipping!!!')
with gzip.open(os.path.join(model_dir, 'class-data.gz'), compresslevel=6, mode='wb') as f:
    f.write(json.dumps(class_data).encode('utf-8'))

# # # HOW TO READ CLASS DATA ##
# with gzip.open('data/faster-phylo-dist-info.json.gz', compresslevel=6, mode='rb') as f:
#     data = json.loads(f.read().decode('utf-8'))
#     print(data)
# # ---------------------- ##

# # model_params['vocab_size'] = data_creator.get_total_tokens()
# # model_params['token_to_class'] = data_creator.get_token_to_class()
# # model_params['wgs_start_token'] = data_creator.get_wgs_start_token()
with open(os.path.join(model_dir, 'params.json'), 'w') as f:
    f.write(json.dumps(model_params, indent=4))

