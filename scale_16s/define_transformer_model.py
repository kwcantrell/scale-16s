import os
import gzip
import json
import numpy as np
from dataset_utils import DatasetCreater

## params ##
model_params = {
    'data_prefix': 'emp500',
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
    # num_layers:1,
    # cycles:1,
    # vocab_size : num_features + 3,
    # cycle_steps : int(total_epochs/cycles),
    # cur_epoch : 0,
}

prefix = model_params['data_prefix']
model_dir = f'models/{prefix}'
if not os.path.exists(model_dir):
        os.makedirs(model_dir)

# fetch data 
data_creator = DatasetCreater(model_params['data_prefix'])
class_data = {'class_data': data_creator.get_data()}
with gzip.open(os.path.join(model_dir, 'class-data.gz'), 'wb') as f:
    f.write(json.dumps(class_data).encode('utf-8'))

# # HOW TO READ CLASS DATA ##
# with gzip.open(os.path.join(model_dir, 'class-data.gz'), 'r') as f:
#     data = f.read().decode('utf-8)
# # ---------------------- ##

model_params['vocab_size'] = data_creator.get_total_tokens()
model_params['token_to_class'] = data_creator.get_token_to_class()
model_params['wgs_start_token'] = data_creator.get_wgs_start_token()
with open(os.path.join(model_dir, 'params.json'), 'w') as f:
    f.write(json.dumps(model_params, indent=4))

