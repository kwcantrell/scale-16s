import numpy as np
import tensorflow as tf
from biom import load_table
from bp import parse_newick
from simple_transformer import QuadReduceTransformer,SentinelEmbedding, EncodingLayer, TransformerBlock, FunnelTransformerBlock, ReduceMeanNorm, CustomSchedule, TokenEmbedding
import pandas as pd
import json
import os
from losses import unifrac_loss
from project_embedding import ProjectEncoder
from utils import align_tables, get_tip_info, get_tip_ids
from dataset_utils import dataset_creater, DataLoader, DataLoaderToken


def get_test_params():
    return {
        'model_path': 'models/test-sent',
        'data_path': 'test-data',
        'logdir':'logs',
        'metadata': 'scale_16s/test-data/metadata.tsv',
        'tree': 'scale_16s/test-data/test-tree.nwk',
        'num_sent': 4,
        'table_16s': 'scale_16s/test-data/test-table-16s.biom',
        'table_wgs': 'scale_16s/test-data/test-table-16s.biom',
        'num_rarefied': 5,
        's_depth_16s': 3,
        's_depth_wgs': 3,
        'fixed_len': 512,
        'mask_value': 0,
        'fix_seed': None,
        'num_heads': 1,
        'rate': 0.1,
        'items_per_sample': 1,
        'batch_size': 3,
        'groups_per_epoch': 2,
        'max_obs': 8192,
        'reduce_num_sent': False
    }
    
def get_emp500_params():
    return {
        'model_path': 'models/emp500-sent',
        'data_path': 'sent-data',
        'logdir': 'logs',
        'metadata': 'data/emp500-16s-metadata.txt',
        'tree': 'data/small-test/phylo-emp500-tree.nwk',
        'num_sent': 512, # must be a power of 2
        'table_16s': 'data/emp500-16s.biom',
        'table_wgs': 'data/emp500-wgs.biom',
        'num_rarefied': 5,
        's_depth_16s': 1000,
        's_depth_wgs': 500000,
        'fixed_len': 1024, # must be a power of 2
        'mask_value': 0,
        'fix_seed': None,
        'num_heads': 8, # must be a power of 2
        'rate': 0.1,
        'batch_size': 32,
        'max_obs': 8192, # must be a power of 2
        'reduce_num_sent': False,
    }
    
def get_input_params(test_params=False, load_data_path=False, load_model=False, make_deterministic=True):
    if make_deterministic:
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()

    if test_params:
        params = get_test_params()
    else:
        params = get_emp500_params()

    # save parameters
    with open(f"{params['model_path']}/parameters.json", 'w') as f:
        f.write(json.dumps(params))
    
    # load tree
    with open(params['tree'], 'r') as f:
        params['tree'] = parse_newick(f.readline())

    # load metadata
    params['metadata'] = pd.read_csv(
        params['metadata'],
        sep=',' if test_params else '\t',
        index_col=0
    )

    # special symbol to add to wgs to prevent dup sample names
    params['wgs_id'] = 'SCALE_ENC'

    print('getting params')
    if not load_data_path:
        # load 
        table_16s = load_table(params['table_16s'])
        table_wgs = load_table(params['table_wgs'])
        if not test_params:
            table_16s, table_wgs = align_tables(table_16s, table_wgs, params['tree'])
        params['table_16s'] = table_16s
        params['table_wgs'] = table_wgs
        tables, cached_dist, tip_info = dataset_creater(**params)
    else:
        model_path = params['model_path']
        data_path = params['data_path']
        parent_path = f'{model_path}/{data_path}'
        table_paths = [f"{parent_path}/table-{i}.biom" for i in range(params['num_rarefied'])]
        tables = [load_table(path).remove_empty(axis='observation') for path in table_paths]
        cached_dist = np.load(f'{parent_path}/cached_dist.npy', allow_pickle=True)
        tip_info = get_tip_info(params['tree'])

    params['tables'] = tables
    # params['extra_tables'] = [load_table(f"random-emp500-tables/table-{i}.biom").remove_empty(axis='observation')
    #                             for i in range(5)]
    params['cached_dist'] = cached_dist
    params['tip_info'] = tip_info
    params['num_epochs'] = 4000
    params['num_batches'] = 60
    print('done getting params')
    # params['data'] = DataLoader(**params)
    # params['feature_ids'] = np.array(get_tip_ids(params['tree']))
    params['total_obs'] = params['tables'][0].ids(axis='observation').shape[0]#params['feature_ids'].shape[0]+1
    params['data'] = DataLoaderToken(**params)
    params['load_model'] = load_model
    return params

# def build_encoder(params):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(params['max_obs'], params['num_sent']) , batch_size=params['batch_size']))
#     # model.add(tf.keras.Input(shape=params['max_obs'], batch_size=params['batch_size']))
#     model.add(tf.keras.layers.Masking(float(params['mask_value'])))
#     model.add(SentinelEmbedding(**params))
#     # model.add(TokenEmbedding(**params))
#     model.add(EncodingLayer(name='encoding_1',**params))
#     model.add(EncodingLayer(name='encoding_2',**params))
#     model.add(EncodingLayer(name='encoding_3',**params))
#     model.add(EncodingLayer(name='encoding_4',**params))
#     model.add(ReduceMeanNorm())
#     model.add(tf.keras.layers.Flatten())

#     lr = CustomSchedule(params['num_sent'])
#     optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#     # optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#     model.compile(
#         optimizer=optimizer,
#         loss=unifrac_loss
#     )
#     model.summary()

#     return model

def build_encoder(params):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=params['max_obs'], batch_size=params['batch_size']))
    # model.add(tf.keras.layers.Masking(float(params['mask_value'])))
    # model.add(SentinelEmbedding(**params))
    model.add(TokenEmbedding(**params))
    model.add(QuadReduceTransformer(**params))
    model.add(EncodingLayer(name='encoding_1',**params))
    model.add(EncodingLayer(name='encoding_2',**params))
    model.add(EncodingLayer(name='encoding_3',**params))
    model.add(EncodingLayer(name='encoding_4',**params))
    model.add(ReduceMeanNorm())
    # model.add(tf.keras.layers.Flatten())

    lr = CustomSchedule(params['num_sent'])
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    # optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(
        optimizer=optimizer,
        loss=unifrac_loss
    )
    model.summary()

    return model

if __name__ == '__main__':
    params = get_input_params(test_params=False, load_data_path=True, load_model=True)
    data = params['data']
    dataset = tf.data.Dataset.from_generator(
        data,
        output_signature=(
            tf.TensorSpec(shape=(params['batch_size'], params['max_obs']), dtype=tf.int32),
            # tf.TensorSpec(shape=(params['batch_size'], params['max_obs'], params['num_sent']), dtype=tf.int32),
            tf.TensorSpec(shape=(params['batch_size'], params['batch_size']), dtype=tf.float32)
        )
    )

    if not params['load_model']:
        # Encoder
        model = build_encoder(params)
        model.fit(
            dataset,
            epochs=params['num_epochs'],
            steps_per_epoch=data.__len__(),
            callbacks=[ProjectEncoder(params)],
        )

        model.save(os.path.join(params['model_path'], 'encoder.keras'))
    else:
        model = tf.keras.models.load_model(os.path.join(params['model_path'], 'encoder.keras'))
        model.fit(
            dataset,
            epochs=params['num_epochs'],
            steps_per_epoch=data.__len__(),
            callbacks=[ProjectEncoder(params)],
        )
        model.save(os.path.join(params['model_path'], 'encoder.keras'))
