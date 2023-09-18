import os
import pytest
import numpy as np
import tensorflow as tf
from bp import parse_newick
from biom import load_table
from dataset_utils import get_sentinel_nodes, get_cached_sent_dist, dataset_creater, DataLoader

def get_data():
    with open('scale_16s/test-data/test-tree.nwk', 'r') as f:
        tree = parse_newick(f.readline())
    return {
    'tree': tree,
    'num_sent': 3,
    'table_16s': load_table('scale_16s/test-data/test-table-16s.biom'),
    'table_wgs': load_table('scale_16s/test-data/test-table-wgs.biom'),
    'num_rarefied': 1,
    's_depth_16s': 4,
    's_depth_wgs': 4,
    'batch_size':1,
    'fixed_len': 4}

def test_get_sentinel_nodes():
    data = get_data()
    tree = data['tree']
    num_sent = data['num_sent']
    sent_nodes, tip_info = get_sentinel_nodes(tree, num_sent)
    assert ['i1', 't6', 't9'] == [tree.name(node) for node in sent_nodes]
    assert {
        't0': 1,
        't1': 2,
        't2': 4,
        't3': 6,
        't4': 7,
        't5': 9,
        't6': 10,
        't7': 11,
        't8': 14,
        't9': 15,
    } == tip_info
    print('Passed: test_get_sentinel_nodes')

def test_dataset_creater():
    np.random.seed(4)
    tf.random.set_seed(4)
    data = get_data()
    sent, clas = dataset_creater(**data)
    data['sent_dist_matrices'] = sent
    data['classes'] = clas
    d_load = DataLoader(**data)
    dist_mat, c = d_load.__getitem__(0)
    true_value = tf.constant([[
        [2., 9., 15.5],
        [1., 8., 14.5],
        [8.5, 9.5, 20.],
        [6., 3., 17.5]
    ]], dtype=tf.float64)
    assert tf.math.reduce_all(tf.equal(true_value,dist_mat)).numpy()
    print('Passed: test_dataset_creater')

if __name__ == '__main__':
    test_get_sentinel_nodes()
    test_dataset_creater()