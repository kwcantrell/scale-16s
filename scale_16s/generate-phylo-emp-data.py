from unifrac import faith_pd, weighted_normalized_fp64
from biom import load_table
from biom.util import biom_open
import numpy as np
import pandas as pd
from skbio import read, write, TreeNode
import os
import matplotlib.pyplot as plt

def _print_table_info(table, table_type):
    print(table_type,
          np.min(table.pa(inplace=False).sum(axis='sample')),
          np.mean(table.pa(inplace=False).sum(axis='sample')),
          np.max(table.pa(inplace=False).sum(axis='sample')),
          table.shape)
    # print(table_type, table.reduce(), table.shape)

def show_s_depth_hist(table):
    counts, bins = np.histogram(table.pa(inplace=False).sum(axis='sample'), bins=100, range=(0,2048))
    print(counts, bins)
    print(['{i:.2f}'.format(i=i) for i in np.cumsum(counts)/10486], np.max(counts))
    # bins = [i for i in range(2048, 4096, 1000)]
    plt.hist(counts, bins)
    plt.show()

def show_fcount_hist(table):
    counts, bins = np.histogram(table.sum(axis='observation'), bins=100, range=(0, 100000))
    print(counts, bins)
    print(['{i:.2f}'.format(i=i) for i in np.cumsum(counts)/10495], np.max(counts))
    # bins = [i for i in range(2048, 4096, 1000)]
    plt.hist(counts, bins)
    plt.show()

def drop_max_s_count(table):
    s_sums = table.pa(inplace=False).sum(axis='sample') <= 2048
    s_ids = table.ids()[s_sums]
    table = table.filter(s_ids, inplace=False)
    return table

def print_sample_counts(table, table_type):
    print(table_type, np.sort(table.sum(axis='sample')))

def save_table(table, path):
    with biom_open(path, 'w') as f:
        table.to_hdf5(f, 'scale-16s-training')
    return

def min_sample_depth(table, depth=1000):
    s_sums = table.sum(axis='sample')
    s_sums = s_sums > depth
    s_ids = table.ids()[s_sums]
    table = table.filter(s_ids, axis='sample', inplace=False)
    table = table.remove_empty(axis='observation', inplace=False)
    return table

def remove_single_tons(table):
    f_sum = table.pa(inplace=False).sum(axis='observation') > 1
    f_ids = table.ids(axis='observation')[f_sum]
    table = table.filter(f_ids, axis='observation')
    return table

def min_feature_count(table, min_frac=0.001):
    s_sums = table.sum(axis='sample')

    f_sums = table.sum(axis='observation')
    f_sums = f_sums > np.mean(s_sums)*min_frac

    o_ids = table.ids(axis='observation')[f_sums]
    table = table.filter(o_ids, axis='observation', inplace=False)
    return table

def get_sample_types(metadata):
    types = {
        'rename': {
            'sample_type': { # sample_type => env_package
                'milk': 'human-milk',
                'breast milk': 'human-milk',
                'urine': 'human-urine',
                'soil': 'soil',
            }
        },
        'env_package': ['human-gut','human-oral','human-skin', 'human-milk', 'human-urine', 'soil'],
        # 'sample_type': ['urine', 'soil'],# 'milk', 'breast milk']
    }
    for st, env in types['rename']['sample_type'].items():
        print(st, env)
        metadata.loc[metadata['sample_type'] == st, 'env_package'] = env
    metadata = metadata.loc[metadata['env_package'].isin(types['env_package'])]
    return metadata

def align_to_tree(table):
    # align table to tree
    with open("data/tree.nwk", 'r') as f:
        tree = read(f, format='newick', into=TreeNode)
    table, tree = table.align_tree(tree)
    with open('phylo-emp500-tree.nwk', 'w') as f:
        tree.write(f)
    print(table.shape)
    return table

# ## load tables ##
table_path_16s = 'data/phylo-emp500-16s.biom'
table_16s = load_table(table_path_16s)
table_ids_16s = table_16s.ids()
_print_table_info(table_16s, '16s')
# print(table_16s.ids(), '16s')

table_path_wgs = 'data/phylo-emp500-wgs.biom'
table_wgs = load_table(table_path_wgs)
table_ids_wgs = table_wgs.ids()
_print_table_info(table_wgs, 'wgs')

merged_table = table_16s.concat(table_wgs, axis='observation')
_print_table_info(merged_table, 'merged')
merged_table = align_to_tree(merged_table)
_print_table_info(merged_table, 'merged')

# ## get samples ##
# metadata_16s = pd.read_csv('data/emp500-16s-metadata.txt', sep='\t', index_col=0)
# print(metadata_16s['empo_v2_4b'].value_counts())
# # metadata_16s = metadata_16s.loc[metadata_16s.index.isin(table_ids_16s)]
# # metadata_16s = get_sample_types(metadata_16s)
# # ids_to_keep_16s = metadata_16s.index.tolist()

# metadata_wgs = pd.read_csv('data/emp500-wgs-metadata.txt', sep='\t', index_col=0)
# print(metadata_wgs['empo_v2_4b'].value_counts())
# # metadata_wgs = metadata_wgs.loc[metadata_wgs.index.isin(table_ids_wgs)]
# # metadata_wgs = get_sample_types(metadata_wgs)
# # ids_to_keep_wgs = metadata_wgs.index.tolist()

# ## filter tables to only samples ##
# table_16s.filter(ids_to_keep_16s)
# _print_table_info(table_16s, '16s')
# # print_sample_counts(table_16s, '16s')

# table_wgs.filter(ids_to_keep_wgs)
# _print_table_info(table_wgs, 'wgs')
# # print_sample_counts(table_wgs, 'wgs')

# table_16s = remove_single_tons(table_16s)
# _print_table_info(table_16s, '16s')

# table_wgs = remove_single_tons(table_wgs)
# _print_table_info(table_wgs, 'wgs')

# table_16s = drop_max_s_count(table_16s)
# _print_table_info(table_16s, '16s')

# table_wgs = drop_max_s_count(table_wgs)
# _print_table_info(table_wgs, 'wgs')

# save_table(table_16s, 'data/emp500-16s.biom')
# save_table(table_wgs, 'data/emp500-wgs.biom')

# ## remove 16s samples with less than 1000 sample depth ##
# table_16s = min_sample_depth(table_16s, 1000)
# _print_table_info(table_16s, '16s')

# ## remove wgs samples with less than 100000 sample depth
# table_wgs = min_sample_depth(table_wgs, 100000)
# _print_table_info(table_wgs, 'wgs')


# show_s_depth_hist([table_16s, table_wgs])
# print_s_type(table_16s, metadata_16s)


## remove low count features ##
# table_16s = min_feature_count(table_16s, 0.001)
# _print_table_info(table_16s, '16s')

# # table_wgs = min_feature_count(table_wgs, 0.001)
# # _print_table_info(table_wgs, 'wgs')

# ## rarefy tables ##
# table_16s = table_16s.subsample(5000)
# save_table(table_16s, 'data/rarefied-table-16s.biom')
# _print_table_info(table_16s, '16s')

# table_wgs = table_wgs.subsample(100000)
# save_table(table_wgs, 'data/rarefied-table-wgs.biom')
# _print_table_info(table_wgs, 'wgs')
