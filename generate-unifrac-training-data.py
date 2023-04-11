from unifrac import faith_pd, weighted_normalized_fp64
from biom import load_table
from biom.util import biom_open
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from skbio import read, write, TreeNode
import os

root_path='unifrac-dataset/'
table_path_16s=os.path.join(root_path, '16s.biom')
table_path_wgs=os.path.join(root_path, 'wgs.biom')
table_16s = load_table(table_path_16s)
print(table_16s.shape)
table_wgs = load_table(table_path_wgs)
print(table_wgs.shape)

# get random 200 gut, skin, oral samples
metadata = pd.read_csv('unifrac-dataset/wgs-metadata.txt', sep='\t', index_col=0)
metadata = metadata.loc[metadata.index.isin(table_wgs.ids(axis='sample'))]
gut_ids = metadata.loc[metadata['env_package'] == 'human-gut'].sample(n=200, random_state=1).index.tolist()
oral_ids = metadata.loc[metadata['env_package'] == 'human-oral'].sample(n=200, random_state=1).index.tolist()
skin_ids = metadata.loc[metadata['env_package'] == 'human-skin'].sample(n=200, random_state=1).index.tolist()
ids_to_keep = gut_ids + skin_ids + oral_ids

# filter tables
table_16s.filter(ids_to_keep)
table_16s.remove_empty(axis='observation')
print(table_16s.shape)
table_wgs.filter(ids_to_keep)
table_wgs.remove_empty(axis='observation')
print(table_wgs.shape)

# combine tables and align with tree
id_map = {s_id: s_id + '.16s' for s_id in table_16s.ids()}
table_16s.update_ids(id_map)
id_map = {s_id: s_id + '.wgs' for s_id in table_wgs.ids()}
table_wgs.update_ids(id_map)
# unfiltered_merged_table = table_16s.concat(table_wgs, axis='observation')
# print(unfiltered_merged_table.shape)
# with open('unifrac-dataset/shared-trees/tree-asv.nwk', 'r') as f:
#     tree = read(f, format='newick', into=TreeNode)
# aligned_merged_table, aligned_tree = unfiltered_merged_table.align_tree(tree)

# # tree.write('unifrac-dataset/shared-trees/aligned-tree.nwk', format='newick')
# def save_table(table, path):
#     with biom_open(path, 'w') as f:
#         table.to_hdf5(f, 'scale-16s-training')
#     return
# # save_table(aligned_merged_table, 'unifrac-dataset/training/data/aligned-merged.biom')

# # split tables and filter features
# al_table = load_table('unifrac-dataset/training/data/aligned-merged.biom')
# print(al_table.shape)
# table_16s = al_table.filter(table_16s.ids(), inplace=False)
# table_16s.remove_empty(axis='observation')
# print(np.sort(table_16s.sum(axis='sample')))
# table_wgs = al_table.filter(table_wgs.ids(), inplace=False)
# table_wgs.remove_empty(axis='observation')
# print(np.sort(table_wgs.sum(axis='sample')))

# def min_feature_count(table, min_frac=0.001):
#     s_sums = table.sum(axis='sample')

#     f_sums = table.sum(axis='observation')
#     f_sums = f_sums > np.mean(s_sums)*min_frac

#     o_ids = table.ids(axis='observation')[f_sums]
#     table = table.filter(o_ids, axis='observation', inplace=False)
#     print(np.median(table.pa(inplace=False).sum(axis='sample')))
#     return table
# table_16s = min_feature_count(table_16s, 0.001)
# table_wgs = min_feature_count(table_wgs, 0.001)

# # merge tables again
table_path = 'unifrac-dataset/training/data/filtered-merged.biom'
table = load_table(table_path)
print(table.shape)
table.remove_empty(axis='observation')
print(table.shape)
# filter_merged_table = table_16s.concat(table_wgs, axis='observation')
# save_table(filter_merged_table, table_path)

# # create input data
# tree_path = 'unifrac-dataset/shared-trees/aligned-tree.nwk'
# distance = weighted_normalized_fp64(table_path, tree_path)
# s_ids = distance.ids
# print(len(s_ids))
# table = filter_merged_table
# MAX_OBS_PER_SAMPLE=int(np.max(table.pa(inplace=False).sum(axis='sample')))
# print(MAX_OBS_PER_SAMPLE)

# def write_lines(lines, path):
#     with open(path, 'w') as f:
#         for line in lines:
#             dist, (s1_obs, s1_counts), (s2_obs, s2_counts) = line
#             f.write('{:.4f}\t'.format(dist))
#             f.write('{}\t'.format(','.join(['%i']*len(s1_obs)) % tuple(s1_obs)))
#             f.write('{}\t'.format(','.join(['%i']*len(s1_counts)) % tuple(s1_counts) ))
#             f.write('{}\t'.format(','.join(['%i']*len(s2_obs)) % tuple(s2_obs) ))
#             f.write('{}\n'.format(','.join(['%i']*len(s2_counts)) % tuple(s2_counts) ))

# def extract_sample_info(table, s_id):
#     s_data = table.data(s_id, dense=False)
#     s_obs = [i + 1 for i in s_data.indices]
#     s_counts = s_data.data
#     return s_obs, s_counts


# lines_per_file=1000
# cur_file=0
# lines = []
# for i in range(len(s_ids)):
#     s1 = s_ids[i]
#     s1_info = extract_sample_info(table, s1)

#     for j in range(i+1, len(s_ids)):
#         s2 = s_ids[j]
#         s2_info = extract_sample_info(table, s2)
#         dist = distance[s1, s2]

#         if len(lines) >= lines_per_file:
#             write_lines(lines, os.path.join(root_path, 'training/input/wu-{:n}'.format(cur_file)))
#             cur_file += 1
#             lines = []

#         lines.append((dist, s1_info, s2_info))
# if len(lines) > 0:
#     write_lines(lines, os.path.join(root_path, 'training/input/wu-{:n}'.format(cur_file)))
