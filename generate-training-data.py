from unifrac import faith_pd
from biom import load_table
from biom.util import biom_open
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from skbio import read, write, TreeNode

# ALIGN TABLE TO TREE. THIS TAKES A WHILE #
# table_path = 'samples/16s.biom'
# table = load_table(table_path)
# print(table.shape)
# print(np.mean(table.sum(axis='sample')), np.min(table.sum(axis='sample')))

# # align table to tree
# with open("dataset/filtered-tree.nwk", 'r') as f:
#     tree = read(f, format='newick', into=TreeNode)
#     table, tree = table.align_tree(tree)
# print(table.shape)
# print(np.max(table.sum(axis='sample')), np.min(table.sum(axis='sample')))
# # tree.write('dataset/filtered-tree.nwk', format='newick')
# del tree

# # Save table #
# table_path = 'dataset/16s-filtered.biom'
# with biom_open(table_path, 'w') as f:
#     table.to_hdf5(f, 'scale-16s-training')

# remove all featuress that appear in < 15 samples #
table_path = 'dataset/16s-filtered.biom'
table = load_table((table_path))
print(table.shape)
sums = table.pa(inplace=False).sum(axis='observation')
sums = sums > 15
o_ids = table.ids(axis='observation')[sums]
table.filter(o_ids, axis='observation')
table.remove_empty(axis='sample')
print(table.shape)


# remove all samples with < 5000 counts
sums = table.sum(axis='sample')
print('mean', np.median(sums))
sums = sums >= 5000
s_ids = table.ids(axis='sample')[sums]
table.filter(s_ids)
table.remove_empty(axis='observation')
table.remove_empty(axis='sample')
print(table.shape)

# Save table #
table_path = 'dataset/training/data/16s-full-train.biom'
tree_path = "dataset/filtered-tree.nwk"
with biom_open(table_path, 'w') as f:
    table.to_hdf5(f, 'scale-16s-training')
# print(table.shape)
print(np.max(table.pa(inplace=False).sum(axis='sample')))

# CREATE TRAINING DATA WITH FILTERED TABLE #
# create traing data
MAX_OBS_PER_SAMPLE=int(np.max(table.pa(inplace=False).sum(axis='sample')))
print(MAX_OBS_PER_SAMPLE)
feature_ids = table.ids(axis='observation')
features = {obs:i for i, obs in enumerate(feature_ids,start=1)}
features['0'] = 0
def create_input(table_path, tree_path, output_path):
    # calculate faith_pd on full table
    distance = faith_pd(table_path, tree_path).to_frame()
    print((distance.head()))

    t = load_table(table_path)

    data = []
    for values, id, _ in t.iter(axis='sample', dense=False):
        obs = feature_ids[values.indices]
        if len(obs) == 0:
            continue
        missing = MAX_OBS_PER_SAMPLE - len(obs)
        obs = np.concatenate((obs,[0]*missing))
        counts = values.data.astype(int)
        counts = np.concatenate((counts, [0]*missing))
        dist = distance.loc[id, 'faith_pd']
        entry = [obs, counts, dist]
        data.append(entry)

    with open(output_path, 'w') as f:
        for obs, counts, dist in data:
            f.write('{num_obs:n}\t'.format(num_obs=len(obs)))
            obs_str = ""
            for ob in obs:
                obs_str += '{ob},'.format(ob=features[ob])
            f.write(obs_str[:-1])
            f.write('\t')
            count_str = ""
            for count in counts:
                count_str += '{count:n},'.format(count=count)
            f.write(count_str[:-1])
            f.write('\t')
            f.write('{dist:.4f}\n'.format(dist=dist))
        
table_path = 'dataset/training/data/16s-full-train.biom'
tree_path = "dataset/filtered-tree.nwk"
# create_input(table_path, tree_path, 'dataset/training/input/16s-filter.txt')

tables_per_rare = 5
rare_levels = [5000, 10000, 15000]
for rare_level in rare_levels:
    for i in range(tables_per_rare):
        r_table = table.subsample(rare_level, axis='sample')

        table_path = 'dataset/training/data/16s-{rare_level}-{i}.biom'.format(rare_level=rare_level, i=i)
        with biom_open(table_path, 'w') as f:
            r_table.to_hdf5(f, 'scale-16s-training')
        print(table.shape, r_table.shape)
        output_path = 'dataset/training/input/16s-{rare_level}-{i}.txt'.format(rare_level=rare_level, i=i)
        create_input(table_path, tree_path, output_path)
