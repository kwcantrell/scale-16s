from bp import parse_newick
from skbio import read, TreeNode
from biom import load_table
from biom.util import biom_open
import numpy as np
import pandas as pd
from skbio import read, write, TreeNode
import os
import os
import gzip
import json
from collections import defaultdict

def _print_table_info(table, table_type):
    print(table_type,
          np.min(table.pa(inplace=False).sum(axis='sample')),
          np.mean(table.pa(inplace=False).sum(axis='sample')),
          np.max(table.pa(inplace=False).sum(axis='sample')),
          table.shape)
    # print(table_type, table.reduce(), table.shape)

# store these for latter
with open('data/phylo-emp500-tree.nwk') as f:
    tree = parse_newick(f.readline())
    print(tree)

# STEP_SIZE=278
# int_since_lat_sent=0
# sentinels = []
# tip_info = {}

# for i in range(1, len(tree)):
#     node = tree.postorderselect(i)

#     if not tree.isleaf(node):
#         if int_since_lat_sent < STEP_SIZE:
#             int_since_lat_sent += 1
#         else:
#             sentinels.append(node)
#             int_since_lat_sent = 0
#     else:
#         name = tree.name(node)
#         tip_info[f'{name}'] = node
    

# cached_dist = [[None] * len(sentinels) for _ in range(len(tree)*2)]
# for indx in range(len(sentinels)):
#     cur_node = sentinels[indx]
#     # seed distances to root while cachinng distance
#     # dist = tree.length(cur_node)
#     dist = 0
#     while cur_node != tree.root():
#         dist += tree.length(cur_node)
#         cached_dist[cur_node][indx] = -2*tree.length(cur_node)
#         cur_node = tree.parent(cur_node)
    
#     # cache root node
#     # dist should be the distance between the root and sentinel[indx]
#     cached_dist[cur_node][indx] = dist

#     # perform a preorder traversal (skip root)
#     for node in range(1, len(tree)):
#         cur_node = tree.preorderselect(node)

#         # grab parent
#         parent = tree.parent(cur_node)
        
#         # get distance
#         dist = cached_dist[parent][indx]
#         dist += tree.length(cur_node)

#         # make sure to check for the seeded distances
#         if cached_dist[cur_node][indx] is not None and cached_dist[cur_node][indx] <= 0:
#             dist += cached_dist[cur_node][indx]

#         cached_dist[cur_node][indx] = dist

#     print(f'finished {indx}')

# def get_dist_to_sent(table):
#     table_ids = table.ids(axis='observation')   
#     # get 16s and wgs features
#     nodes = []
#     sent_dist = []
#     names = []
#     postorder = []
#     for name in table_ids:
#         try:
#             node = tip_info[f'{name}']
#             nodes.append(node)
#             postorder.append(tree.postorder(node))
#             names.append(tree.name(node))
#             sent_dist.append(cached_dist[node])
#         except:
#             pass
#     sort = np.argsort(np.asarray(postorder))
#     nodes = np.asarray(nodes, dtype=np.int32)[sort]
#     names = np.asarray(names)[sort]
#     sent_dist = np.asarray(sent_dist, dtype=np.float32)[sort, :]
#     print(sent_dist, sent_dist.shape)
#     return nodes, sent_dist, names

# # ## load tables ##
# table_path_16s = 'data/phylo-emp500-16s.biom'
# table_16s = load_table(table_path_16s)
# nodes_16s, sent_dist_16s, names_16s = get_dist_to_sent(table_16s)

# table_path_wgs = 'data/phylo-emp500-wgs.biom'
# table_wgs = load_table(table_path_wgs)
# nodes_wgs, sent_dist_wgs, names_wgs = get_dist_to_sent(table_wgs)
# # print(names_16s)

# np.savez_compressed('data/phylo-emp500-sentinel-tree-info.npz',
#          nodes_16s=nodes_16s, sent_dist_16s=sent_dist_16s, names_16s=names_16s,
#          nodes_wgs=nodes_wgs, sent_dist_wgs=sent_dist_wgs, names_wgs=names_wgs)

