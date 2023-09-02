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
import unittest

def _print_table_info(table, table_type):
    print(table_type,
          np.min(table.pa(inplace=False).sum(axis='sample')),
          np.mean(table.pa(inplace=False).sum(axis='sample')),
          np.max(table.pa(inplace=False).sum(axis='sample')),
          table.shape)
    # print(table_type, table.reduce(), table.shape)

# store these for latter
with open('data/sam-tree.nwk') as f:
    tree = parse_newick(f.readline())
    print(tree)

STEP_SIZE=1
int_since_lat_sent=0
sentinels = []
tip_info = {}

print(len(tree))

for i in range(1, len(tree)):
    node = tree.postorderselect(i)

    if not tree.isleaf(node):
        
        print(tree.name(node), 'not leaf')
        if int_since_lat_sent < STEP_SIZE:
            int_since_lat_sent += 1
        else:
            sentinels.append(node)
            int_since_lat_sent = 0
    else:
        print(tree.name(node), 'leaf')
        name = tree.name(node)
        tip_info[f'{name}'] = node

print("Sentinels: ")
print([tree.name(s) for s in sentinels])
    

cached_dist = [[None] * len(sentinels) for _ in range(len(tree)*2)]
for indx in range(len(sentinels)):
    cur_node = sentinels[indx]
    # seed distances to root while cachinng distance
    # dist = tree.length(cur_node)
    dist = 0
    while cur_node != tree.root():
        dist += tree.length(cur_node)
        cached_dist[cur_node][indx] = -2*tree.length(cur_node)
        cur_node = tree.parent(cur_node)
    
    # cache root node
    # dist should be the distance between the root and sentinel[indx]
    cached_dist[cur_node][indx] = dist

    # perform a preorder traversal (skip root)
    for node in range(1, len(tree)):
        cur_node = tree.preorderselect(node)

        # grab parent
        parent = tree.parent(cur_node)
        
        # get distance
        dist = cached_dist[parent][indx]
        dist += tree.length(cur_node)

        # make sure to check for the seeded distances
        if cached_dist[cur_node][indx] is not None and cached_dist[cur_node][indx] <= 0:
            dist += cached_dist[cur_node][indx]

        cached_dist[cur_node][indx] = dist

    print(f'finished {indx}')

print("Cached dist: ")
print(cached_dist)

def get_dist_to_sent(table):
    table_ids = table.ids(axis='observation')   
    # get 16s and wgs features
    nodes = []
    sent_dist = []
    names = []
    postorder = []
    for name in table_ids:
        try:
            node = tip_info[f'{name}']
            nodes.append(node)
            postorder.append(tree.postorder(node))
            names.append(tree.name(node))
            sent_dist.append(cached_dist[node])
        except:
            pass
    sort = np.argsort(np.asarray(postorder))
    nodes = np.asarray(nodes, dtype=np.int32)[sort]
    names = np.asarray(names)[sort]
    sent_dist = np.asarray(sent_dist, dtype=np.float32)[sort, :]
    print(sent_dist, sent_dist.shape)
    return nodes, sent_dist, names

def embedding(table, tree):
    observation_ids = table.ids(axis='observation')   
    sample_ids = table.ids(axis='sample')
    # get 16s and wgs features
    nodes = []
    sent_dist = []
    names = []
    postorder = []
    for name in observation_ids:
        try:
            node = tip_info[f'{name}']
            nodes.append(node)
            postorder.append(tree.postorder(node))
            names.append(tree.name(node))
            sent_dist.append(cached_dist[node])
        except:
            pass
    sort = np.argsort(np.asarray(postorder))
    nodes = np.asarray(nodes, dtype=np.int32)[sort]
    names = np.asarray(names)[sort]
    sent_dist = np.asarray(sent_dist, dtype=np.float32)[sort, :]
    embedding = []
    num_sentinels = len(sent_dist[0])
    for i in range(len(sample_ids)):
        sample_embedding = []
        biom_column = table.data(sample_ids[i])
        for j in range(len(biom_column)):
            if biom_column[j] == 0:
                sample_embedding.append(list(np.zeros(num_sentinels)))
            else:
                sample_embedding.append(list(sent_dist[j]))
        embedding.append(sample_embedding)
    return embedding


##TESTING
table_path_test = 'data/test-table.biom'
table_test = load_table(table_path_test)
nodes_test, sent_dist_test, names_test = get_dist_to_sent(table_test)
print('nodes_test, sent_dist_test, names_test:')
print(nodes_test, sent_dist_test, names_test)
sent_dist_test[0] = list(np.zeros(3))
print(sent_dist_test)

embedding1 = embedding(table_test, tree)
print(embedding1)

embedding1_solution = [
                        [[0,0,0],
                        [1,6,3.5],
                        [0,0,0],
                        [1,6,3.5],
                        [0,0,0],
                        [8.5,7.5,9],
                        [0,0,0],
                        [7,2,7.5],
                        [6.5,1.5,7],
                        [11.5,14.5,9]],

                        [[1.5,6.5,4],
                        [1,6,3.5],
                        [2,7,4.5],
                        [0,0,0],
                        [7.5,6.5,8],
                        [0,0,0],
                        [6,1,6.5],
                        [7,2,7.5],
                        [6.5,1.5,7],
                        [0,0,0]],

                        [[1.5,6.5,4],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [7.5,6.5,8],
                        [0,0,0],
                        [6,1,6.5],
                        [0,0,0],
                        [6.5,1.5,7],
                        [0,0,0]],

                        [[1.5,6.5,4],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [7.5,6.5,8],
                        [0,0,0],
                        [6,1,6.5],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0]],

                        [[1.5,6.5,4],
                        [0,0,0],
                        [2,7,4.5],
                        [0,0,0],
                        [7.5,6.5,8],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [11.5,14.5,9]]
                    ]

class testEmbedding(unittest.TestCase):
    def runTest(self):
        table_path_test = 'data/test-table.biom'
        table_test = load_table(table_path_test)
        with open('data/sam-tree.nwk') as f:
            tree = parse_newick(f.readline())
        embedding1 = embedding(table_test, tree)
        self.assertEqual(embedding1, embedding1_solution, "incorrect embedding")

#unittest.main()

with open('data/sam-tree.nwk') as f:
    tree = parse_newick(f.readline())
    print(tree)
table_path_test = 'data/test-table.biom'
table_test = load_table(table_path_test)
embedding_emp500 = embedding(table_test, tree)

with open('data/phylo-emp500-tree.nwk') as f:
    tree_emp500 = parse_newick(f.readline())
    print(tree_emp500)
table_path_emp500 = 'data/phylo-emp500-wgs.biom'
table_emp500 = load_table(table_path_emp500)
embedding_emp500 = embedding(table_emp500, tree_emp500)

# table_path_wgs = 'data/phylo-emp500-wgs.biom'
# table_wgs = load_table(table_path_wgs)
# nodes_wgs, sent_dist_wgs, names_wgs = get_dist_to_sent(table_wgs)
# print(names_16s)

np.savez_compressed('data/phylo-emp500-sentinel-tree-info.npz',
         nodes_wgs=nodes_test, sent_dist_wgs=sent_dist_test, names_wgs=names_test)

np.savez_compressed('data/phylo-emp500-embedding.npz',
        embedding=embedding_emp500)

