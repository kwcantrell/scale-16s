from bp import parse_newick
from skbio import read, TreeNode
from biom import load_table
from biom.util import biom_open
import numpy as np
import pandas as pd
from skbio import read, write, TreeNode
import os
import gzip
import json
from collections import defaultdict
import unittest

def pick_sentinels(tree):
    STEP_SIZE=1
    int_since_lat_sent=0
    sentinels = []
    tip_info = {}

    for i in range(1, len(tree)):
        node = tree.postorderselect(i)

        if not tree.isleaf(node):
            if int_since_lat_sent < STEP_SIZE:
                int_since_lat_sent += 1
            else:
                sentinels.append(node)
                int_since_lat_sent = 0
        else:
            name = tree.name(node)
            tip_info[f'{name}'] = node
    return sentinels, tip_info

def createCachedDist(sentinels, tree):
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
    return cached_dist


def embedding(table, tree, tip_info, cached_dist):
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


###Running phylo-emp500:
with open('data/phylo-emp500-tree.nwk') as f:
    tree = parse_newick(f.readline())
    print(tree)

table = load_table('data/phylo-emp500-wgs.biom')

sentinels, tip_info = pick_sentinels(tree)
print('1')
cached_dist = createCachedDist(sentinels, tree)
print('2')
embedding_result = embedding(table, tree, tip_info, cached_dist)
print('3')

np.savez_compressed('data/phylo-emp500-embedding.npz',
        embedding=embedding_result)