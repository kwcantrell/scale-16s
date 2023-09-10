from bp import parse_newick
from biom import load_table
import numpy as np

def _print_table_info(table, table_type):
    print(table_type,
          np.min(table.pa(inplace=False).sum(axis='sample')),
          np.mean(table.pa(inplace=False).sum(axis='sample')),
          np.max(table.pa(inplace=False).sum(axis='sample')),
          table.shape)

def get_cached_sent_dist(tree_path):
    with open(tree_path, 'r') as f:
        tree = parse_newick(f.readline())

    # this block of code extracts the sentinels
    # step size determines how often to sample a sentinel node.
    STEP_SIZE= int(len(tree)*0.1)# make this # 10% of the total tree size\. so for example with 1000 tips, make this number 100
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
            tip_info[f'{name}'] = i

    # calculate distances to sentinels for each node in the tree
    cached_dist = [[None] * len(sentinels) for _ in range(len(tree)+1)]
    for indx in range(len(sentinels)): # iterate over sentinel nodes
        cur_node = sentinels[indx]

        # seed distances to root while cachinng distance
        dist = 0
        while cur_node != tree.root():
            dist += tree.length(cur_node)
            cached_dist[tree.postorder(cur_node)][indx] = -2*tree.length(cur_node)
            cur_node = tree.parent(cur_node)
        
        # cache root node
        # dist should be the distance between the root and sentinel[indx]
        cached_dist[tree.postorder(cur_node)][indx] = dist

        # perform a preorder traversal (skip root)
        for node in range(1, len(tree)):
            cur_node = tree.preorderselect(node)

            # grab parent
            parent = tree.parent(cur_node)
            
            # get distance
            dist = cached_dist[tree.postorder(parent)][indx]
            dist += tree.length(cur_node)

            # make sure to check for the seeded distances
            if cached_dist[tree.postorder(cur_node)][indx] is not None and cached_dist[tree.postorder(cur_node)][indx] <= 0:
                dist += cached_dist[tree.postorder(cur_node)][indx]

            cached_dist[tree.postorder(cur_node)][indx] = dist

    return cached_dist, tip_info
    
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

if __name__ == '__main__':
    print(get_cached_sent_dist('test-data/test-tree.nwk'))

# np.savez_compressed('data/phylo-emp500-sentinel-tree-info.npz',
#          nodes_16s=nodes_16s, sent_dist_16s=sent_dist_16s, names_16s=names_16s,
#          nodes_wgs=nodes_wgs, sent_dist_wgs=sent_dist_wgs, names_wgs=names_wgs)

