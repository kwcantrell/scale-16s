import pandas as pd

def align_tables(table_16s, table_wgs, tree):
    # align tables to metadata
    s_ids = set(pd.read_csv('data/emp500-16s-metadata.txt', sep='\t', index_col=0).index)
    table_ids = set(table_16s.ids())
    table_ids = table_ids  & set(table_wgs.ids())
    overlap = table_ids & s_ids
    table_16s.filter(overlap, axis='sample', inplace=True)
    table_wgs.filter(overlap, axis='sample', inplace=True)

    # align tables to tree
    names = {tree.name(i) for i, v in enumerate(tree.B) if v}
    def filter_obs(table, names):
        obs_ids = set(table.ids(axis='observation'))    
        overlap = obs_ids & names
        table = table.filter(overlap, axis='observation', inplace=False).remove_empty()
        return table
    table_16s = filter_obs(table_16s, names)
    table_wgs = filter_obs(table_wgs, names)
    return table_16s, table_wgs

def get_tip_ids(tree):
    tip_names = []
    for i in range(1, len(tree) + 1):
        node = tree.postorderselect(i)
        if tree.isleaf(node):
            tip_names.append(tree.name(node))
    return tip_names

def get_sentinel_nodes(tree, num_sent):
    # this block of code extracts the sentinels
    # step size determines how often to sample a sentinel node.
    # STEP_SIZE= int(len(tree)*0.1)# make this # 10% of the total tree size\. so for example with 1000 tips, make this number 100
    step_size= int(len(tree)/num_sent)# make this # 10% of the total tree size\. so for example with 1000 tips, make this number 100
    int_since_lat_sent=1
    sentinels = []
    tip_info = {}
    for i in range(1, len(tree) + 1):
        node = tree.postorderselect(i)

        if int_since_lat_sent % step_size == 0:
            sentinels.append(node)

        int_since_lat_sent += 1
        
        if tree.isleaf(node):
            name = tree.name(node)
            tip_info[f'{name}'] = i
    return  sentinels, tip_info

def get_cached_sent_dist(tree, num_sent):
    sentinels, tip_info = get_sentinel_nodes(tree, num_sent)
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

def get_tip_info(tree):
    tip_info = {}
    for i in range(1, len(tree) + 1):
        node = tree.postorderselect(i)
        if tree.isleaf(node):
            name = tree.name(node)
            tip_info[f'{name}'] = i
    return tip_info
