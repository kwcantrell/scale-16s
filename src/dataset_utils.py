import numpy as np
from biom import load_table
from collections import defaultdict
from bp import parse_newick
from find_sentinel import get_cached_sent_dist

class DatasetCreater():
    def __init__(self, **kwargs):
        """
        args: table_16s_path, table_wgs_path, tree_path
        """
        # extract sample vectors
          # need file paths to biom table(s) and tree
          # need 16s table and wgs table
        table_16s = load_table(kwargs['table_16s_path'])
        table_wgs = load_table(kwargs['table_16s_path']) # CHANGE TO WGS
        sample_ids = [id for id in table_16s.ids()] + [id + 'a' for id in table_wgs.ids()]
        sample_ids = {id:i for i, id in enumerate(sample_ids)}

        with open(kwargs['tree_path'], 'r') as f:
            tree = parse_newick(f.readline())

        # rename obs/change order to postorder positions
        # get postorder position of observations in tree
        ids_to_postorder = {}
        for postorder in range(1, len(tree)):
            node = tree.postorderselect(postorder)
            if tree.isleaf(node): 
                name = tree.name(node)
                ids_to_postorder[name] = postorder
        sort_f = lambda ids: sorted(ids, key=lambda id: ids_to_postorder[id])
        table_16s = table_16s.sort(sort_f=sort_f, axis='observation')
        table_wgs = table_wgs.sort(sort_f=sort_f, axis='observation')

        # create a bunch of rarified samples
        if 'num_rarefied' not in kwargs:
            kwargs['num_rarefied'] = 1000
        if 'dep_16s' not in kwargs:
            kwargs['dep_16s'] = 1000
        if 'dep_wgs' not in kwargs:
            kwargs['dep_wgs'] = 500000

        # remove samples with less that sample_depths
        def filter_samples(table, s_depth):
            sums = table.sum(axis='sample')
            sums = sums >= s_depth
            table.filter(table.ids(axis='sample')[sums])
        filter_samples(table_16s, kwargs['dep_16s'])
        filter_samples(table_wgs, kwargs['dep_wgs'])

        tables = []
        for _ in range(kwargs['num_rarefied']):
            tables.append(table_16s.subsample(n=kwargs['dep_16s']))
            tables.append(table_wgs.subsample(n=kwargs['dep_wgs']))

        # calculate cached dist
        cached_dist, tip_info = get_cached_sent_dist(kwargs['tree_path']) # need to implement

        # create helper dist vector
          # need # sent nodes
        # for each sample vec, calc sent-dist-matrix
        self.sent_dist_matrices = []
        self.sample_classes = []
        for table in tables:
            samples = defaultdict(list)
            o_ids = table.ids(axis='observation')
            s_ids = table.ids(axis='sample')
            data = table.matrix_data
            for (feat, samp), obs in data.todok(True).items():
                if obs > 0:
                    samples[s_ids[samp]].append(cached_dist[tip_info[o_ids[feat]]])
            for s_id, dist_mat in samples.items():
                self.sent_dist_matrices.append(np.array(dist_mat))
                self.sample_classes.append(sample_ids[s_id])

    def get_data(self):
        return self.sent_dist_matrices, self.sample_classes

if __name__ == '__main__':
    DatasetCreater(table_16s_path='test-data/test-table.biom',
                tree_path='test-data/test-tree.nwk', num_rarefied=4,
                dep_16s=4, dep_wgs=4)

# class DatasetCreater():
#     def __init__(self, prefix_path):
#         self.prefix_path = prefix_path
        
#         self.table_16s = load_table(f'data/{self.prefix_path}-16s.biom')
#         self.table_wgs = load_table(f'data/{self.prefix_path}-wgs.biom')
#         self.dist_info = np.load('data/phylo-emp500-sentinel-tree-info.npz',
#             allow_pickle=True)

#         def normailize_distances(dists):
#             dists -= np.mean(dists)
#             dists /= np.max(np.abs(dists))
#             return dists
            
#         self.sent_dist_16s = normailize_distances(self.dist_info['sent_dist_16s'])
#         self.sent_dist_wgs = normailize_distances(self.dist_info['sent_dist_wgs'])

#         # align table to sort features in postorder traversal
#         self.table_16s.filter(self.dist_info['names_16s'], axis='observation', inplace=True)
#         self.table_16s.remove_empty(axis='sample', inplace=True)
#         self.table_16s.remove_empty(axis='observation', inplace=True)
#         self.table_16s = self.table_16s.sort_order(self.dist_info['names_16s'], axis='observation')
#         self.table_16s.norm(axis='sample', inplace=True)

#         self.table_wgs.filter(self.dist_info['names_wgs'], axis='observation', inplace=True)
#         self.table_wgs.remove_empty(axis='sample', inplace=True)
#         self.table_wgs.remove_empty(axis='observation', inplace=True)
#         self.table_wgs = self.table_wgs.sort_order(self.dist_info['names_wgs'], axis='observation')
#         self.table_wgs.norm(axis='sample', inplace=True)

#         self.obs_16s, self.abund_16s = self._get_feature_tokens(self.table_16s)
#         self.obs_wgs, self.abund_wgs = self._get_feature_tokens(self.table_wgs)

#         # get class info
#         metadata_16s = pd.read_csv(f'data/{prefix_path}-16s-metadata.txt', sep='\t', index_col=0)
#         metadata_wgs = pd.read_csv(f'data/{prefix_path}-wgs-metadata.txt', sep='\t', index_col=0)
#         labels_16s = self._get_labels(metadata_16s, self.table_16s)
#         labels_wgs = self._get_labels(metadata_wgs, self.table_wgs)
        
#         self.unique_classes = set(labels_16s + labels_wgs)
#         self.num_unique_classes = len(self.unique_classes)
#         self.class_to_token = {c:i for i, c in enumerate(self.unique_classes)}
#         self.labels_16s = [self.class_to_token[l] for l in labels_16s]
#         self.labels_wgs = [self.class_to_token[l] for l in labels_wgs]
        
#         # split samples into their repective classes ##
#         self.data = [list() for _ in range(self.num_unique_classes)]
#         for (sample, abdun, label) in zip(self.sent_dist_16s, self.abund_16s, self.labels_16s):
#             self.data[label].append((sample, abdun))

#         for (sample, abund, label) in zip(self.sent_dist_wgs, self.abund_wgs, self.labels_wgs):
#             self.data[label].append((sample, abund))

#     def _get_feature_tokens(self, table):
#         data = table.matrix_data
#         samples = defaultdict(list)
#         abundance = defaultdict(list)
#         for (feat, samp), obs in data.todok(True).items():
#             samples[int(samp)].append(int(feat))
#             abundance[int(samp)].append(obs)
#         return ([samples[i] for i in range(table.shape[1])],
#                [abundance[i] for i in range(table.shape[1])])


#     def _get_labels(self, metadata, table):
#         return metadata.loc[table.ids(), 'empo_4'].values.tolist()

#     def get_data(self):    
#         return self.data

#     def get_total_tokens(self):
#         return self.tolal_unique_feat + self.additional_tokens
    
#     def get_token_to_class(self):
#         return {t:c for c, t in self.class_to_token.items()}

#     def get_16s_start_token(self):
#         return self.additional_tokens

#     def get_wgs_start_token(self):
#         return self.additional_tokens + self.num_feat_16s



