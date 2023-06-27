import pandas as pd
import numpy as np
from biom import load_table
from biom.util import biom_open
from collections import defaultdict
# import tensorflow as tf

# class DatasetCreater():
#     def __init__(self, prefix_path):
#         self.prefix_path = prefix_path
        
#         self.table_16s = load_table(f'data/{self.prefix_path}-16s.biom')
#         self.table_wgs = load_table(f'data/{self.prefix_path}-wgs.biom')

#         self.token_16s = self._get_feature_tokens(self.table_16s)
#         self.token_wgs = self._get_feature_tokens(self.table_wgs)

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

#         ## map tokens to embedding placements ##
#         self.num_feat_16s = self.table_16s.shape[0]
#         self.num_feat_wgs = self.table_wgs.shape[0]
#         self.tolal_unique_feat = self.num_feat_16s + self.num_feat_wgs

#         # +3 to account for mask postion, start sequence, end sequence
#         self.additional_tokens = 3

#         # 16s tokens are placed first in the embedding matrix
#         self.token_16s = [list(map(lambda x: x + self.additional_tokens, tokens))
#                     for tokens in self.token_16s]

#         # wgs tokens will be placed right after embedding matrix
#         self.token_wgs = [list(map(lambda x: x + self.additional_tokens + self.num_feat_16s, tokens))
#                     for tokens in self.token_wgs]

#         ## split samples into their repective classes ##
#         self.data = [list() for _ in range(self.num_unique_classes)]
#         for (sample, label) in zip(self.token_16s, self.labels_16s):
#             self.data[label].append(sample)

#         for (sample, label) in zip(self.token_wgs, self.labels_wgs):
#             self.data[label].append(sample)


#     def _get_feature_tokens(self, table):
#         data = table.matrix_data
#         samples = defaultdict(list)
#         for feat, samp in data.todok(True).keys():
#             samples[int(samp)].append(int(feat))
#         return [samples[i] for i in range(table.shape[1])]

#     def _get_labels(self, metadata, table):
#         return metadata.loc[table.ids(), 'empo_v2_4b'].values.tolist()

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

class DatasetCreater():
    def __init__(self, prefix_path):
        self.prefix_path = prefix_path
        
        self.table_16s = load_table(f'data/{self.prefix_path}-16s.biom')
        self.table_wgs = load_table(f'data/{self.prefix_path}-wgs.biom')
        self.dist_info = np.load('data/phylo-emp500-sentinel-tree-info.npz',
            allow_pickle=True)

        def normailize_distances(dists):
            dists -= np.mean(dists)
            dists /= np.max(np.abs(dists))
            return dists
            
        self.sent_dist_16s = normailize_distances(self.dist_info['sent_dist_16s'])
        self.sent_dist_wgs = normailize_distances(self.dist_info['sent_dist_wgs'])

        # align table to sort features in postorder traversal
        self.table_16s.filter(self.dist_info['names_16s'], axis='observation', inplace=True)
        self.table_16s.remove_empty(axis='sample', inplace=True)
        self.table_16s.remove_empty(axis='observation', inplace=True)
        self.table_16s = self.table_16s.sort_order(self.dist_info['names_16s'], axis='observation')
        self.table_16s.norm(axis='sample', inplace=True)

        self.table_wgs.filter(self.dist_info['names_wgs'], axis='observation', inplace=True)
        self.table_wgs.remove_empty(axis='sample', inplace=True)
        self.table_wgs.remove_empty(axis='observation', inplace=True)
        self.table_wgs = self.table_wgs.sort_order(self.dist_info['names_wgs'], axis='observation')
        self.table_wgs.norm(axis='sample', inplace=True)

        self.obs_16s, self.abund_16s = self._get_feature_tokens(self.table_16s)
        self.obs_wgs, self.abund_wgs = self._get_feature_tokens(self.table_wgs)

        # get class info
        metadata_16s = pd.read_csv(f'data/{prefix_path}-16s-metadata.txt', sep='\t', index_col=0)
        metadata_wgs = pd.read_csv(f'data/{prefix_path}-wgs-metadata.txt', sep='\t', index_col=0)
        labels_16s = self._get_labels(metadata_16s, self.table_16s)
        labels_wgs = self._get_labels(metadata_wgs, self.table_wgs)
        
        self.unique_classes = set(labels_16s + labels_wgs)
        self.num_unique_classes = len(self.unique_classes)
        self.class_to_token = {c:i for i, c in enumerate(self.unique_classes)}
        self.labels_16s = [self.class_to_token[l] for l in labels_16s]
        self.labels_wgs = [self.class_to_token[l] for l in labels_wgs]
        
        # split samples into their repective classes ##
        self.data = [list() for _ in range(self.num_unique_classes)]
        for (sample, abdun, label) in zip(self.sent_dist_16s, self.abund_16s, self.labels_16s):
            self.data[label].append((sample, abdun))

        for (sample, abund, label) in zip(self.sent_dist_wgs, self.abund_wgs, self.labels_wgs):
            self.data[label].append((sample, abund))

    def _get_feature_tokens(self, table):
        data = table.matrix_data
        samples = defaultdict(list)
        abundance = defaultdict(list)
        for (feat, samp), obs in data.todok(True).items():
            samples[int(samp)].append(int(feat))
            abundance[int(samp)].append(obs)
        return ([samples[i] for i in range(table.shape[1])],
               [abundance[i] for i in range(table.shape[1])])


    def _get_labels(self, metadata, table):
        return metadata.loc[table.ids(), 'empo_4'].values.tolist()

    def get_data(self):    
        return self.data

    def get_total_tokens(self):
        return self.tolal_unique_feat + self.additional_tokens
    
    def get_token_to_class(self):
        return {t:c for c, t in self.class_to_token.items()}

    def get_16s_start_token(self):
        return self.additional_tokens

    def get_wgs_start_token(self):
        return self.additional_tokens + self.num_feat_16s



