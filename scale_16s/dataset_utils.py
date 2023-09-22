import numpy as np
import random
from biom.util import biom_open
import os
from unifrac import unweighted
from utils import get_cached_sent_dist
from biom import Table
from bp import parse_newick
from utils import get_tip_ids, get_tip_info

def generate_random_table(obs, num_samples, sample_start, num_features=None, min_features=None, max_features=None):
    features = []
    total_obs = obs.shape[0]
    feature_range = num_features is None
    shuffle_indx = np.arange(total_obs)
    for i in range(num_samples):
        np.random.shuffle(shuffle_indx)
        if feature_range:
            if max_features is None:
                max_features = total_obs
            num_features = np.random.randint(min_features, max_features)
        indices = shuffle_indx[:num_features]
        features.append(indices)
    
    data = np.zeros((total_obs, num_samples), dtype=np.int32)
    samples = [f'SCALE16S{i}' for i in range(sample_start,  sample_start + num_samples)]
    for i, indices in enumerate(features):
        data[indices, i] = 1

    return Table(data, obs, samples), samples

# with open('data/small-test/phylo-emp500-tree.nwk', 'r') as f:
#     tree = parse_newick(f.readline())
# print(tree)
# feature_ids = np.array(get_tip_ids(tree))
# samples_per_table = 64
# cur_table_inx = 0
# min_features = 256
# max_features = [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 8192]
# for mf in max_features:
#     for _ in range(5):
#         table,_ = generate_random_table(feature_ids, samples_per_table,
#                                     sample_start=cur_table_inx*samples_per_table,
#                                     min_features=min_features, max_features=mf)

#         print(cur_table_inx, table.shape)
#         with biom_open(f"random-emp500-tables/table-{cur_table_inx}.biom", 'w') as f:
#                 table.to_hdf5(f, 'scale-16s-training')
#         cur_table_inx += 1


def dataset_creater(**kwargs):
    """
    args: table_16s, table_wgs, tree, num_rarefied, s_depth_16s, s_depth_wgs
    """
    table_16s = kwargs['table_16s']
    table_wgs = kwargs['table_wgs']
    tree = kwargs['tree']

    # change order of observations in biom table postorder positions
    ids_to_postorder = {}
    for postorder in range(1, len(tree)):
        node = tree.postorderselect(postorder)
        if tree.isleaf(node): 
            name = tree.name(node)
            ids_to_postorder[name] = postorder
    sort_f = lambda ids: sorted(ids, key=lambda id: ids_to_postorder[id])
    table_16s = table_16s.sort(sort_f=sort_f, axis='observation')
    table_wgs = table_wgs.sort(sort_f=sort_f, axis='observation')

    # remove samples with less that sample_depths
    print('filtering tables...')
    def filter_samples(table, s_depth):
        sums = table.sum(axis='sample')
        sums = sums >= s_depth
        return table.filter(table.ids(axis='sample')[sums], inplace=False).remove_empty()
        
    def filter_observations(table):
        sums = table.sum(axis='observation')
        sums = sums >= 5
        return table.filter(table.ids(axis='observation')[sums], axis='observation',inplace=False).remove_empty()

    table_16s = filter_observations(table_16s)
    table_16s = filter_samples(table_16s, kwargs['s_depth_16s'])
    table_wgs = filter_observations(table_wgs)
    table_wgs = filter_samples(table_wgs, kwargs['s_depth_wgs'])

     # calculate cached dist
    print('get_cached_sent...')
    cached_dist, tip_info = get_cached_sent_dist(tree, kwargs['num_sent'])
    print('done cached')

    # do not need!!!
    table_ids = set(table_16s.ids())
    overlap = table_ids  & set(table_wgs.ids())
    table_16s = table_16s.filter(overlap, axis='sample', inplace=False).remove_empty()
    table_wgs = table_wgs.filter(overlap, axis='sample', inplace=False).remove_empty()
    ####

    # add token to end of wgs samples to avoid duplicates
    update_ids = {id:f'{id}-{kwargs["wgs_id"]}' for id in table_wgs.ids()}
    table_wgs.update_ids(update_ids)

    tables = []
    print('rarefying tables...')
    def postorder_sort(seq):
        alist = list(seq)
        alist.sort(key=lambda ob: tip_info[ob])
        return alist
    
    for _ in range(kwargs['num_rarefied']):
        rare_16s = table_16s.subsample(n=kwargs['s_depth_16s'])
        print('16s largest num obs:', max(rare_16s.pa(inplace=False).sum(axis='sample')))
        rare_wgs = table_wgs.subsample(n=kwargs['s_depth_wgs'])
        print('wgs largest num obs:', max(rare_wgs.pa(inplace=False).sum(axis='sample')))
        tables.append(rare_16s.merge(rare_wgs).sort(postorder_sort, axis='observation'))

    print('done rarefying')

    print('save tables')
    model_path = kwargs['model_path']
    data_path = kwargs['data_path']
    parent_path = f'{model_path}/{data_path}'
    if not os.path.exists(parent_path):
                os.makedirs(parent_path)
    for i, table in enumerate(tables):
        with biom_open(f"{parent_path}/table-{i}.biom", 'w') as f:
            table.to_hdf5(f, 'scale-16s-training')

   
    print('save cached_dist')
    np.save(
        f'{parent_path}/cached_dist.npy',
        cached_dist
    )
    print('done')
    return tables, cached_dist, tip_info


class DataLoader:
    def __init__(self, **kwargs):
        self.train_on_extras = False
        self.tables = kwargs['tables']
        self.extra_tables = kwargs['extra_tables']
        self.extra_per_epoch = 5
        self.tree = kwargs['tree']
        self.cached_dist = np.array(kwargs['cached_dist'], dtype=np.float32)
        self.tip_info = kwargs['tip_info']
        self.batch_size = kwargs['batch_size']
        self.num_batches = int((self.tables[0].shape[1] + (self.batch_size * self.extra_per_epoch))  / self.batch_size)
        self.max_obs = kwargs['max_obs']
        self.num_sent = kwargs['num_sent']
        self.mask_value = kwargs['mask_value'] if 'mask_value' in kwargs else -1
        self.num_epochs = kwargs['num_epochs']
        self.metadata = kwargs['metadata']
        self.wgs_id = kwargs['wgs_id']
        self.log_table = self.tables[0].copy()
        self.log_table_cached_dist_indx = np.apply_along_axis(
            lambda ob: self.tip_info[ob[0]],
            1,
            np.expand_dims(self.log_table.ids(axis='observation'), axis=1)
        )

        if self.train_on_extras:
            self.num_batches = 15
            self.extra_per_epoch = 15

        self.on_epoch_end()

    def __len__(self):
        return self.num_batches

    def _get_batch_data(self, batch_samples, table, cached_dist_indx ):
        def _sample_data_helper(sample, table_data, cached_dist, cached_dist_indx, output_shape):
            sample = sample[0]
            non_zero_mask = table_data(sample, axis='sample') > 0.5
            padded_data = np.zeros(output_shape)
            sample_data = cached_dist[np.compress(non_zero_mask, cached_dist_indx)]
            padded_data[:sample_data.shape[0], :] = sample_data
            return padded_data

        table_ids = table.ids()
        samples = np.empty_like(batch_samples)
        np.copyto(samples, table_ids[np.isin(table_ids, batch_samples)])
        samples = np.expand_dims(samples, axis=1)
        batch_data = np.apply_along_axis(_sample_data_helper, 1, samples, 
                                         table.data, self.cached_dist, cached_dist_indx,
                                         (self.max_obs, self.num_sent))
        return batch_data, samples.flatten()

    def _get_log_item(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_samples = self.log_table.ids()[start:end]
        data, s_ids = self._get_batch_data(batch_samples, self.log_table, self.log_table_cached_dist_indx)
        is_wgs = [self.wgs_id in id for id in s_ids]
        s_ids = [s_id.replace(f'-{self.wgs_id}', '') for s_id in s_ids]
        return data, s_ids, self._get_metadata(s_ids), is_wgs

    def __getitem__(self, index):
        batch_samples = self._get_batch_sample_names(index)
        data, samples = self._get_batch_data(batch_samples, self.epoch_table, self.cached_dist_indx)        
        return data, self.unifrac_matrix.filter(samples).data

    def _get_batch_sample_names(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        return self.sample_ids[start:end]
    
    def __call__(self):
        for _ in range(self.num_epochs):
            for i in range(self.__len__()):
                yield self.__getitem__(i)
            self.on_epoch_end()
        
    def _get_metadata(self, samples):
        s_data = self.metadata.loc[self.metadata.index.isin(samples), 'sample_type'].to_dict()
        return [s_data[s_id] for s_id in samples]

    def on_epoch_end(self):
        shuffle_indx = np.arange(len(self.extra_tables))
        np.random.shuffle(shuffle_indx)
        extra_tables = [self.extra_tables[i]
                            for i in shuffle_indx[:self.extra_per_epoch]]
        if not self.train_on_extras:
            # epoch_table = self.tables[random.randrange(len(self.tables))]
            epoch_table = self.tables[0]
            self.epoch_table = epoch_table.merge(extra_tables)
        else:
            epoch_table = extra_tables[0]
            self.epoch_table = epoch_table.merge(extra_tables[1:])

        self.sample_ids = self.epoch_table.ids().copy()
        np.random.shuffle(self.sample_ids)
        self.cached_dist_indx = np.apply_along_axis(
            lambda ob: self.tip_info[ob[0]],
            1,
            np.expand_dims(self.epoch_table.ids(axis='observation'), axis=1)
        )

        self.unifrac_matrix = unweighted(self.epoch_table, self.tree)

class DataLoaderToken:
    def __init__(self, **kwargs):
        self.tables = kwargs['tables']
        self.tree = kwargs['tree']
        self.feature_ids = self.tables[0].ids(axis='observation')#kwargs['feature_ids']
        self.batch_size = kwargs['batch_size']
        self.num_batches = kwargs['num_batches']
        self.samples_per_epoch = self.batch_size*self.num_batches
        self.max_obs = kwargs['max_obs']
        self.min_features = 1
        self.mask_value = kwargs['mask_value'] if 'mask_value' in kwargs else -1
        self.num_epochs = kwargs['num_epochs']
        self.metadata = kwargs['metadata']
        self.wgs_id = kwargs['wgs_id']

        # log data
        samples = self.tables[0].ids()
        data = np.zeros((self.feature_ids.shape[0], samples.shape[0]), dtype=np.int32)
        log_table = Table(data, self.feature_ids.copy(), samples)
        self.log_table = log_table.merge(self.tables[0])
        self.log_table = self.log_table.align_to(log_table, axis='observation')
        self.on_epoch_end()

    def __len__(self):
        return self.num_batches

    def _get_batch_data(self, batch_samples, table):
        def _sample_data_helper(sample, table_data, output_size):
            sample = sample[0]
            feature_tokens = np.argwhere(table_data(sample, axis='sample') > 0.5).flatten() + 1 # add one to account for mask token
            padded_data = np.zeros(output_size, dtype=np.int32)
            padded_data[:feature_tokens.shape[0]] = feature_tokens
            return padded_data
            
        samples = batch_samples.copy()
        samples = np.expand_dims(samples, axis=1)
        batch_data = np.apply_along_axis(_sample_data_helper, 1, samples, 
                                         table.data, self.max_obs)
        return batch_data

    def _get_log_item(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_samples = self.log_table.ids()[start:end]
        data = self._get_batch_data(batch_samples, self.log_table)
        is_wgs = [self.wgs_id in id for id in batch_samples]
        s_ids = [s_id.replace(f'-{self.wgs_id}', '') for s_id in batch_samples]
        return data, s_ids, self._get_metadata(s_ids), is_wgs

    def __getitem__(self, index):
        batch_samples = self._get_batch_sample_names(index)
        data = self._get_batch_data(batch_samples, self.epoch_table)        
        return data, self.unifrac_matrix.filter(batch_samples).data

    def _get_batch_sample_names(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        return self.epoch_samples[start:end]
    
    def __call__(self):
        for _ in range(self.num_epochs):
            for i in range(self.__len__()):
                yield self.__getitem__(i)
            self.on_epoch_end()
        
    def _get_metadata(self, samples):
        s_data = self.metadata.loc[self.metadata.index.isin(samples), 'sample_type'].to_dict()
        return [s_data[s_id] for s_id in samples]

    def on_epoch_end(self):
        self.epoch_table, self.epoch_samples = generate_random_table(self.feature_ids.copy(), self.samples_per_epoch,
                                                                     0, min_features=self.min_features, max_features=self.max_obs)
        self.unifrac_matrix = unweighted(self.epoch_table, self.tree)
