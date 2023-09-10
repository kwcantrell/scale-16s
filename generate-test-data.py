from biom.table import Table
from biom.util import biom_open
import numpy as np

sample_ids = ['s%d' % i for i in range(5)]
observation_ids = ['t%d' % i for i in range(10)]
data = (np.random.rand(10, 5) > 0.5).astype(np.int32)

test_table = Table(data, observation_ids, sample_ids)
with biom_open('test-table.biom', 'w') as f:
    test_table.to_hdf5(f, 'tes data')
