import numpy as np
import random
from biom.util import biom_open, load_table
import os
from unifrac import unweighted
from utils import get_cached_sent_dist
from biom import Table
from bp import parse_newick
from utils import get_tip_ids

table_16s = load_table('data/emp500-16s.biom')
table_wgs = load_table('data/emp500-wgs.biom')
with open('data/small-test/phylo-emp500-tree.nwk') as f:
    tree = parse_newick(f.readline())


table_16s = table_16s.subsample(n=1000)
table_wgs = table_wgs.subsample(n=500000)
update_ids = {id:'wgs'}