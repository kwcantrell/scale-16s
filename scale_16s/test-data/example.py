from skbio import TreeNode
from biom import load_table

# load/iterate TreeNode
tree = TreeNode.read('test-tree.nwk')
for node in tree.postorder():
    print(node.name, node.length)

# load/extraction ids from biom object
table = load_table('test-table.biom')
sample_ids = table.ids(axis='sample')
observation_ids = table.ids(axis='observation')
