import pandas as pd
import numpy as np
import pandas as pd

m = pd.read_csv('emp_qiime_mapping_release1.tsv', sep='\t', index_col=0)
print(m['sample_scientific_name'].value_counts())