import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import faiss
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import featurize_prottrans, embed_tm_vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns 
from biom import load_table


#Load the ProtTrans model and ProtTrans tokenizer
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
gc.collect()

device = torch.device('gpu')
model = model.to(device)
model = model.eval()

tm_vec_model_cpnt = "tm_vec_swiss_model.ckpt"
tm_vec_model_config = "tm_vec_swiss_model_params.json"

#Load the TM-Vec model
tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
model_deep = model_deep.to(device)
model_deep = model_deep.eval()

REL_PATH='samples'
WGS_BASE_TABLE='wgs_table'
BASE_TABLE_16S='table_16s'
SAMPLE_TYPE='env_package'
wgs_table = load_table('%s/16s.biom' % REL_PATH)
wgs_table = wgs_table.head()
with open('16s-table.tsv', 'r') as f:
    lines = f.readlines()
    lines = lines[2:]

sequences = [l.split('\t')[0] for l in lines]



#Loop through the sequences and embed them
i = 0
embed_all_sequences = []

while i < len(sequences): 
    protrans_sequence = featurize_prottrans(sequences[i:i+1], model, tokenizer, device)
    embedded_sequence = embed_tm_vec(protrans_sequence, model_deep, device)
    embed_all_sequences.append(embedded_sequence)
    i = i + 1
print(embed_all_sequences)