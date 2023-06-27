import os
import math
import json
import gzip
import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
from datetime import datetime
# from dataset_utils import get_data, get_data_params

def _append_cur_time(path):
    cur_time = datetime.now()
    return f"{path}-{cur_time}"

def project_embbeddings(embeddings, labels, log_dir='logs/imdb-example/',
                        metadata_path='metadata.tsv', checkpoint_path="embedding.ckpt",
                        embedding_tensor_name="embedding/.ATTRIBUTES/VARIABLE_VALUE"):

    # Set up a logs directory, so Tensorboard knows where to look for files.
    # log_dir='logs/imdb-example/'
    log_dir = _append_cur_time(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, metadata_path), "w") as f:
        f.write('class\tsequence_type\n')
        for c, s in labels:
            f.write(f"{c}\t{s}\n")

    # Save the weights we want to analyze as a variable.
    weights = tf.Variable(np.array(embeddings))

    # Create a checkpoint from embedding, the filename and key are the
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, checkpoint_path))

    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = embedding_tensor_name
    embedding.metadata_path = metadata_path
    projector.visualize_embeddings(log_dir, config)

def project_data_on_trained_model(model_prefix):
    model_dir = f'models/{model_prefix}'
    model = tf.saved_model.load(os.path.join(model_dir, 'encoder'))
    with gzip.open(os.path.join(model_dir, 'class-data.gz'), 'rb') as f:
        data = json.loads(f.read().decode('utf-8'))['class_data']
    with open(os.path.join(model_dir, 'params.json')) as f:
        model_params = json.load(f)
    token_to_class = model_params['token_to_class']

    max_len = 2048
    bs = 64
    labels = []
    embeddings = []
    for clss, samples in enumerate(data):
        # print(len(samples))
        size = math.ceil(len(samples)/bs)
        class_label = model_params['token_to_class'][str(clss)]
        for b in range(size):       
            batch = samples[b*bs:(b+1)*bs]

            # create metadata
            metadata = []
            for sample in batch:
                if len(sample) < 1:
                    metadata.append((class_label, 'UNK'))
                elif sample[0] < model_params['wgs_start_token']:
                    metadata.append((class_label, '16s'))
                else:
                    metadata.append((class_label, 'wgs'))
            
            # metadata = [(class_label, '16s') if sample[0] < model_params['wgs_start_token'] else (class_label, 'wgs') 
            #                   for sample in batch]
            
            batch = tf.keras.utils.pad_sequences(batch, maxlen=max_len,
                                                padding='post', truncating='post',
                                                value=0).tolist()
            # for sample in batch:
            x = tf.convert_to_tensor(batch, dtype=tf.int32)
            pred = model(x, False, None)
            embeddings.append(pred)
            labels.append(metadata)

    labels = tf.concat(labels, axis=0)
    embeddings = tf.concat(embeddings, axis=0)
    project_embbeddings(embeddings, labels, os.path.join(model_dir, 'full-embbeddings/'))

if __name__ == '__main__':
    project_data_on_trained_model('emp500')