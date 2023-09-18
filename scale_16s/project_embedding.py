import os
import math
import json
import gzip
import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
from datetime import datetime

class ProjectEncoder(tf.keras.callbacks.Callback):
    def __init__(self, kwargs, log_step=10,
                        metadata_path='metadata.tsv', checkpoint_path="embedding.ckpt",
                        tensor_name="embedding/.ATTRIBUTES/VARIABLE_VALUE"):
        super().__init__()
        self.data = kwargs['data']
        self.model_path = kwargs['model_path'] 
        self.logdir = kwargs['logdir']
        self.parent_dir = f'{self.model_path}/{self.logdir}'
        self.metadata_path = metadata_path
        self.checkpoint_path = checkpoint_path
        self.tensor_name = tensor_name
        self.log_step = log_step
        self.cur_step = 0
        self._set_up_logdir()

    def _set_up_logdir(self):
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)

    def _log_epoch_data(self):
        tf.print('loggin data...')
        self.model.save(os.path.join(self.model_path, 'encoder.keras'))
        epoch_data = []
        with open(os.path.join(self.parent_dir, self.metadata_path), 'w') as f:
            f.write('sample_id\tsample_type\tseq_type\n')
            for idx in range(int(self.data.log_table.ids().shape[0] / self.data.batch_size)): # number of 64 size batches in tables[0]
                data, batch_sample_ids, metadata, is_wgs = self.data._get_log_item(idx)
                embedding_vecs = self.model(tf.convert_to_tensor(data), False, None)
                epoch_data.append(embedding_vecs)
                for s_id, s_type, wgs in zip(batch_sample_ids, metadata, is_wgs):
                    seq = 'wgs' if wgs else '16s'
                    f.write(f'{s_id}\t{s_type}\t{seq}\n')
        
        embeddings = tf.concat(epoch_data, axis=0)       
        weights = tf.Variable(np.array(embeddings))
        # Create a checkpoint from embedding, the filename and key are the
    
        # name of the tensor.
        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(os.path.join(self.parent_dir, self.checkpoint_path))

        # Set up config.
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()

        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
        embedding.tensor_name = self.tensor_name
        embedding.metadata_path = self.metadata_path
        projector.visualize_embeddings(self.parent_dir, config)
        tf.print('done logging.')

    def on_epoch_end(self, epoch, logs=None):
        if self.cur_step % self.log_step == 0:
            self._log_epoch_data()
            self.cur_step = 0
        self.cur_step += 1
        return super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        self._log_epoch_data()
        return super().on_train_end(logs)

class ProjectFeatures(tf.keras.callbacks.Callback):
    def __init__(self, kwargs, log_step=25,
                        checkpoint_path="embedding.ckpt",
                        tensor_name="embedding/.ATTRIBUTES/VARIABLE_VALUE"):
        super().__init__()
        self.data = kwargs['data']
        self.model_path = kwargs['model_path'] 
        self.logdir = kwargs['logdir']
        self.parent_dir = f'{self.model_path}/{self.logdir}'
        self.checkpoint_path = checkpoint_path
        self.tensor_name = tensor_name
        self.log_step = log_step
        self.cur_step = 0

    def _log_epoch_data(self):
        
        embeddings = tf.concat(epoch_data, axis=0)       
        weights = tf.Variable(np.array(embeddings))
        # Create a checkpoint from embedding, the filename and key are the
    
        # name of the tensor.
        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(os.path.join(self.parent_dir, self.checkpoint_path))

        # Set up config.
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()

        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
        embedding.tensor_name = self.tensor_name
        embedding.metadata_path = self.metadata_path
        projector.visualize_embeddings(self.parent_dir, config)
        tf.print('done logging.')

    def on_epoch_end(self, epoch, logs=None):
        if self.cur_step % self.log_step == 0:
            self._log_epoch_data()
            self.cur_step = 0
        self.cur_step += 1
        return super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        self._log_epoch_data()
        return super().on_train_end(logs)
