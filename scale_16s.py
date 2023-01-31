import tensorflow as tf
from transformer.transformer import Transformer, CustomSchedule, masked_loss, masked_accuracy


# load datasets
wgs_data = tf.data.Dataset.load('transformer/datasets/wgs')
data_16s = tf.data.Dataset.load('transformer/datasets/16s')

# load tokenizers
wgs_tokenizer = tf.keras.models.load_model('transformer/tokenizers/wgs_tokenizer')
tokenizer_16s = tf.keras.models.load_model('transformer/tokenizers/16s_tokenizer')

def prepare_batch(batch_16s, wgs_batch):
    tokens_16s = tokenizer_16s(batch_16s)
    wgs_tokens = wgs_tokenizer(wgs_batch)

    wgs_inputs = wgs_tokens[:, :-1]
    wgs_labels = wgs_tokens[:, 1:]

    return (tokens_16s, wgs_inputs), wgs_labels

BUFFER_SIZE = 20000
BATCH_SIZE = 2

def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))

# create training and validation examples
examples = tf.data.Dataset.zip((data_16s, wgs_data))
num_examples = examples.cardinality().numpy()
train_examples = examples.take(int(0.75*num_examples))
val_examples = examples.skip(int(0.75*num_examples))

# clean up memeory
del examples
del wgs_data
del data_16s

# Create training and validation set batches.
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)
# for (pt, en), l in val_batches.take(1):
#     print(pt)
#     print(en)
#     print(l)
#     break

# hyperparams
num_layers = 1
d_model = 64
dff = 128
num_heads = 1
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=len(tokenizer_16s.layers[0].get_vocabulary()),
    target_vocab_size=len(wgs_tokenizer.layers[0].get_vocabulary()),
    dropout_rate=dropout_rate)

# output = transformer((pt, en))
# print(output)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

transformer.fit(train_batches,
                epochs=20,
                validation_data=val_batches)