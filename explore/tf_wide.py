
import os; os.environ['OMP_NUM_THREADS'] = '1'
import tensorflow as tf

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there

# A, paths
data_path = "../input/"
data_path = "/home/darragh/tdata/data/"
train = data_path+'train.csv'               # path to training file
test = data_path+'test.csv'                 # path to testing file
submission = 'sub_proba.csv'  # path of to be outputted submission file


_CSV_COLUMN_DEFAULTS = [ [''] for _ in range(8)]
_CSV_COLUMNS = ['ip','app','device','os','channel','click_time','attributed_time','is_attributed']

def build_model_columns():
    app     = tf.feature_column.categorical_column_with_vocabulary_list('app', [ str(i) for i in range(650)])
    device  = tf.feature_column.categorical_column_with_vocabulary_list('device', [ str(i) for i in range(3500)])
    os      = tf.feature_column.categorical_column_with_vocabulary_list('os', [ str(i) for i in range(720)])
    channel = tf.feature_column.categorical_column_with_vocabulary_list('channel', [ str(i) for i in range(500)])
    ip      = tf.feature_column.categorical_column_with_hash_bucket('ip', hash_bucket_size=200000)

    base_columns = [app, device, os, channel, ip]
    
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['app', 'channel'], hash_bucket_size=50000),
        tf.feature_column.crossed_column(
            ['os', 'channel'], hash_bucket_size=50000),
    ]
    
    wide_columns = base_columns + crossed_columns
    
    return wide_columns

def input_fn(data_file, num_epochs, shuffle, batch_size):

    def parse_csv(value):    
        print('Parsing', train)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('is_attributed')
        features.pop('attributed_time')
        features.pop('click_time')
        return features, tf.equal(labels, '1')
    
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(train)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    
    return dataset


def build_estimator(model_dir):
    """Build an estimator appropriate for the given model type."""
    wide_columns = build_model_columns()
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(
        intra_op_parallelism_threads=4, use_per_session_threads=1, inter_op_parallelism_threads=1))
    return tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns, config=run_config)


model = build_estimator('/home/darragh/tdata/tf')
for n in range(18000):
    model.train(input_fn=lambda: input_fn(train, 1, True, 10000))