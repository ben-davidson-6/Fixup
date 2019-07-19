from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import tensorflow as tf



import os
from pprint import pprint

from stanford_product import StanfordModel, StanfordProductData
import logging
logging.getLogger().setLevel(logging.INFO)


# define the network
model = StanfordModel()
model.add_block(3, 64)
model.add_block(4, 128)
model.add_block(6, 256)
model.add_block(3, 512)

# define parameters used
epochs_to_train = 400
val_images = 59795
train_images = 60258
batch_size = 256
throttle_mins = 0
train_examples = 4
train_identities = 7
val_identities = 40
val_examples = 4
model_name = 'stanford_product'
model_dir = '/home/ben/tensorflow_models/'
model_dir = os.path.join(model_dir, model_name)

params = dict()
params['train_identities'] = train_identities
params['train_examples'] = train_examples
params['size'] = 227
params['train_batch_size'] = params['train_identities']*params['train_examples']
params['val_identities'] = val_identities
params['val_examples'] = val_examples
params['val_batch_size'] = params['val_identities']*params['val_examples']
params['steps_per_epoch'] = train_images // params['train_batch_size']
params['val_steps_per_epoch'] = val_images // params['val_batch_size']
params['total_steps_train'] = params['steps_per_epoch'] * epochs_to_train
params['throttle_eval'] = throttle_mins * 60
params['momentum'] = 0.9
params['bias_reduction'] = 0.1
params['epochs_to_reduce_at'] = [150, 300]
params['initial_learning_rate'] = 0.1
params['epoch_reduction_factor'] = 0.1
params['alpha'] = 0.2
params['nu'] = 0.
params['cutoff'] = 0.5
params['add_summary'] = True
params['units'] = 128
params['beta_0'] = 1.2
params['beta_increase'] = 0.1

pprint(params)

# get data loader
stanford_data = StanfordProductData(
    params['train_identities'],
    params['train_examples'],
    params['val_identities'],
    params['val_examples'],
    params['steps_per_epoch'],
    params['val_steps_per_epoch'],
    params['size'])

run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=params['steps_per_epoch'],
    save_summary_steps=200,
    keep_checkpoint_max=10)

p = '/home/ben/tensorflow_models/cifar_10/model.ckpt-78000'
ws = tf.estimator.WarmStartSettings(
    ckpt_to_initialize_from=p,
    vars_to_warm_start='^(?!.*(beta_margins|dense/kernel)).*$')

fixup_estimator = tf.estimator.Estimator(
    model_dir=model_dir,
    model_fn=model.model_fn,
    params=params,
    config=run_config,
    warm_start_from=ws)


# training/evaluation specs for run
train_spec = tf.estimator.TrainSpec(
    input_fn=stanford_data.build_training_data,
    max_steps=params['total_steps_train'],)


def map_compare(best_eval_result, current_eval_result):
    key = StanfordModel.map_name
    return best_eval_result[key] < current_eval_result[key]


serve_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
    {'images': tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])}
)

exporter = tf.estimator.BestExporter(
    compare_fn=map_compare,
    exports_to_keep=5,
    serving_input_receiver_fn=serve_fn
)

eval_spec = tf.estimator.EvalSpec(
    input_fn=stanford_data.build_validation_data,
    steps=None,
    throttle_secs=params['throttle_eval'],
    start_delay_secs=0,
    exporters=exporter)

# run train and evaluate
tf.estimator.train_and_evaluate(
    fixup_estimator,
    train_spec,
    eval_spec)
