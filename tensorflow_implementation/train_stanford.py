from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


from stanford_product import StanfordModel, StanfordProductData
from pprint import pprint
import tensorflow.estimator as estimator
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
val_examples = 24080
train_examples = 95973
batch_size = 256
throttle_mins = 1
# model_dir = 'C:\\Users\\Ben\\PycharmProjects\\Fixup\\tensorflow_implementation\\models\\stanford'
model_dir = '/content/models/stanford'
# p = 'C:\\Users\\Ben\\PycharmProjects\\Fixup\\tensorflow_implementation\\models\\cifar10Fixup\\model.ckpt-78000'
p = '/content/models/cifar10Fixup/model.ckpt-78000'

params = dict()
params['identities'] = 6
params['examples'] = 3
params['size'] = 227
params['batch_size'] = params['identities']*params['examples']
params['steps_per_epoch'] = train_examples // params['batch_size']
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
    params['identities'],
    params['examples'],
    params['steps_per_epoch'],
    val_examples//params['batch_size'],
    params['size'])

run_config = estimator.RunConfig(
    save_checkpoints_steps=params['steps_per_epoch'],
    save_summary_steps=200,
    keep_checkpoint_max=10)


ws = estimator.WarmStartSettings(
    ckpt_to_initialize_from=p,
    vars_to_warm_start='^(?!.*(beta_margins|dense/kernel)).*$')
fixup_estimator = estimator.Estimator(
    model_dir=model_dir,
    model_fn=model.model_fn,
    params=params,
    config=run_config,
    warm_start_from=ws)

# training/evaluation specs for run
train_spec = estimator.TrainSpec(
    input_fn=stanford_data.build_training_data,
    max_steps=params['total_steps_train'],)

eval_spec = estimator.EvalSpec(
    input_fn=stanford_data.build_validation_data,
    steps=None,
    throttle_secs=params['throttle_eval'],
    start_delay_secs=0)

# run train and evaluate
estimator.train_and_evaluate(
    fixup_estimator,
    train_spec,
    eval_spec)
