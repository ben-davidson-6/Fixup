import tensorflow as tf
from tensorflow_implementation.resnet_fixup import FixUpResnet
from tensorflow_implementation.cifar10 import CIFAR10
import tensorflow.python.estimator.estimator_lib as estimator
from pprint import pprint


# get data loader
cifar_data = CIFAR10()

# define the network
model = FixUpResnet(classes=10)
model.add_block(3, 64)
model.add_block(4, 128)
model.add_block(6, 256)
model.add_block(3, 512)

# define parameters used
epochs_to_train = 200
val_examples = 10000
train_examples = 50000
batch_size = 256
throttle_mins = 10

params = dict()
params['steps_per_epoch'] = train_examples // batch_size
params['total_steps_train'] = params['steps_per_epoch'] * epochs_to_train
params['throttle_eval'] = throttle_mins * 60
params['momentum'] = 0.9
params['bias_reduction'] = 0.1
params['epochs_to_reduce_at'] = [80, 140]
params['initial_learning_rate'] = 0.1
params['epoch_reduction_factor'] = 0.1
pprint(params)

run_config = estimator.RunConfig(
    save_checkpoints_steps=params['steps_per_epoch'],
    save_summary_steps=100,
    keep_checkpoint_max=5
)

fixup_estimator = estimator.Estimator(
    model_dir='C:\\Users\\Ben\\PycharmProjects\\Fixup\\tensorflow_implementation\\models\\test',
    model_fn=model.model_fn,
    params=params,
    config=run_config)

# training/evaluation specs for run
train_spec = estimator.TrainSpec(
    input_fn=cifar_data.build_training_data,
    max_steps=params['total_steps_train']
)
eval_spec = estimator.EvalSpec(
    input_fn=cifar_data.build_validation_data,
    steps=None,
    throttle_secs=params['throttle_eval'])

# run train and evaluate
estimator.train_and_evaluate(
    fixup_estimator,
    train_spec,
    eval_spec
   )


