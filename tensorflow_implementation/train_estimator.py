import tensorflow as tf
from tensorflow_implementation.resnet_fixup import FixUpResnet
from tensorflow_implementation.cifar10 import CIFAR10
import tensorflow.python.estimator.estimator_lib as estimator
from pprint import pprint




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
model_dir = 'C:\\Users\\Ben\\PycharmProjects\\Fixup\\tensorflow_implementation\\models\\cifar10Mixup'

params = dict()
params['batch_size'] = batch_size
params['steps_per_epoch'] = train_examples // params['batch_size']
params['total_steps_train'] = params['steps_per_epoch'] * epochs_to_train
params['throttle_eval'] = throttle_mins * 60
params['momentum'] = 0.9
params['bias_reduction'] = 0.1
params['epochs_to_reduce_at'] = [40, 120]
params['initial_learning_rate'] = 0.1
params['epoch_reduction_factor'] = 0.1
params['mixup_val'] = 0.7
pprint(params)


# get data loader
cifar_data = CIFAR10(batch_size=params['batch_size'], mixup_val=params['mixup_val'])


run_config = estimator.RunConfig(
    save_checkpoints_steps=params['steps_per_epoch'],
    save_summary_steps=500,
    keep_checkpoint_max=5
)

fixup_estimator = estimator.Estimator(
    model_dir=model_dir,
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
    throttle_secs=params['throttle_eval'],
    start_delay_secs=0)

# run train and evaluate
estimator.train_and_evaluate(
    fixup_estimator,
    train_spec,
    eval_spec
   )


