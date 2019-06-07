import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from resnet_fixup import FixUpResnet
from cifar10 import CIFAR10


from tensorflow_tools.training_model import TrainingModel
from tensorflow_tools.train import Trainer
from tensorflow_tools.constants import LOSS_TENSOR_0, TRAIN_STEP

import tensorflow as tf


def make_learning_rate_tensor(learn, reduction_steps, global_step, reduce_factor):
    if len(reduction_steps) == 1:
        return tf.cond(
            global_step < reduction_steps[0],
            lambda: learn,
            lambda: learn * reduce_factor
        )
    else:
        return tf.cond(
            global_step < reduction_steps[0],
            lambda: learn,
            lambda: make_learning_rate_tensor(
                learn*reduce_factor,
                reduction_steps[1:],
                global_step,
                reduce_factor)
        )


def lr_schedule(init_learn, global_step, steps_per_epoch=225):
    reduce_learning_rate_at = [120, 180]
    reduce_learning_rate_at = [x*steps_per_epoch for x in reduce_learning_rate_at]
    learn = tf.constant(init_learn)
    learning_rate = make_learning_rate_tensor(learn, reduce_learning_rate_at, global_step, 0.1)
    return learning_rate


def training_step(learning_rate):
    g = tf.get_default_graph()
    loss = g.get_tensor_by_name(LOSS_TENSOR_0)
    all_vars = tf.trainable_variables()
    normal_vars = [x for x in all_vars if ('fixup_bias' not in x.op.name and 'fixup_scale' not in x.op.name)]
    scale_and_bias_vars = [x for x in all_vars if 'fixup_bias' in x.op.name or 'fixup_scale' in x.op.name]
    optimiser_normal = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    optimiser_scale_and_bias = tf.train.MomentumOptimizer(learning_rate*0.1, momentum=0.9)
    grads = tf.gradients(loss, normal_vars + scale_and_bias_vars)
    grads1 = grads[:len(normal_vars)]
    grads2 = grads[len(normal_vars):]
    tran_op_normal = optimiser_normal.apply_gradients(zip(grads1, normal_vars))
    train_op_scale_bias = optimiser_scale_and_bias.apply_gradients(zip(grads2, scale_and_bias_vars))

    global_step = tf.train.get_global_step()
    global_step_update = tf.assign(global_step, global_step + 1, name='update_global_step')
    train_op = tf.group(tran_op_normal, train_op_scale_bias, global_step_update, name=TRAIN_STEP)
    return train_op


def accuracy(name):
    g = tf.get_default_graph()
    logits = g.get_tensor_by_name('logits:0')
    labels = g.get_tensor_by_name('labels:0')
    probs = tf.nn.softmax(logits)
    estimated_label = tf.argmax(probs, axis=1)
    return tf.metrics.accuracy(labels, estimated_label, name=name)


def train_model():
    data = CIFAR10()
    net = FixUpResnet(classes=10)
    net.add_block(3, 64)
    net.add_block(4, 128)
    net.add_block(6, 256)
    net.add_block(3, 512)

    model = TrainingModel(net, data)
    model.build_network()
    model.add_loss(FixUpResnet.loss)

    model.add_training_op(training_step, lr_schedule, init_learn=0.1)

    model.add_standard_loss_summaries()
    model.add_metric(accuracy, 'accuracyBatch')

    train = Trainer('cifar10', model, ['validation', 'training'])
    # train.add_early_stopping('validation', max_stall=15)
    train.train(200)


if __name__ == '__main__':
    train_model()