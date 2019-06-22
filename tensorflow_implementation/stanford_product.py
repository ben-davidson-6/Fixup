import tensorflow as tf
import tensorflow.estimator as estimator
from resnet_fixup import FixUpResnet
from margin_loss import margin_loss, PAIRWISE_DISTANCES

from random import shuffle
import pandas as pd
import os
import numpy as np


data_folder = 'C:\\Users\\Ben\\PycharmProjects\\Fixup\\Stanford_Online_Products'
pickle_path = os.path.join(data_folder, 'df.pickle')


def preprocess():
    # read data into dataframe with path and class name
    class_text_fnames = [x for x in os.listdir(data_folder) if x[-4:] == '.txt']
    # text_files = [os.path.join(data_folder, x) for x in class_text_fnames]
    class_names = [x[:-4] for x in class_text_fnames]
    class_folders = [os.path.join(data_folder, x) for x in class_names]
    data = [(os.path.join(class_folder, im_name), os.path.basename(class_folder))
     for class_folder in class_folders for im_name in os.listdir(class_folder)]
    df = pd.DataFrame(data, columns=['path', 'class_name'])

    # make integer label
    df['labels'] = pd.Categorical(df['class_name']).codes.astype(np.int32)

    train = np.random.random([df.shape[0]]) < 0.8
    df['train'] = train

    df = df.sample(frac=1)
    df.to_pickle(pickle_path)


class StanfordProductData():

    def __init__(self, batch_size,):
        self.batch_size = batch_size
        df = pd.read_pickle(pickle_path)
        self.train_df = df[df['train']]
        self.val_df = df[~df['train']]

    def neccessary_processing(self, path, label):
        raw_f = tf.read_file(path)
        image = tf.image.decode_jpeg(raw_f, channels=3)
        image = image/255
        image -= tf.constant([0.4914, 0.4822, 0.4465])[None, None]
        image /= tf.constant([0.2023, 0.1994, 0.2010])[None, None]
        image = tf.image.resize_bilinear(image[None], tf.constant((64, 64)))[0]

        return image, label

    def augmentations(self, image, label):
        image = tf.image.random_flip_left_right(image)
        return image, label

    def build_training_data(self):
        ds = tf.data.Dataset.from_tensor_slices(
            (self.train_df['path'],
            self.train_df['labels']))
        ds = ds.shuffle(5000).repeat()
        ds = ds.map(lambda im, l: self.augmentations(*self.neccessary_processing(im, l)), num_parallel_calls=4)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds

    def build_validation_data(self):
        ds = tf.data.Dataset.from_tensor_slices(
            (self.val_df['path'],
             self.val_df['labels']))
        ds = ds.shuffle(5000)
        ds = ds.map(self.neccessary_processing, num_parallel_calls=4)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds


class CIFAR10Dataset():

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def neccessary_processing(self, image, label):
        image = image/255
        image -= tf.constant([0.4914, 0.4822, 0.4465])[None, None]
        image /= tf.constant([0.2023, 0.1994, 0.2010])[None, None]
        return image, label

    def augmentations(self, image, label):
        # image augmentations
        image = tf.image.random_flip_left_right(image)
        padding = tf.constant([
            [6, 6],
            [6, 6],
            [0, 0],
        ])
        image = tf.pad(image, padding)
        image = tf.random_crop(image, tf.constant([32, 32, 3]))
        return image, label

    def train_generator(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        def gen():
            for image, label in zip(x_train, y_train):
                yield image, label[0]
        return gen

    def build_training_data(self):
        gen = self.train_generator()
        ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((32, 32, 3), ()))
        ds = ds.shuffle(5000).repeat()
        ds = ds.map(lambda im, l: self.augmentations(*self.neccessary_processing(im, l)), num_parallel_calls=4)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds

    def validation_generator(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        def gen():
            for image, label in zip(x_test, y_test):
                yield image, label[0]
        return gen

    def build_validation_data(self):
        gen = self.validation_generator()
        ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((32, 32, 3), ()))
        ds = ds.shuffle(1000)
        ds = ds.map(self.neccessary_processing, num_parallel_calls=4)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds


class StanfordModel(FixUpResnet):

    def embedding_net(self, features, params):
        net = tf.layers.dense(features, params['units'], use_bias=False, activation=None)
        net = tf.nn.l2_normalize(net, axis=1)
        return net

    def model_fn(self, features, labels, mode, params):
        endpoints = self.build_network(features)
        net = endpoints['feature'][:, 0, 0, :]
        embedding = self.embedding_net(net, params)
        betas = tf.get_variable('beta_margins', initializer=params['betas']*tf.ones([12]))
        if mode == estimator.ModeKeys.PREDICT:
            return self.predict_spec(embedding)
        loss = StanfordModel.loss(embedding, labels, betas, params)
        if mode == estimator.ModeKeys.TRAIN:
            return self.train_spec(loss, params)
        if mode == estimator.ModeKeys.EVAL:
            return self.eval_spec(loss, labels)

    #####################################################################
    # Defining train/eval/predict for estimator
    #####################################################################

    def predict_spec(self, features):
        named_predictions = {
            'embedding': features}
        return estimator.EstimatorSpec(
            estimator.ModeKeys.PREDICT,
            predictions=named_predictions)

    def train_spec(self, loss, params):
        train_op = self._training_op(loss, params)
        return estimator.EstimatorSpec(
            estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    def eval_spec(self, loss, labels):
        g = tf.get_default_graph()
        D = g.get_tensor_by_name(PAIRWISE_DISTANCES + ':0')
        D *= -1
        _, top_1 = tf.nn.top_k(D, 2)
        top_1 = top_1[:, 1]
        estimated = tf.gather_nd(labels, top_1[:, None])

        # Define the metrics:
        metrics_dict = {
            'Map@1': tf.metrics.accuracy(labels, estimated)}

        # return eval spec
        return estimator.EstimatorSpec(
            estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=metrics_dict)

    #####################################################################
    # Traingin op
    #####################################################################

    def _split_vars_for_optimisers(self):
        all_vars = tf.trainable_variables()
        normal_vars = [x for x in all_vars
                       if ('fixup_bias' not in x.op.name
                           and 'fixup_scale' not in x.op.name
                           and 'beta' not in x.op.name)]
        scale_and_bias_vars = [x for x in all_vars
                               if 'fixup_bias' in x.op.name
                               or 'fixup_scale' in x.op.name]
        beta = [x for x in all_vars if 'beta_margins' in x.op.name]
        return normal_vars, scale_and_bias_vars, beta

    def _training_op(self, loss, params):
        normal_vars, scale_and_bias_vars, beta = self._split_vars_for_optimisers()
        learning_rate = self._make_learning_rate(params)
        tf.summary.scalar('learning_rate', learning_rate)

        optimiser_normal = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=params['momentum'])
        optimiser_scale_and_bias = tf.train.MomentumOptimizer(
            learning_rate * params['bias_reduction'],
            momentum=params['momentum'])
        optimiser_beta = tf.train.MomentumOptimizer(
            learning_rate * params['beta_increase'],
            momentum=params['momentum'])

        grads = tf.gradients(loss, normal_vars + scale_and_bias_vars + beta)
        # tf.summary.merge([tf.summary.histogram("%s-grad" % g.name, g) for g in grads])
        grads1 = grads[:len(normal_vars)]
        grads2 = grads[len(normal_vars):len(normal_vars) + len(scale_and_bias_vars)]
        grads3 = grads[len(normal_vars) + len(scale_and_bias_vars):]

        tran_op_normal = optimiser_normal.apply_gradients(
            zip(grads1, normal_vars))
        train_op_scale_bias = optimiser_scale_and_bias.apply_gradients(
            zip(grads2, scale_and_bias_vars))
        train_op_betas = optimiser_beta.apply_gradients(
            zip(grads3, beta),
            global_step=tf.train.get_or_create_global_step())
        train_op = tf.group(tran_op_normal, train_op_scale_bias, train_op_betas)
        return train_op

    def _epoch_reduced_lr(self, learn, reduction_steps, global_step, reduce_factor):
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
                lambda: self._epoch_reduced_lr(
                    learn * reduce_factor,
                    reduction_steps[1:],
                    global_step,
                    reduce_factor)
            )

    def _make_learning_rate(self, params):
        reduce_learning_rate_at = params['epochs_to_reduce_at']
        init_learn = params['initial_learning_rate']
        steps_per_epoch = params['steps_per_epoch']
        reduction_factor = params['epoch_reduction_factor']
        global_step = tf.train.get_or_create_global_step()

        reduce_learning_rate_at = [x * steps_per_epoch for x in reduce_learning_rate_at]
        learn = tf.constant(init_learn)
        return self._epoch_reduced_lr(learn, reduce_learning_rate_at, global_step, reduction_factor)

    @staticmethod
    def loss(features, labels, betas, params):
        return margin_loss(labels, features, betas, params)


if __name__ == '__main__':
    s = StanfordProductData(1)
    print(s.train_df.shape)
    print(s.val_df.shape)