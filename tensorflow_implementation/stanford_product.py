import tensorflow as tf
import tensorflow.estimator as estimator
from resnet_fixup import FixUpResnet
from margin_loss import margin_loss, PAIRWISE_DISTANCES

import pandas as pd
import os
import numpy as np
from random import sample

data_folder = '/home/ben/datasets/Stanford_Online_Products'
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
    df['identity_txt'] = df['path'].map(lambda x: os.path.basename(x).split('_')[0])
    df['identity'] = pd.Categorical(df['identity_txt']).codes.astype(np.int32)
    unique_codes = df['identity'].unique()
    train_identities = pd.Series(unique_codes).sample(frac=0.5)
    df['train'] = df['identity'].isin(train_identities)

    df = df.sample(frac=1)
    df.to_pickle(pickle_path)


class StanfordProductData():

    df = pd.read_pickle(pickle_path)
    train_df = df[df['train']]
    val_df = df[~df['train']]
    train_identities = pd.Series(train_df['identity'].unique())
    val_identities = pd.Series(val_df['identity'].unique())
    identities = list(range(22634))
    print(train_df.shape)
    print(val_df.shape)

    def __init__(
            self,
            train_identities,
            train_examples,
            val_identities,
            val_examples,
            steps_per_train_epoch,
            steps_per_val_epoch,
            size):
        self.train_batch_size = train_identities*train_examples
        self.train_identities = train_identities
        self.train_examples = train_examples
        self.val_batch_size = val_identities * val_examples
        self.val_identities = val_identities
        self.val_examples = val_examples

        self.steps_per_train_epoch = steps_per_train_epoch
        self.steps_per_val_epoch = steps_per_val_epoch
        self.size = size

    def data_generator(self, train=False):
        if train:
            df = self.train_df
            steps = self.steps_per_train_epoch
            n_examples = self.train_examples
            n_ids = self.train_identities
            identities = StanfordProductData.train_identities
        else:
            df = self.val_df
            steps = self.steps_per_val_epoch
            n_examples = self.val_examples
            n_ids = self.val_identities
            identities = StanfordProductData.val_identities

        df = df.sample(frac=1)
        df = df.groupby('identity')

        paths, labels = [], []
        for _ in range(steps):
            sampled_identities = identities.sample(n=n_ids)
            for ident in sampled_identities:
                ident_df = df.get_group(ident)
                to_sample = min(n_examples, ident_df.shape[0])
                ident_df = ident_df.sample(n=to_sample)

                paths += ident_df['path'].tolist()
                labels += ident_df['identity'].tolist()

        for p, l in zip(paths, labels):
            yield p, l

    def neccessary_processing(self, path, label):
        raw_f = tf.read_file(path)
        image = tf.image.decode_jpeg(raw_f, channels=3)
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize_bilinear(image[None], tf.constant((256, 256)))[0]
        image = tf.image.random_crop(image, tf.constant((self.size, self.size, 3)))
        tf.summary.image('image', image[None])
        return image, label

    def augmentations(self, image, label):
        image = tf.image.random_flip_left_right(image)
        return image, label

    def build_training_data(self):
        ds = tf.data.Dataset.from_generator(
            lambda: self.data_generator(train=True),
            (tf.string, tf.int32),
            ((), ()))
        ds = ds.repeat()
        ds = ds.map(lambda im, l: self.augmentations(*self.neccessary_processing(im, l)), num_parallel_calls=4)
        ds = ds.batch(self.train_batch_size, drop_remainder=True)
        ds = ds.map(lambda x, y: ({'images': x}, y))
        ds = ds.prefetch(1)
        return ds

    def build_validation_data(self):
        ds = tf.data.Dataset.from_generator(
            lambda: self.data_generator(train=False),
            (tf.string, tf.int32),
            ((), ()))
        ds = ds.map(self.neccessary_processing, num_parallel_calls=4)
        ds = ds.batch(self.val_batch_size, drop_remainder=True)
        ds = ds.map(lambda x, y: ({'images': x}, y))
        ds = ds.prefetch(1)
        return ds



class StanfordModel(FixUpResnet):

    map_name = 'Map@1'

    def embedding_net(self, features, params):
        net = tf.layers.dense(features, params['units'], use_bias=False, activation=None)
        net = tf.nn.l2_normalize(net, axis=1)
        return net

    def model_fn(self, features, labels, mode, params):
        features = features['images']
        tf.summary.image('image', features)
        endpoints = self.build_network(features)
        net = endpoints['feature'][:, 0, 0, :]
        embedding = self.embedding_net(net, params)

        betas = tf.get_variable('beta_margins', initializer=params['beta_0']*tf.ones([22634]))
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
            StanfordModel.map_name: tf.metrics.accuracy(labels, estimated)}

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
    StanfordProductData()