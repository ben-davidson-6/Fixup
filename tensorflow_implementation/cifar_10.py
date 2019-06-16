import tensorflow as tf
import tensorflow.estimator as estimator

from resnet_fixup import FixUpResnet


class CIFAR10Model(FixUpResnet):

    def model_fn(self, features, labels, mode, params):
        endpoints = self.build_network(features)
        logits, predictions = endpoints['logits'], endpoints['predictions']
        if mode == estimator.ModeKeys.PREDICT:
            return self.predict_spec(predictions, params)
        loss = self.loss(logits, labels)
        if mode == estimator.ModeKeys.TRAIN:
            return self.train_spec(loss, params)
        if mode == estimator.ModeKeys.EVAL:
            return self.eval_spec(loss, predictions, labels, params)

    #####################################################################
    # Defining train/eval/predict for estimator
    #####################################################################

    def predict_spec(self, predictions, params):
        named_predictions = {
            'probabilites': predictions,
            'top_1': tf.argmax(predictions, axis=1)
        }
        return estimator.EstimatorSpec(
            estimator.ModeKeys.PREDICT,
            predictions=named_predictions)

    def train_spec(self, loss, params):
        train_op = self._training_op(loss, params)
        return estimator.EstimatorSpec(
            estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    def eval_spec(self, loss, predictions, labels, params):
        # Define the metrics:
        metrics_dict = {
            'Accuracy': tf.metrics.accuracy(tf.argmax(predictions, axis=-1), tf.argmax(labels, axis=-1))}

        # return eval spec
        return estimator.EstimatorSpec(
            estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=metrics_dict)

    #####################################################################
    # Helping setup the estimator
    #####################################################################

    def _split_variables_by_bias_scale(self):
        all_vars = tf.trainable_variables()
        normal_vars = [x for x in all_vars if ('fixup_bias' not in x.op.name and 'fixup_scale' not in x.op.name)]
        scale_and_bias_vars = [x for x in all_vars if 'fixup_bias' in x.op.name or 'fixup_scale' in x.op.name]
        return normal_vars, scale_and_bias_vars

    def _training_op(self, loss, params):
        normal_vars, scale_and_bias_vars = self._split_variables_by_bias_scale()
        learning_rate = self._make_learning_rate(params)
        tf.summary.scalar('learning_rate', learning_rate)
        optimiser_normal = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=params['momentum'])
        optimiser_scale_and_bias = tf.train.MomentumOptimizer(
            learning_rate * params['bias_reduction'],
            momentum=params['momentum'])

        grads = tf.gradients(loss, normal_vars + scale_and_bias_vars)
        grads1 = grads[:len(normal_vars)]
        grads2 = grads[len(normal_vars):]
        tran_op_normal = optimiser_normal.apply_gradients(zip(grads1, normal_vars))
        train_op_scale_bias = optimiser_scale_and_bias.apply_gradients(
            zip(grads2, scale_and_bias_vars), global_step=tf.train.get_or_create_global_step())

        train_op = tf.group(tran_op_normal, train_op_scale_bias)
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

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels,
            logits=logits,
        )) + FixUpResnet._weight_decay()

class CIFAR10Dataset():

    def __init__(self, mixup_val, batch_size):
        self.mixup_val = mixup_val
        self.batch_size = batch_size

    def neccessary_processing(self, image, label):
        image = image/255
        image -= tf.constant([0.4914, 0.4822, 0.4465])[None, None]
        image /= tf.constant([0.2023, 0.1994, 0.2010])[None, None]
        label = tf.one_hot(label, 10, dtype=tf.float32)
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

    def mixup(self, images, labels):
        beta = tf.distributions.Beta(self.mixup_val, self.mixup_val)
        images_reversed = tf.reverse(images, [0])
        labels_reversed = tf.reverse(labels, [0])
        lambdas = beta.sample(self.batch_size)
        images = lambdas[:, None, None, None]*images + (1 - lambdas[:, None, None, None])*images_reversed
        labels = lambdas[:, None]*labels + (1 - lambdas[:, None])*labels_reversed
        return images, labels

    def build_training_data(self):
        gen = self.train_generator()
        ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((32, 32, 3), ()))
        ds = ds.shuffle(5000).repeat()
        ds = ds.map(lambda im, l: self.augmentations(*self.neccessary_processing(im, l)), num_parallel_calls=4)
        ds = ds.batch(self.batch_size)
        ds = ds.map(self.mixup)
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
