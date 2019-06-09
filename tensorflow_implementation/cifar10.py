import tensorflow as tf


class CIFAR10():

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
