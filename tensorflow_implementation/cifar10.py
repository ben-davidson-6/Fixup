import tensorflow as tf


class CIFAR10():

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
        ds = ds.shuffle(1000).repeat()
        ds = ds.map(lambda im, l: self.augmentations(*self.neccessary_processing(im, l)), num_parallel_calls=4)
        ds = ds.batch(256)
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
        ds = ds.batch(256)
        ds = ds.prefetch(1)
        return ds

    def build_pipeline(self):
        train_dataset = self.build_training_data()
        val_dataset = self.build_validation_data()
        self.add_tf_dataset('training', train_dataset)
        self.add_tf_dataset('validation', val_dataset)
        image, label = self.iterator.get_next()
        self.add_gettable_tensor('images', image)
        self.add_gettable_tensor('labels', label)

    def is_training_set(self, dataset_name):
        return 'training' in dataset_name


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # print(x_train[0])
    pass