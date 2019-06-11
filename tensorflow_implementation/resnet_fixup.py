import tensorflow as tf
import tensorflow.estimator as estimator


class Block:
    def __init__(self, units, depth, starting_layer):
        self.units = units
        self.depth = depth
        self.starting_layer = starting_layer


class FixUpResnet():

    def __init__(self, classes=10):
        self.blocks = []
        self.classes = classes

    def build_network(self, images):
        net = images
        # 3x3 conv c with relu
        # relu(net*c + b)
        net = self.pre_residual_blocks(net)

        # Build each residual block, from the
        # provided configuration, and stack them together
        net = self.block_graph(net)

        # remove any spatial dimensions leftover by average
        # pooling
        net = tf.reduce_mean(net, [1, 2], keepdims=True)

        # push through fully connected layer and get softmax
        logits = self.fc_layer(net)
        predictions = tf.nn.softmax(logits, axis=1)
        return logits, predictions

    def model_fn(self, features, labels, mode, params):
        logits, predictions = self.build_network(features)
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

    #####################################################################
    # Build the network
    #####################################################################

    def number_of_residual_layers(self):
        starting_layer = 0
        for b in self.blocks:
            starting_layer += b.units
        return starting_layer

    def add_block(self, units, depth):
        starting_layer = self.number_of_residual_layers()
        self.blocks.append(Block(units, depth, starting_layer))

    def get_block_channels(self, block):
        return self.blocks[block].depth

    @staticmethod
    def _he_initialiser(n):
        he_init = tf.sqrt(2. / n)
        weights_init = tf.random_normal_initializer(
            mean=0.,
            stddev=he_init)
        return weights_init

    def _fixup_init(self, n):
        he_init = tf.sqrt(2. / n)
        m = 3
        scale_coeff = self.number_of_residual_layers() ** (-1/(2*m - 2))
        weight_std = he_init * scale_coeff
        weights_init = tf.random_normal_initializer(
            mean=0.,
            stddev=weight_std)
        return weights_init

    def pre_residual_blocks(self, net):
        depth = self.get_block_channels(0)
        bias_0 = tf.Variable(initial_value=0., name='fixup_bias')
        net = self.convolve(
            net,
            fsize=3,
            in_channel=3,
            out_channel=depth,
            stride=1,
            pad='SAME',
            initer='he'
        )
        net = tf.nn.relu(net + bias_0)
        return net

    @staticmethod
    def shortcut_connection(net, downsample=False):
        if downsample:
            shortcut = tf.nn.avg_pool(
                net,
                ksize=[1, 1, 1, 1],
                strides=[1, 2, 2, 1],
                padding='VALID')
            shortcut = tf.concat([shortcut, tf.zeros_like(shortcut)], axis=3)
        else:
            shortcut = net
        return shortcut

    def convolve(self, net, fsize, in_channel, out_channel, stride, pad, initer):
        n = out_channel*fsize**2
        if initer == 'he':
            initer = FixUpResnet._he_initialiser(n)
        elif initer == 'fixup':
            initer = self._fixup_init(n)
        elif initer == 'zero':
            initer = tf.zeros_initializer()
        else:
            raise ValueError('no initer named {}'.format(initer))
        w = tf.Variable(
            initial_value=initer([fsize, fsize, in_channel, out_channel]),
        )
        return tf.nn.conv2d(net, w, [1, stride, stride, 1], pad)

    def build_unit_graph(self, net, block_depth, downsample):
        bottleneck_depth = block_depth // 2
        start_channels = block_depth//2 if downsample else block_depth

        bias_0 = tf.Variable(initial_value=0., name='fixup_bias_0')
        bias_1 = tf.Variable(initial_value=0., name='fixup_bias_1')
        bias_2 = tf.Variable(initial_value=0., name='fixup_bias_2')
        bias_3 = tf.Variable(initial_value=0., name='fixup_bias_3')
        bias_4 = tf.Variable(initial_value=0., name='fixup_bias_4')
        bias_5 = tf.Variable(initial_value=0., name='fixup_bias_5')
        scale_0 = tf.Variable(initial_value=1., name='fixup_scale')

        shortcut = FixUpResnet.shortcut_connection(net, downsample)
        shortcut = shortcut + bias_0

        # block unit
        # conv relu conv relu conv
        net = self.convolve(
            net + bias_0,
            fsize=1,
            in_channel=start_channels,
            out_channel=bottleneck_depth,
            stride=1 if not downsample else 2,
            pad='SAME',
            initer='fixup')
        net = tf.nn.relu(net + bias_1)
        net = self.convolve(
            net + bias_2,
            fsize=3,
            in_channel=bottleneck_depth,
            out_channel=bottleneck_depth,
            stride=1,
            pad='SAME',
            initer='fixup')
        net = tf.nn.relu(net + bias_3)
        net = self.convolve(
            net + bias_4,
            fsize=1,
            in_channel=bottleneck_depth,
            out_channel=block_depth,
            stride=1,
            pad='SAME',
            initer='zero')
        net = scale_0*net + bias_5
        return tf.nn.relu(net + shortcut)

    def build_single_block_graph(self, b, net, subsample_first=False):
        for unit in range(b.units):
            net = self.build_unit_graph(net, b.depth, subsample_first)
            subsample_first = False
        return net

    def block_graph(self, net):
        for block_num, block in enumerate(self.blocks):
            first_block = block_num == 0
            net = self.build_single_block_graph(block, net, subsample_first=(not first_block))
        return net

    def fc_layer(self, net):
        out_depth = self.get_block_channels(-1)
        w = tf.get_variable(
            'fc_kernel',
            initializer=tf.zeros([1, 1, out_depth, self.classes]))
        b_0 = tf.Variable(initial_value=0., name='fixup_bias_fc_0')
        net = net + b_0
        net = tf.nn.conv2d(net, w, [1, 1, 1, 1], 'VALID')
        return net[:, 0, 0, :]

    @staticmethod
    def _weight_decay():
        vars = tf.trainable_variables()
        non_bias_vars = [
            tf.nn.l2_loss(v) for v in vars
            if 'bias' not in v.name and 'scale' not in v.name]
        l2_loss = tf.add_n(non_bias_vars)/len(non_bias_vars) * 0.0001
        return l2_loss

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels,
            logits=logits,
        )) + FixUpResnet._weight_decay()
