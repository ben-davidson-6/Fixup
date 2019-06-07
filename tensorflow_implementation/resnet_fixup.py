from tensorflow_tools.network import Network
import tensorflow as tf


class Block:
    def __init__(self, units, depth, starting_layer):
        self.units = units
        self.depth = depth
        self.starting_layer = starting_layer


class FixUpResnet(Network):

    def __init__(self, classes=10):
        Network.__init__(self)
        self.blocks = []
        self.classes = classes

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

    def build_network(self, inputs, **kwargs):
        if type(inputs) == dict:
            net = inputs['images']
        else:
            net = inputs
        net = self.pre_residual_blocks(net)
        net = self.block_graph(net)
        net = tf.reduce_mean(net, [1, 2], keepdims=True)
        net = self.fc_layer(net)
        tf.identity(net, name='logits')
        return net

    @staticmethod
    def _weight_decay():
        vars = tf.trainable_variables()
        non_bias_vars = [
            tf.nn.l2_loss(v) for v in vars
            if 'bias' not in v.name and 'scale' not in v.name]
        l2_loss = tf.add_n(non_bias_vars)/len(non_bias_vars) * 0.0001
        return l2_loss

    @staticmethod
    def loss():
        g = tf.get_default_graph()
        logits = g.get_tensor_by_name('logits:0')
        labels = g.get_tensor_by_name('labels:0')
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(labels, 10, dtype=tf.float32),
            logits=logits,
        )) + FixUpResnet._weight_decay()
