# Resnets with fixup initialisation

Tensorflow implementation of [preactivation resnets](https://arxiv.org/pdf/1603.05027.pdf) with [fixup intialisation](https://arxiv.org/abs/1901.09321). Resnets can be created by defining the recurrent blocks, their depth, and their number of bottleneck units. 

## To use the network
```python
from resnet_fixup import FixUpResnet
import tensorflow as tf

# define a placeholder or dataset iterator
image = tf.placeholder(tf.float32, [None, 32, 32, 3])
resnet = FixUpResnet(classes=10)
resnet.add_block(units=3, depth=64)
resnet.add_block(units=4, depth=128)
resnet.add_block(units=6, depth=256)
resnet.add_block(units=3, depth=512)

logits, predictions = resnet.build_network(image)
```

## Training results

- Training ResNet50 with mixup gives 6.5% validation error after 200 epochs