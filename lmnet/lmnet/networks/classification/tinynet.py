import functools

import tensorflow as tf

from lmnet.blocks import lmnet_block
from lmnet.networks.classification.base import Base

class TinyNet(Base):
    """
    This network is only for Tutorial, Not for Production.
    TinyNet contains 5 CONV layers and 2 POOL layers.
    """
    version = 0.1

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.activation = tf.nn.relu

    def _get_lmnet_block(self, is_training, channels_data_format):
        return functools.partial(lmnet_block,
                                 activation=self.activation,
                                 custom_getter=self.custom_getter,
                                 is_training=is_training,
                                 is_debug=self.is_debug,
                                 use_bias=False,
                                 data_format=channels_data_format)

    def _space_to_depth(self, inputs=None, block_size=2, name=''):
        if self.data_format != 'NHWC':
            inputs = tf.transpose(inputs, perm=[self.data_format.find(d) for d in 'NHWC'])

        output = tf.space_to_depth(inputs, block_size=block_size, name=name)

        if self.data_format != 'NHWC':
            output = tf.transpose(output, perm=['NHWC'.find(d) for d in self.data_format])
        return output

    def base(self, images, is_training, *args, **kwargs):
        """Base network.

        Args:
            images: Input images.
            is_training: A flag for if is training.
        Returns:
            tf.Tensor: Inference result.
        """

        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        _lmnet_block = self._get_lmnet_block(is_training, channels_data_format)

        self.images = images

        x = _lmnet_block('conv1', images, 32, 3)
        x = _lmnet_block('conv2', x, 64, 3)
        x = self._space_to_depth(name='pool1', inputs=x)
        x = _lmnet_block('conv3', x, 128, 3)
        x = _lmnet_block('conv4', x, 64, 1, activation=tf.nn.relu)

        x = tf.layers.dropout(x, training=is_training)

        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        x = tf.layers.conv2d(name='conv5',
                             inputs=x,
                             filters=self.num_classes,
                             kernel_size=1,
                             kernel_initializer=kernel_initializer,
                             activation=None,
                             use_bias=True,
                             data_format=channels_data_format)

        self._heatmap_layer = x

        h = x.get_shape()[1].value if self.data_format == 'NHWC' else x.get_shape()[2].value
        w = x.get_shape()[2].value if self.data_format == 'NHWC' else x.get_shape()[3].value
        x = tf.layers.average_pooling2d(name='pool2',
                                        inputs=x,
                                        pool_size=[h, w],
                                        padding='VALID',
                                        strides=1,
                                        data_format=channels_data_format)

        self.base_output = tf.reshape(x, [-1, self.num_classes], name='pool7_reshape')

        return self.base_output


class TinyNetQuantize(TinyNet):
    version = 1.0

    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        assert weight_quantizer
        assert activation_quantizer

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               weight_quantization=weight_quantization)

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            args: Args.
            kwargs: Kwargs.
        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var
