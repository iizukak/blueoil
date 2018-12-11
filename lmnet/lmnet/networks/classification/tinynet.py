import functools

import tensorflow as tf

from lmnet.blocks import lmnet_block
from lmnet.networks.classification.base import Base

class TinyNet(Base):
    """
    This network is only for Tutorial, Not for Production.
    TinyNet contains 4 (CONV+POOL) layers.
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


    def base(self, images, is_training, *args, **kwargs):
        """Base network.

        Args:
            images: Input images.
            is_training: A flag for if is training.
        Returns:
            tf.Tensor: Inference result.
        """

        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        self.images = images

        x = tf.layers.conv2d(name="conv1",
                             inputs=images,
                             filters=16,
                             kernel_size=3,
                             activation=self.activation,
                             use_bias=False,
                             data_format=channels_data_format)

        x = tf.layers.conv2d(name="conv2",
                             inputs=x,
                             filters=32,
                             kernel_size=3,
                             activation=self.activation,
                             use_bias=False,
                             data_format=channels_data_format)

        if self.data_format != 'NHWC':
            x = tf.transpose(x, perm=[self.data_format.find(d) for d in 'NHWC'])
        x = tf.space_to_depth(x, block_size=2, name="space_to_depth")

        x = tf.layers.conv2d(name="conv3",
                             inputs=x,
                             filters=64,
                             kernel_size=3,
                             activation=self.activation,
                             use_bias=False,
                             data_format=channels_data_format)

        four_letter_data_format = 'NHWC' if channels_data_format == 'channels_last' else 'NCHW'
        x = tf.contrib.layers.batch_norm(x,
                                         decay=0.99,
                                         scale=True,
                                         center=True,
                                         updates_collections=None,
                                         is_training=is_training,
                                         data_format=four_letter_data_format)

        x = tf.layers.dropout(x, training=is_training)

        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        x = tf.layers.conv2d(name='conv4',
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
