""" UnetAttention model """
import tensorflow as tf

# import os
# import sys

# sys.path.append('../..')

# from ..batchflow.batchflow.models.tf import EncoderDecoder
from seismicpro.batchflow.batchflow.models.tf.layers import conv_block
# from ..batchflow.batchflow.models.utils import unpack_args

from seismicpro.models.unet_attention import UnetAttGauss1, EncoderDecoderWithBranch


class UnetAttGauss(UnetAttGauss1):
    """Class for Unet Attention model."""

    def body(self, inputs, name='body', *args, **kwargs):
        _ = args
        raw = inputs

        main_config = kwargs.pop('main')
        attn_config = kwargs.pop('attn')

        with tf.variable_scope('main_branch'):
            main = EncoderDecoderWithBranch.body(raw, **{**kwargs, **main_config}) # pylint: disable=not-a-mapping

            #Get a single channel with linear activation for the main branch
            main = conv_block(main, layout='c', filters=1, units=1, name='head_main')

        with tf.variable_scope('attention_branch'):
            att = EncoderDecoderWithBranch.body(raw, **{**kwargs, **attn_config}) # pylint: disable=not-a-mapping

            print('before head_att', att)

            att = conv_block(att, layout='P', keepdims=True, name='pooling')
            print('after global pooling', att)

            att = conv_block(att, layout='c', kernel_size=1, filters=2, keepdims=True,
                             name='head_att')

            print('after head_att', att)
            #Get a single channel with sigmoid activation for the attention branch

        return main, att, raw

    def head(self, inputs, *args, **kwargs):
        _ = args
        main, att, raw = inputs

        print("att", att)
        mu, sigma = tf.split(att, num_or_size_splits=2, axis=2)
        print("mu", mu)
        print("sigma", sigma)

        print("raw", raw)
        # sigm_x = tf.fill(tf.shape(raw), 0.0)
        arange = tf.range(0, tf.cast(tf.shape(raw)[1], 'float'), dtype='float')
        print("arange", arange)
        arange = tf.expand_dims(arange, axis=1)
        # arange = arange + sigm_x

        #         arange = tf.expand_dims(arange, axis=0)
        print("arange", arange)
        print("arange-mu", arange - mu)

        # Apply sigmoid function to the above obtained domain
        attention_gaussian = sigma * tf.exp(-tf.square(arange - mu))
        print("attention_gaussian soft", attention_gaussian)
        attention_gaussian = tf.sigmoid(100 * (attention_gaussian - 0.5))

        print("attention_gaussian hard", attention_gaussian)

        self.store_to_attr("attention_gaussian", attention_gaussian)
        self.store_to_attr("main", attention_gaussian)
        self.store_to_attr("mu", attention_gaussian)
        self.store_to_attr("sigma", attention_gaussian)

        # Get a model output that is a superposition of raw input and main branches
        # according to attention mask
        out_lift = raw * (1 - attention_gaussian) + main * (attention_gaussian)
        print("out_lift", out_lift)
        self.store_to_attr("out_lift", out_lift)

        print(tf.stack([out_lift, attention_gaussian], axis=0))

        return tf.stack([out_lift, attention_gaussian], axis=0)
