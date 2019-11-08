
import tensorflow as tf

# import os
# import sys

# sys.path.append('../..')

# from ..batchflow.batchflow.models.tf import EncoderDecoder
# from seismicpro.batchflow.batchflow.models.tf.layers import conv_block

# from seismicpro.batchflow.batchflow.config import Config
from seismicpro.batchflow.batchflow.models.tf import TFModel, UNet, VGG7


class UnetAttParams(TFModel):
    """Class for Unet Attention model."""

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['body/main/base_class'] = UNet
        config['body/att/base_class'] = VGG7

        # main_base_conf = UNet.default_config()
        # att_base_conf = VGG7.default_config()
        #
        # config['body/main'] = {'body': main_base_conf['body'], 'head': main_base_conf['head']}
        # config['body/attn'] = {'body': att_base_conf['body'], 'head': att_base_conf['head']}

        return config

    def build_config(self, names=None):
        config = super().build_config(names)

        main_base_conf = config.get('body/main/base_class').default_config()
        config['body/main/body'] = {**main_base_conf['body'], **config['body/main/body']}
        config['body/main/head'] = {**main_base_conf['head'], **config['body/main/head']}

        att_base_conf = config.get('body/att/base_class').default_config()
        config['body/att/body'] = {**att_base_conf['body'], **config['body/att/body']}
        config['body/att/head'] = {**att_base_conf['head'], **config['body/att/head']}

        # if config.get('body/main/body') is None:
        #     config['body/main/body'] = main_base_conf['body']
        # if config.get('body/main/head') is None:
        #     config['body/main/head'] = main_base_conf['head']
        #
        #
        # if config.get('body/att/body') is None:
        #     config['body/att/body'] = att_base_conf['body']
        # if config.get('body/att/head') is None:
        #     config['body/main/head'] = att_base_conf['head']

        return config

    def body(self, inputs, name='body', *args, **kwargs):
        _ = args
        raw = inputs

        main_config = kwargs.pop('main')
        attn_config = kwargs.pop('att')

        with tf.variable_scope('main_branch'):

            base_class = main_config.pop('base_class')

            main_body = main_config['body']
            main = base_class.body(raw, **{**kwargs, **main_body})
            main_head = main_config['head']
            main = base_class.head(main, **{**kwargs, **main_head})

            #
            # # Get a single channel with linear activation for the main branch
            # main = conv_block(main, layout='c', filters=1, units=1, name='head_main')

        with tf.variable_scope('attention_branch'):

            base_class = attn_config.pop('base_class')

            att_body = attn_config['body']
            att = base_class.body(raw, **{**kwargs, **att_body})
            att_head = attn_config['head']
            att = base_class.head(att, **{**kwargs, **att_head})

            # # Get a single channel with sigmoid activation for the attention branch
            # att = conv_block(att, layout='fa', units=2, name='head_att', activation=h_sigmoid)

        return main, att, raw

    def head(self, inputs, *args, **kwargs):
        _ = args
        main, att, raw = inputs
        self.store_to_attr("out_main", main)

        l0, l1 = tf.split(att, num_or_size_splits=2, axis=1)
        l0 = tf.expand_dims(l0, axis=1)
        l1 = tf.expand_dims(l1, axis=1)

        size_f = tf.cast(tf.shape(raw)[1], 'float')
        arange = tf.range(0, size_f, dtype='float')
        arange = tf.expand_dims(arange, axis=1)
        arange = tf.expand_dims(arange, axis=0)

        y1 = tf.sigmoid((arange - l0 * size_f))
        y2 = tf.sigmoid((l1 * size_f - arange))

        mask = y1 * y2
        self.store_to_attr("mask", mask)

        out = raw * (1 - mask) + main * mask
        self.store_to_attr("out_lift", out)

        return tf.stack([out, mask], axis=0)
