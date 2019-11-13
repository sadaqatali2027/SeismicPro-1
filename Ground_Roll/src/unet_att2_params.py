""" Class for Unet Attention with attention branch that learns rectangular mask parameters. """

import tensorflow as tf
from seismicpro.batchflow.batchflow.models.tf import TFModel, UNet, VGG7


class UnetAttParams(TFModel):
    """
    Class for Unet Attention with attention branch that learns rectangular mask parameters.

    baseline config:
    ```
    model_config = {
        'initial_block/inputs': 'trace_raw',
        'inputs': dict(trace_raw={'shape': (3000, 1)},
                       lift={'name': 'targets', 'shape': (3000, 1)}),
        'loss': (attention_loss_gauss, {'balance': 0.05}),
        'optimizer': ('Adam', {'learning_rate': 0.0001}),
        'common/data_format': 'channels_last',
        'common/main_base_class': UNet,
        'common/att_base_class': VGG7,
        'body': {
            'main': {
                'encoder/blocks': dict(layout='ca ca',
                                       filters=[16, 32, 64, 128],
                                       kernel_size=[7, 5, 5, 5],
                                       activation=tf.nn.elu),
                'encoder/downsample': dict(layout='pd',
                                           pool_size=2,
                                           pool_strides=2,
                                           dropout_rate=0.05),
                'embedding': dict(layout='ca ca', kernel_size=5, filters=256),
                'decoder/blocks': dict(layout='ca ca',
                                       filters=[16, 32, 64, 128][::-1],
                                       kernel_size=[7, 5, 5, 5][::-1],
                                       activation=tf.nn.elu),
                'decoder/upsample': dict(layout='tad',
                                         kernel_size=[7, 5, 5, 5][::-1],
                                         strides=2,
                                         dropout_rate=0.05,
                                         activation=tf.nn.elu, ),
            },
            'att': {},
        },
        'head': {
            'main': dict(layout='c', filters=1, units=1),
            'att': dict(layout='fa', units=2, activation=h_sigmoid),
        }
    }
    ```
    """

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/main_base_class'] = UNet
        config['common/att_base_class'] = VGG7

        return config

    def body(self, inputs, name='body', *args, **kwargs):
        _ = args
        raw = inputs

        main_config = kwargs.pop('main')
        attn_config = kwargs.pop('att')

        with tf.variable_scope('main_branch'):
            base_class = kwargs['main_base_class']
            main = base_class.body(raw, *args, **{**kwargs, **main_config})

        with tf.variable_scope('attention_branch'):
            base_class = kwargs['att_base_class']
            att = base_class.body(raw, *args, **{**kwargs, **attn_config})

        return main, att, raw

    def head(self, inputs, *args, **kwargs):
        _ = args
        main, att, raw = inputs

        main_config = kwargs.pop('main')
        attn_config = kwargs.pop('att')

        with tf.variable_scope('main_head'):
            base_class = kwargs['main_base_class']
            main = base_class.head(main, *args, **{**kwargs, **main_config})

        self.store_to_attr("out_main", main)

        with tf.variable_scope('attention_head'):
            base_class = kwargs['att_base_class']
            att = base_class.head(att, *args, **{**kwargs, **attn_config})

        with tf.variable_scope('apply_mask'):
            l0, l1 = tf.split(att, num_or_size_splits=2, axis=1)
            l0 = tf.expand_dims(l0, axis=1)
            l1 = tf.expand_dims(l1, axis=1)

            size_f = tf.cast(tf.shape(raw)[1], 'float')

            arange = tf.range(0, size_f, dtype='float')
            arange = tf.expand_dims(arange, axis=1)
            arange = tf.expand_dims(arange, axis=0)

            m0 = tf.sigmoid(arange - l0 * size_f)
            m1 = tf.sigmoid(l1 * size_f - arange)

            mask = m0 * m1
            
            mode = kwargs.get('mode', 'filter')
            if mode == 'noise':
                out = raw - main * mask
            else:
                out = raw * (1 - mask) + main * mask

        self.store_to_attr("mask", mask)
        self.store_to_attr("out_lift", out)

        return tf.stack([out, mask], axis=0)
