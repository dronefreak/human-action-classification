from __future__ import absolute_import

import tensorflow as tf

from tf_pose import network_base


class MobilenetNetwork(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0, conv_width2=None):
        self.conv_width = conv_width
        self.conv_width2 = conv_width2 if conv_width2 else conv_width
        self.num_refine = 4
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    def setup(self):
        min_depth = 8
        depth = lambda d: max(int(d * self.conv_width), min_depth)
        depth2 = lambda d: max(int(d * self.conv_width2), min_depth)

        with tf.variable_scope(None, 'MobilenetV1'):
            (self.feed('image')
             .convb(3, 3, depth(32), 2, name='Conv2d_0')
             .separable_conv(3, 3, depth(64), 1, name='Conv2d_1')
             .separable_conv(3, 3, depth(128), 2, name='Conv2d_2')
             .separable_conv(3, 3, depth(128), 1, name='Conv2d_3')
             .separable_conv(3, 3, depth(256), 2, name='Conv2d_4')
             .separable_conv(3, 3, depth(256), 1, name='Conv2d_5')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_6')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_7')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_8')
             # .separable_conv(3, 3, depth(512), 1, name='Conv2d_9')
             # .separable_conv(3, 3, depth(512), 1, name='Conv2d_10')
             # .separable_conv(3, 3, depth(512), 1, name='Conv2d_11')
             # .separable_conv(3, 3, depth(1024), 2, name='Conv2d_12')
             # .separable_conv(3, 3, depth(1024), 1, name='Conv2d_13')
             )

        (self.feed('Conv2d_1').max_pool(2, 2, 2, 2, name='Conv2d_1_pool'))
        (self.feed('Conv2d_7').upsample(2, name='Conv2d_7_upsample'))

        (self.feed('Conv2d_1_pool', 'Conv2d_3', 'Conv2d_7_upsample')
            .concat(3, name='feat_concat'))

        feature_lv = 'feat_concat'
        with tf.variable_scope(None, 'Openpose'):
            prefix = 'MConv_Stage1'
            (self.feed(feature_lv)
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_1')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_2')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_3')
             .separable_conv(1, 1, depth2(512), 1, name=prefix + '_L1_4')
             .separable_conv(1, 1, 38, 1, relu=False, name=prefix + '_L1_5'))

            (self.feed(feature_lv)
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_1')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_2')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_3')
             .separable_conv(1, 1, depth2(512), 1, name=prefix + '_L2_4')
             .separable_conv(1, 1, 19, 1, relu=False, name=prefix + '_L2_5'))

            for stage_id in range(self.num_refine):
                prefix_prev = 'MConv_Stage%d' % (stage_id + 1)
                prefix = 'MConv_Stage%d' % (stage_id + 2)
                (self.feed(prefix_prev + '_L1_5',
                           prefix_prev + '_L2_5',
                           feature_lv)
                 .concat(3, name=prefix + '_concat')
                 .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L1_1')
                 .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L1_2')
                 .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L1_3')
                 .separable_conv(1, 1, depth2(128), 1, name=prefix + '_L1_4')
                 .separable_conv(1, 1, 38, 1, relu=False, name=prefix + '_L1_5'))

                (self.feed(prefix + '_concat')
                 .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L2_1')
                 .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L2_2')
                 .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L2_3')
                 .separable_conv(1, 1, depth2(128), 1, name=prefix + '_L2_4')
                 .separable_conv(1, 1, 19, 1, relu=False, name=prefix + '_L2_5'))

            # final result
            (self.feed('MConv_Stage%d_L2_5' % self.get_refine_num(),
                       'MConv_Stage%d_L1_5' % self.get_refine_num())
             .concat(3, name='concat_stage7'))

    def loss_l1_l2(self):
        l1s = []
        l2s = []
        for layer_name in sorted(self.layers.keys()):
            if '_L1_5' in layer_name:
                l1s.append(self.layers[layer_name])
            if '_L2_5' in layer_name:
                l2s.append(self.layers[layer_name])

        return l1s, l2s

    def loss_last(self):
        return self.get_output('MConv_Stage%d_L1_5' % self.get_refine_num()), \
               self.get_output('MConv_Stage%d_L2_5' % self.get_refine_num())

    def restorable_variables(self):
        vs = {v.op.name: v for v in tf.global_variables() if
              'MobilenetV1/Conv2d' in v.op.name and
              'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and 'Ada' not in v.op.name
              }
        return vs

    def get_refine_num(self):
        return self.num_refine + 1
