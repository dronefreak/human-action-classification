from __future__ import absolute_import

from tf_pose import network_base


class DSConvNetwork(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0):
        self.conv_width = conv_width
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    def setup(self):
        (self.feed('image')
         .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
         # .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=True)     # TODO
         .separable_conv(3, 3, round(self.conv_width * 64), 2, name='conv1_2')
         # .max_pool(2, 2, 2, 2, name='pool1_stage1')
         .separable_conv(3, 3, round(self.conv_width * 128), 1, name='conv2_1')
         .separable_conv(3, 3, round(self.conv_width * 128), 2, name='conv2_2')
         # .max_pool(2, 2, 2, 2, name='pool2_stage1')
         .separable_conv(3, 3, round(self.conv_width * 256), 1, name='conv3_1')
         .separable_conv(3, 3, round(self.conv_width * 256), 1, name='conv3_2')
         .separable_conv(3, 3, round(self.conv_width * 256), 1, name='conv3_3')
         .separable_conv(3, 3, round(self.conv_width * 256), 2, name='conv3_4')
         # .max_pool(2, 2, 2, 2, name='pool3_stage1')
         .separable_conv(3, 3, round(self.conv_width * 512), 1, name='conv4_1')
         .separable_conv(3, 3, round(self.conv_width * 512), 1, name='conv4_2')
         .separable_conv(3, 3, round(self.conv_width * 256), 1, name='conv4_3_CPM')
         .separable_conv(3, 3, 128, 1, name='conv4_4_CPM')
         .separable_conv(3, 3, round(self.conv_width * 128), 1, name='conv5_1_CPM_L1')
         .separable_conv(3, 3, round(self.conv_width * 128), 1, name='conv5_2_CPM_L1')
         .separable_conv(3, 3, round(self.conv_width * 128), 1, name='conv5_3_CPM_L1')
         .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L1')
         .conv(1, 1, 38, 1, 1, relu=False, name='conv5_5_CPM_L1'))

        (self.feed('conv4_4_CPM')
         .separable_conv(3, 3, round(self.conv_width * 128), 1, name='conv5_1_CPM_L2')
         .separable_conv(3, 3, round(self.conv_width * 128), 1, name='conv5_2_CPM_L2')
         .separable_conv(3, 3, round(self.conv_width * 128), 1, name='conv5_3_CPM_L2')
         .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L2')
         .conv(1, 1, 19, 1, 1, relu=False, name='conv5_5_CPM_L2'))

        (self.feed('conv5_5_CPM_L1',
                   'conv5_5_CPM_L2',
                   'conv4_4_CPM')
         .concat(3, name='concat_stage2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv1_stage2_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv2_stage2_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv3_stage2_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv4_stage2_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv5_stage2_L1')
         .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L1')
         .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage2_L1'))

        (self.feed('concat_stage2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv1_stage2_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv2_stage2_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv3_stage2_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv4_stage2_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv5_stage2_L2')
         .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L2')
         .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage2_L2'))

        (self.feed('Mconv7_stage2_L1',
                   'Mconv7_stage2_L2',
                   'conv4_4_CPM')
         .concat(3, name='concat_stage3')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv1_stage3_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv2_stage3_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv3_stage3_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv4_stage3_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv5_stage3_L1')
         .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L1')
         .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage3_L1'))

        (self.feed('concat_stage3')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv1_stage3_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv2_stage3_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv3_stage3_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv4_stage3_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv5_stage3_L2')
         .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L2')
         .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage3_L2'))

        (self.feed('Mconv7_stage3_L1',
                   'Mconv7_stage3_L2',
                   'conv4_4_CPM')
         .concat(3, name='concat_stage4')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv1_stage4_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv2_stage4_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv3_stage4_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv4_stage4_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv5_stage4_L1')
         .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L1')
         .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage4_L1'))

        (self.feed('concat_stage4')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv1_stage4_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv2_stage4_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv3_stage4_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv4_stage4_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv5_stage4_L2')
         .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L2')
         .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage4_L2'))

        (self.feed('Mconv7_stage4_L1',
                   'Mconv7_stage4_L2',
                   'conv4_4_CPM')
         .concat(3, name='concat_stage5')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv1_stage5_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv2_stage5_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv3_stage5_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv4_stage5_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv5_stage5_L1')
         .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L1')
         .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage5_L1'))

        (self.feed('concat_stage5')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv1_stage5_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv2_stage5_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv3_stage5_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv4_stage5_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv5_stage5_L2')
         .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L2')
         .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage5_L2'))

        (self.feed('Mconv7_stage5_L1',
                   'Mconv7_stage5_L2',
                   'conv4_4_CPM')
         .concat(3, name='concat_stage6')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv1_stage6_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv2_stage6_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv3_stage6_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv4_stage6_L1')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv5_stage6_L1')
         .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L1')
         .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage6_L1'))

        (self.feed('concat_stage6')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv1_stage6_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv2_stage6_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv3_stage6_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv4_stage6_L2')
         .separable_conv(7, 7, round(self.conv_width * 128), 1, name='Mconv5_stage6_L2')
         .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L2')
         .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage6_L2'))

        (self.feed('Mconv7_stage6_L2',
                   'Mconv7_stage6_L1')
         .concat(3, name='concat_stage7'))
