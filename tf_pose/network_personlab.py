from __future__ import absolute_import

from tf_pose import network_base
from tf_pose.slim.nets.resnet_v2 import resnet_v2_101


class PersonLabNetwork(network_base.BaseNetwork):
    """
    Reference : PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model

    pretrained architecture * weights from :
        https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
    """

    def __init__(self, inputs, trainable=True, backbone='resnet152'):
        """
        :param inputs:
        :param backbone: resnet101, resnet152, mobilenet-v2-1.0
        """
        self.backbone = backbone
        super().__init__(inputs, trainable)

    def setup(self):
        if self.backbone == 'resnet101':
            net, end_points = resnet_v2_101(self.inputs, is_training=self.trainable, global_pool=False,
                                            output_stride=16)
            pass
        pass

