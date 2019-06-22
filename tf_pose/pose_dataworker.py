import argparse

from tensorpack.dataflow.remote import send_dataflow_zmq

from tf_pose.pose_dataset import get_dataflow_batch
from tf_pose.pose_augment import set_network_input_wh, set_network_scale

if __name__ == '__main__':
    """
    OpenPose Data Preparation might be a bottleneck for training.
    You can run multiple workers to generate input batches in multi-nodes to make training process faster.
    """
    parser = argparse.ArgumentParser(description='Worker for preparing input batches.')
    parser.add_argument('--datapath', type=str, default='/coco/annotations/')
    parser.add_argument('--imgpath', type=str, default='/coco/')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--master', type=str, default='tcp://csi-cluster-gpu20.dakao.io:1027')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--scale-factor', type=int, default=2)
    args = parser.parse_args()

    set_network_input_wh(args.input_width, args.input_height)
    set_network_scale(args.scale_factor)

    df = get_dataflow_batch(args.datapath, args.train, args.batchsize, args.imgpath)

    send_dataflow_zmq(df, args.master, hwm=10)
