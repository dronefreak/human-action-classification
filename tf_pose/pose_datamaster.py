import argparse
import logging
import time

from tensorpack.dataflow.remote import RemoteDataZMQ

from tf_pose.pose_dataset import CocoPose

logging.basicConfig(level=logging.DEBUG, format='[lmdb_dataset] %(asctime)s %(levelname)s %(message)s')

if __name__ == '__main__':
    """
    Speed Test for Getting Input batches from other nodes
    """
    parser = argparse.ArgumentParser(description='Worker for preparing input batches.')
    parser.add_argument('--listen', type=str, default='tcp://0.0.0.0:1027')
    parser.add_argument('--show', type=bool, default=False)
    args = parser.parse_args()

    df = RemoteDataZMQ(args.listen)

    logging.info('tcp queue start')
    df.reset_state()
    t = time.time()
    for i, dp in enumerate(df.get_data()):
        if i == 100:
            break
        logging.info('Input batch %d received.' % i)
        if i == 0:
            for d in dp:
                logging.info('%d dp shape={}'.format(d.shape))

        if args.show:
            CocoPose.display_image(dp[0][0], dp[1][0], dp[2][0])

    logging.info('Speed Test Done for 100 Batches in %f seconds.' % (time.time() - t))
