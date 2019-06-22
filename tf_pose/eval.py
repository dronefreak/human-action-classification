import sys
import os
import numpy as np
import logging
import argparse
import json, re
from tqdm import tqdm

from common import read_imgfile
from estimator import TfPoseEstimator
from networks import model_wh, get_graph_path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

eval_size = -1


def round_int(val):
    return int(round(val))


def write_coco_json(human, image_w, image_h):
    keypoints = []
    coco_ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    for coco_id in coco_ids:
        if coco_id not in human.body_parts.keys():
            keypoints.extend([0, 0, 0])
            continue
        body_part = human.body_parts[coco_id]
        keypoints.extend([round_int(body_part.x * image_w), round_int(body_part.y * image_h), 2])
    return keypoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--resize', type=str, default='0x0', help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=8.0, help='if provided, resize heatmaps before they are post-processed. default=8.0')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
    parser.add_argument('--cocoyear', type=str, default='2014')
    parser.add_argument('--coco-dir', type=str, default='/data/public/rw/coco/')
    parser.add_argument('--data-idx', type=int, default=-1)
    parser.add_argument('--multi-scale', type=bool, default=False)
    args = parser.parse_args()

    cocoyear_list = ['2014', '2017']
    if args.cocoyear not in cocoyear_list:
        logger.error('cocoyear should be one of %s' % str(cocoyear_list))
        sys.exit(-1)

    # TODO : Scales

    image_dir = args.coco_dir + 'val%s/' % args.cocoyear
    coco_json_file = args.coco_dir + 'annotations/person_keypoints_val%s.json' % args.cocoyear
    cocoGt = COCO(coco_json_file)
    catIds = cocoGt.getCatIds(catNms=['person'])
    keys = cocoGt.getImgIds(catIds=catIds)
    if args.data_idx < 0:
        if eval_size > 0:
            keys = keys[:eval_size]  # only use the first #eval_size elements.
        pass
    else:
        keys = [keys[args.data_idx]]
    logger.info('validation %s set size=%d' % (coco_json_file, len(keys)))
    write_json = 'etcs/%s_%s_%f.json' % (args.model, args.resize, args.resize_out_ratio)

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    result = []
    for i, k in enumerate(tqdm(keys)):
        img_meta = cocoGt.loadImgs(k)[0]
        img_idx = img_meta['id']

        img_name = os.path.join(image_dir, img_meta['file_name'])
        image = read_imgfile(img_name, None, None)
        if image is None:
            logger.error('image not found, path=%s' % img_name)
            sys.exit(-1)

        # inference the image with the specified network
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        scores = 0
        ann_idx = cocoGt.getAnnIds(imgIds=[img_idx], catIds=[1])
        anns = cocoGt.loadAnns(ann_idx)
        for human in humans:
            item = {
                'image_id': img_idx,
                'category_id': 1,
                'keypoints': write_coco_json(human, img_meta['width'], img_meta['height']),
                'score': human.score
            }
            result.append(item)
            scores += item['score']

        avg_score = scores / len(humans) if len(humans) > 0 else 0
        if args.data_idx >= 0:
            logger.info('score:', k, len(humans), len(anns), avg_score)

            import matplotlib.pyplot as plt
            fig = plt.figure()
            a = fig.add_subplot(2, 3, 1)
            plt.imshow(e.draw_humans(image, humans, True))

            a = fig.add_subplot(2, 3, 2)
            # plt.imshow(cv2.resize(image, (e.heatMat.shape[1], e.heatMat.shape[0])), alpha=0.5)
            tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
            plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            tmp2 = e.pafMat.transpose((2, 0, 1))
            tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

            a = fig.add_subplot(2, 3, 4)
            a.set_title('Vectormap-x')
            # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            a = fig.add_subplot(2, 3, 5)
            a.set_title('Vectormap-y')
            # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            plt.show()

    fp = open(write_json, 'w')
    json.dump(result, fp)
    fp.close()

    cocoDt = cocoGt.loadRes(write_json)
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = keys
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print(''.join(["%11.4f |" % x for x in cocoEval.stats]))

    pred = json.load(open(write_json, 'r'))
