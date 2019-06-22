import argparse
import logging
import time
import os
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.label_image as label_img
import scripts.label_image_scene as label_img_scene

logger = logging.getLogger('Pose_Action_and_Scene_Understanding')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
address = '/home/caffemodel/coursework/CVPR/pose-estimation-detection/images-3/'
file_list = os.listdir(address)
true_label_pose = []
true_label_scene = []
img_name = []
img_count = 0
total_time = 0
if __name__ == '__main__':
	for  file in file_list:
		parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
		parser.add_argument('--show-process', type=bool, default=False,
							help='for debug purpose, if enabled, speed for inference is dropped.')
		args = parser.parse_args()

		logger.debug('initialization %s : %s' % ('mobilenet_thin', get_graph_path('mobilenet_thin')))
		e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
		image = cv2.imread(address+file)
		logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

		# count = 0
		
		logger.debug('+image processing+')
		logger.debug('+postprocessing+')
		start_time = time.time()
		humans = e.inference(image, upsample_size=4.0)
		img = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		
		logger.debug('+classification+')
		# Getting only the skeletal structure (with white background) of the actual image
		image = np.zeros(image.shape,dtype=np.uint8)
		image.fill(255) 
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		
		# Classification
		pose_class = label_img.classify(image)
		scene_class = label_img_scene.classify(address+file)
		end_time = time.time()
		img_count += 1
		total_time = total_time + (end_time-start_time)
		logger.debug('+Completed image: {}+'.format(img_count))
		true_label_pose.append(pose_class)
		true_label_scene.append(scene_class)
		img_name.append(file)

	outF = open("pose.txt", "w")
	for line in true_label_pose:
		outF.write(line)
		outF.write("\n")
	outF.close()

	outF = open("scene.txt", "w")
	for line in true_label_scene:
		outF.write(line)
		outF.write("\n")
	outF.close()


	outF = open("img.txt", "w")
	for line in img_name:
		outF.write(line)
		outF.write("\n")
	outF.close()
# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python3 run_image.py --image=test.png
# =============================================================================
