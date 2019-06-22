import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

true_label_pose = []
true_label_scene = []

pred_label_pose = []
pred_label_scene = []

##True Labels
outF = open("pose-gT.txt", "r")
for line in outF:
	true_label_pose.append(line)
outF.close()

outF = open("scene-gT.txt", "r")
for line in outF:
	true_label_scene.append(line)

outF.close()

##Predicted Labels
outF = open("pose.txt", "r")
for line in outF:
	pred_label_pose.append(line)
outF.close()

outF = open("scene.txt", "r")
for line in outF:
	pred_label_scene.append(line)

outF.close()


print(len(true_label_pose))
print(len(pred_label_pose))

acc_pose = accuracy_score(true_label_pose,pred_label_pose)
#avp_pose = average_precision_score(true_label_pose,pred_label_pose)
#bri_pose = brier_score_loss(true_label_pose,pred_label_pose)
#f1_pose = f1_score(true_label_pose,pred_label_pose, average='samples')
#lg_pose = log_loss(true_label_pose,pred_label_pose)
# pr_pose = precision_score(true_label_pose,pred_label_pose, average='samples')
# re_pose = recall_score(true_label_pose,pred_label_pose)
# roc_pose = roc_auc_score(true_label_pose,pred_label_pose)

acc_scene = accuracy_score(true_label_scene,pred_label_scene)
#avp_scene = average_precision_score(true_label_scene,pred_label_scene)
#b#ri_scene = brier_score_loss(true_label_scene,pred_label_scene)
#f1_scene = f1_score(true_label_scene,pred_label_scene, average='samples')
#lg_scene = log_loss(true_label_scene,pred_label_scene)
# pr_scene = precision_score(true_label_scene,pred_label_scene, average='samples')
# re_scene = recall_score(true_label_scene,pred_label_scene)
# roc_scene = roc_auc_score(true_label_scene,pred_label_scene)

print(acc_pose)
print(acc_scene)