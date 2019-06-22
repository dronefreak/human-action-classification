# pose-estimation-detection

Pose estimation & detection has been minimally implemented using the OpenPose implementation https://github.com/ildoonet/tf-pose-estimation with Tensorflow. For the binary classification of poses, namely the classes : sitting or standing, the model used, MobileNet (a CNN originally trained on the ImageNet Large Visual Recognition Challenge dataset), was retrained (final layer) on a dataset of ~1500 images of poses.

The model is able to estimate the human poses as well as classify the current pose to a fairly good degree of accuracy.

### Demo

**An alternative for improving the model along with deep learning is to include heuristics, in the form of calculation of the skeletal points’ relative distances from each other.**

 **FPS & estimation/detection varies with the CPU/GPU power.**
 
### Training Examples

- For sitting pose
![alt text](/images/sitting.jpeg)

- For standing pose
![alt text](/images/standing.jpeg)

### Dependencies

The following are required :

- python3
- tensorflow 1.9.0 (works well even with CPU version)
- opencv3
- slim
- slidingwindow
  - https://github.com/adamrehn/slidingwindow

### Cloning & installing dependencies

```bash
$ git clone https://github.com/SyBorg91/pose-estimation-detection
$ cd pose-estimation-detection
$ pip3 install -r requirements.txt
```

### Pose Estimation with realtime webcam feed

```
$ python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
```

Run the above command to start pose estimation with the onboard webcam.

## References

### OpenPose

[1] https://github.com/CMU-Perceptual-Computing-Lab/openpose

[2] Training Codes : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

[3] Custom Caffe by Openpose : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train

[4] Keras Openpose : https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation

[5] Keras Openpose2 : https://github.com/kevinlin311tw/keras-openpose-reproduce

### Lifting from the deep

[1] Arxiv Paper : https://arxiv.org/abs/1701.00295

[2] https://github.com/DenisTome/Lifting-from-the-Deep-release

### Mobilenet

[1] Original Paper : https://arxiv.org/abs/1704.04861

[2] Pretrained model (Pose estimation) : https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

[3] Retrained model (Pose detection) : https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/

### Libraries

[1] Tensorpack : https://github.com/ppwwyyxx/tensorpack

### Tensorflow Tips

[1] Freeze graph : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

[2] Optimize graph : https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2



