# Human Action Classification

Pose estimation & detection has been minimally implemented using the OpenPose implementation https://github.com/ildoonet/tf-pose-estimation with Tensorflow. For the binary classification of poses, namely the classes : sitting or standing, the model used, MobileNet (a CNN originally trained on the ImageNet Large Visual Recognition Challenge dataset), was retrained (final layer) on a dataset of ~1500 images of poses.

The model is able to estimate the human poses as well as classify the current pose to a fairly good degree of accuracy.

### Demo

**An alternative for improving the model along with deep learning is to include heuristics, in the form of calculation of the skeletal points’ relative distances from each other.**

 **FPS & estimation/detection varies with the CPU/GPU power.**
 
### Testing Ouputs for a Single Image

![alt text](show.png)

![alt text](show1.png)
### Dependencies

The following are required :

- python3
- tensorflow 1.9.0 (works well even with CPU version)
- opencv3
- slim
- slidingwindow
  - https://github.com/adamrehn/slidingwindow

### Compiling Locally

```bash
$ git clone https://github.com/dronefreak/human-action-classification.git
$ cd phuman-action-classification
$ sudo -H pip3 install -r requirements.txt
```
Please check the dependency tree before executing the `pip` install command.

### Pose Estimation and Action Classification on a Single Image

```
$ python3 run_image.py --image=1.jpg
```

Also, please do not forget to change the `address` variable in the code according to your local machine.

## References

### Lifting from the deep

[1] Arxiv Paper : https://arxiv.org/abs/1701.00295

[2] https://github.com/DenisTome/Lifting-from-the-Deep-release

### Mobilenet

[1] Original Paper : https://arxiv.org/abs/1704.04861

[2] Pretrained model (Pose estimation) : https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md


### Libraries

[1] Tensorpack : https://github.com/ppwwyyxx/tensorpack

### Tensorflow Tips

[1] Freeze graph : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

[2] Optimize graph : https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2



