# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(image_file, input_height=299, input_width=299,
				input_mean=0, input_std=255):
    
  float_caster = tf.cast(image_file, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def classify(image_file):
    
# =============================================================================
#   Note : Provide your own absolute file path for the following
#   You can choose the retrained graph of either as v1.0 or v2.0 
#   Both models are retrained inception models (on my procured dataset)
#   v1.0 was trained for 500 epocs on a preliminary dataset of poses.
#   v2.0 was trained for 4000 epocs on a dataset containing the previous dataset
#   and more.
# =============================================================================
  # Change the path to your convenience
  file_path = os.path.abspath(os.path.dirname(__file__))
  path = os.path.join(file_path, '../models/graph/retrained/retrained_v1.0/')
  model_file = path+'retrained_graph.pb'
  label_file = path+'retrained_labels.txt'
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(image_file,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)

  labels = load_labels(label_file)

  print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
  template = "{} (score={:0.5f})"
  label = ''
  if results[0] > results[1]:
      label = labels[0]
      result = results[0]
  else:
      label = labels[1]
      result = results[1]

  return template.format(label, result)
