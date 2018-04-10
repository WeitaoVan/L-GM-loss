# https://github.com/WeitaoVan/L-GM-loss
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
"""Contains definitions for the preactivation form of Residual Networks.
Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.contrib.slim as slim

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   data_format):
  """Standard building block for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

  return tf.identity(inputs, name)

def inference_lgm(image, resnet_size, is_training, num_classes=100, labels=None, reuse=False, output_feat=False):
  print('built lgm inference')
  _, feat = inference(image, resnet_size, is_training, num_classes=num_classes, reuse=reuse, output_feat=True)
  with tf.variable_scope('rbf_loss', reuse=reuse):
      logits, likelihood_reg_loss, means = lgm_logits(feat, num_classes, labels=labels, alpha=0.1, lambda_=0.01)
  if output_feat:
    return logits, likelihood_reg_loss, means, feat
  else:
    return logits, likelihood_reg_loss, means


def inference(x, resnet_size, is_training, num_classes=100, reuse=False, output_feat=False, data_format='NHWC', input_noise=0., drop_ratio=None):
  if data_format == 'NHWC':
    data_format = 'channels_last'
  elif data_format == 'NCHW':  # Very slow for tensorflow 1.0. Claimed to be fast for 1.4
    data_format = 'channels_first'
  else:
    raise ValueError('invalid data_format')
  if input_noise > 0:
    # random flip
    x = flip_randomly(x, True, False)
    # gaussian noise
    noise = tf.random_normal(tf.shape(x))
    x = x + noise * input_noise
    # random crop
    N = x.get_shape().as_list()[0]
    x = tf.random_crop(x, (N, 32, 32, 3))
    print('flip, noise, random crop in graph.')
  with tf.variable_scope('', reuse=reuse):
    network = cifar10_resnet_v2_generator(resnet_size, num_classes, 
                                         data_format=data_format)
    return network(x, is_training, output_feat=output_feat, drop_ratio=drop_ratio)

def flip_randomly(inputs, horizontally, vertically, name=None):
  """Flip images randomly. Make separate flipping decision for each image.

  Args:
      inputs (4-D tensor): Input images (batch size, height, width, channels).
      horizontally (bool): If True, flip horizontally with 50% probability. Otherwise, don't.
      vertically (bool): If True, flip vertically with 50% probability. Otherwise, don't.
      scope: A name for the operation.
  """
  with tf.name_scope(name, "flip_randomly") as scope:
    batch_size, height, width, _ = tf.unstack(tf.shape(inputs))
    vertical_choices = (tf.random_uniform([batch_size], 0, 2, tf.int32) *
                            tf.to_int32(vertically))
    horizontal_choices = (tf.random_uniform([batch_size], 0, 2, tf.int32) *
                              tf.to_int32(horizontally))
    vertically_flipped = tf.reverse_sequence(inputs, vertical_choices * height, 1)
    both_flipped = tf.reverse_sequence(vertically_flipped, horizontal_choices * width, 2)
    return tf.identity(both_flipped, name=scope)

def imagenet_inference(x, resnet_size, is_training, num_classes=1000, reuse=False, output_feat=False):
  with tf.variable_scope('', reuse=reuse):
    network = imagenet_resnet_v2(resnet_size, num_classes, data_format='channels_last')
    return network(x, is_training, output_feat=output_feat)  

def inference_center(image, resnet_size, is_training, num_classes=100, labels=None, reuse=False, loss_weight=0.05):
  print('built center loss inference')
  logits, feat = inference(image, resnet_size, is_training, num_classes=num_classes, reuse=reuse, output_feat=True)
  if labels is None:
    # psudo-labels for adversarial robust test
    labels = tf.argmax(logits, 1)
    d_to_centers = center_loss_layer(feat, labels, num_classes, validation=True, reuse=reuse)
    return logits, d_to_centers, feat
  else:
    center_loss, means, centers_op = center_loss_layer(feat, labels, num_classes, alfa=0.99)
    center_loss = loss_weight * center_loss
    return logits, center_loss, means, centers_op
  
def cifar10_resnet_v2_generator(resnet_size, num_classes, data_format=None):
  """Generator for CIFAR-10 ResNet v2 models.
  Args:
    resnet_size: A single integer for the size of the ResNet model.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  Raises:
    ValueError: If `resnet_size` is invalid.
  """
  if resnet_size % 6 != 2:
    raise ValueError('resnet_size must be 6n + 2:', resnet_size)

  num_blocks = (resnet_size - 2) // 6

  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training, K=64, output_feat=False, drop_ratio=None):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert from channels_last (NHWC) to channels_first (NCHW). This
      # provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=K, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = block_layer(
        inputs=inputs, filters=K, block_fn=building_block, blocks=num_blocks,
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=K*2, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    if drop_ratio is not None:
      inputs = slim.dropout(inputs, keep_prob=1.0-drop_ratio, is_training=is_training, scope='dropout1')    
    inputs = block_layer(
        inputs=inputs, filters=K*4, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    if drop_ratio is not None:
      inputs = slim.dropout(inputs, keep_prob=1.0-drop_ratio, is_training=is_training, scope='dropout1')    
      
    inputs = batch_norm_relu(inputs, is_training, data_format)
    H = inputs.get_shape().as_list()[1]
    # pool_size=8 for 32x32 input; 7 for 28x28 input
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=H, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(inputs, [-1, K*4])
    feat = tf.identity(inputs, 'feature')
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    if output_feat:
      return inputs, feat
    else:
      return inputs

  return model


def imagenet_resnet_v2_generator(block_fn, layers, num_classes,
                                 data_format=None):
  """Generator for ImageNet ResNet v2 models.
  Args:
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  """
  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training, output_feat=False):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert from channels_last (NHWC) to channels_first (NCHW). This
      # provides a large performance boost on GPU.
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=64, kernel_size=7, strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=7, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(inputs,
                        [-1, 512 if block_fn is building_block else 2048])
    feat = tf.identity(inputs, 'feature')
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    if output_feat:
      return inputs, feat
    else:
      return inputs
  return model


def imagenet_resnet_v2(resnet_size, num_classes, data_format=None):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': building_block, 'layers': [2, 2, 2, 2]},
      34: {'block': building_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_size not in model_params:
    raise ValueError('Not a valid resnet_size:', resnet_size)

  params = model_params[resnet_size]
  return imagenet_resnet_v2_generator(
      params['block'], params['layers'], num_classes, data_format)

def lgm_logits(feat, num_classes, labels=None, alpha=0.1, lambda_=0.01):
  '''
  The 3 input hyper-params are explained in the paper.\n
  Support 2 modes: Train, Validation\n
  (1)Train:\n
  return logits, likelihood_reg_loss\n
  (2)Validation:\n
  Set labels=None\n
  return logits\n
  '''
  N = feat.get_shape().as_list()[0]
  feat_len = feat.get_shape()[1]
  means = tf.get_variable('rbf_centers', [num_classes, feat_len], dtype=tf.float32, 
                                initializer=tf.contrib.layers.xavier_initializer())

  XY = tf.matmul(feat, means, transpose_b=True)
  XX = tf.reduce_sum(tf.square(feat), axis=1, keep_dims=True)
  YY = tf.reduce_sum(tf.square(tf.transpose(means)), axis=0, keep_dims=True)
  neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

  if labels is None:
    # Validation mode
    psudo_labels = tf.argmax(neg_sqr_dist, axis=1)
    means_batch = tf.gather(means, psudo_labels)
    likelihood_reg_loss = lambda_ * tf.nn.l2_loss(feat - means_batch, name='likelihood_regularization') * (1. / N)
    # In fact, in validation mode, we only need to output neg_sqr_dist. 
    # The likelihood_reg_loss and means are only for research purposes.
    return neg_sqr_dist, likelihood_reg_loss, means
  # *(1 + alpha)
  ALPHA = tf.one_hot(labels, num_classes, on_value=alpha, dtype=tf.float32)
  K = ALPHA + tf.ones([N, num_classes], dtype=tf.float32) 
  logits_with_margin = tf.multiply(neg_sqr_dist, K)
  # likelihood regularization
  means_batch = tf.gather(means, labels)
  likelihood_reg_loss = lambda_ * tf.nn.l2_loss(feat - means_batch, name='center_regularization') * (1. / N) 
  print('LGM loss built with alpha=%f, lambda=%f\n' %(alpha, lambda_))
  return logits_with_margin, likelihood_reg_loss, means

def center_loss_layer(features, label, nrof_classes, alfa=0.95, validation=False, reuse=False):
  """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
     (http://ydwen.github.io/papers/WenECCV16.pdf)
  """
  nrof_features = features.get_shape()[1]
  with tf.variable_scope("center_loss", reuse=reuse):
    means = tf.get_variable('means', [nrof_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
  label = tf.reshape(label, [-1])
  means_batch = tf.gather(means, label)
  diff = (1 - alfa) * (means_batch - features)
  centers_op = tf.assign(means, tf.scatter_sub(means, label, diff))
  loss = tf.nn.l2_loss(features - means_batch)
  if not validation:
    return loss, means, centers_op
  else:
    return tf.reduce_sum(tf.square(features - means_batch), 1)


def step_rampup(global_step, rampup_length):
  result = tf.cond(global_step < rampup_length,
                     lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
  return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
  global_step = tf.to_float(global_step)
  rampup_length = tf.to_float(rampup_length)
  def ramp():
    phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
    return tf.exp(-5.0 * phase * phase)

  result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
  return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
  global_step = tf.to_float(global_step)
  rampdown_length = tf.to_float(rampdown_length)
  training_length = tf.to_float(training_length)
  def ramp():
    phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
    return tf.exp(-12.5 * phase * phase)

  result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
  return tf.identity(result, name="sigmoid_rampdown")
  
