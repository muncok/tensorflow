from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
import os

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export

@tf_export("quantize_emu")
def quantize_emu(input, data_format='channels_last', output_data_type=0, output_precision=23, output_exponent_bits=5, channel_blocking_type=0, input_channels_per_block=4096, round_mode=0, quantize_gradients=0, gradient_precision=23):
  #_quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'quantemu.so'))
  _quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_root_dir_with_all_resources(), '../../core/user_ops/quantemu.so'))
  return _quantemu_module.quantize_emu(input, data_format, output_data_type, output_precision, output_exponent_bits, channel_blocking_type, input_channels_per_block, round_mode, quantize_gradients, gradient_precision)

@ops.RegisterGradient("QuantizeEmu")
def _quantize_emu_grad(op, grad):
  if op.get_attr("quantize_gradients") is 1:
    mbits = op.get_attr("gradient_precision")
    #_quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'quantemu.so'))
    _quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_root_dir_with_all_resources(), '../../core/user_ops/quantemu.so'))
    return [_quantemu_module.quantize_emu(grad, data_format=op.get_attr("data_format"), output_data_type=op.get_attr("output_data_type"), output_precision=mbits, output_exponent_bits=op.get_attr("output_exponent_bits"), channel_blocking_type=op.get_attr("channel_blocking_type"), input_channels_per_block=op.get_attr("input_channels_per_block"), round_mode=op.get_attr("round_mode"), quantize_gradients=op.get_attr("quantize_gradients"), gradient_precision=op.get_attr("gradient_precision"))]
  else :
    return [grad]
