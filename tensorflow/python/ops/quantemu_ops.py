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
def quantize_emu(input, data_format='channels_last', allocate_copy=0, output_data_type=0, pruning_algo=0, output_unsigned=0, output_precision=23, output_exponent_bits=5, channel_blocking_type=0, input_channels_per_block=4096, round_mode=0, quantize_gradients=0, gradient_precision=23, quantize_gradients_only=0):
  _quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_root_dir_with_all_resources(), '../../core/user_ops/quantemu.so'))
  return _quantemu_module.quantize_emu(
		input, 
		data_format, 
		allocate_copy, 
		output_data_type, 
              	pruning_algo,
                output_unsigned, 
		output_precision, 
		output_exponent_bits, 
		channel_blocking_type, 
		input_channels_per_block, 
		round_mode, 
		quantize_gradients, 
		gradient_precision, 
		quantize_gradients_only)

@ops.RegisterGradient("QuantizeEmu")
def _quantize_emu_grad(op, grad):
  if op.get_attr("quantize_gradients") is 1 : 
    _quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_root_dir_with_all_resources(), '../../core/user_ops/quantemu.so'))
    return [_quantemu_module.quantize_emu(
		grad, 
		data_format=op.get_attr("data_format"), 
		allocate_copy=int(os.getenv('QUANTEMU_ALLOCATE_COPY_GRAD', 0)), 
		output_data_type=op.get_attr("output_data_type"), 
		pruning_algo=op.get_attr("pruning_algo"), 
		output_unsigned=op.get_attr("output_unsigned"), 
		output_precision=op.get_attr("gradient_precision"),  # Output precision is gradient precision 
		output_exponent_bits=op.get_attr("output_exponent_bits"), 
		channel_blocking_type=int(os.getenv('QUANTEMU_CBLOCK_TYPE_GRAD', 0)), 
		input_channels_per_block=int(os.getenv('QUANTEMU_CBLOCK_SIZE_GRAD', 4096)), 
		round_mode=int(os.getenv('QUANTEMU_RMODE_GRAD', 0)), 
		quantize_gradients=op.get_attr("quantize_gradients"), 
		gradient_precision=op.get_attr("gradient_precision"), 
		quantize_gradients_only=op.get_attr("quantize_gradients_only"))]
  else :
    return [grad]
