from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export

import sys
import os
import tensorflow as tf

@tf_export("quantize_emu")
def quantize_emu(input, data_format='channels_last', allocate_copy=0, output_data_type=0, output_unsigned=0, output_precision=23, output_exponent_bits=5, channel_blocking_type=0, input_channels_per_block=4096, round_mode=0, quantize_gradients=0, quantize_gradients_only=0):
  #_quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'quantemu.so'))
  _quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_root_dir_with_all_resources(), '../../core/user_ops/quantemu.so'))
  if quantize_gradients_only is 0:
    return _quantemu_module.quantize_emu(
		input, 
		data_format, 
		allocate_copy, 
		output_data_type, 
		output_unsigned, 
		output_precision, 
		output_exponent_bits, 
		channel_blocking_type, 
		input_channels_per_block, 
		round_mode, 
		quantize_gradients, 
		quantize_gradients_only)
  else:
    return input

@ops.RegisterGradient("QuantizeEmu")
def _quantize_emu_grad(op, grad):
  if op.get_attr("quantize_gradients") is 1 : 
    _quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_root_dir_with_all_resources(), '../../core/user_ops/quantemu.so'))
    return [_quantemu_module.quantize_emu(
		grad, 
		data_format=op.get_attr("data_format"), 
		allocate_copy=int(os.getenv('QUANTEMU_ALLOCATE_COPY_GRAD', 0)), 
		output_data_type=op.get_attr("output_data_type"), 
		output_unsigned=op.get_attr("output_unsigned"), 
		output_precision=int(os.getenv('QUANTEMU_PRECISION_GRAD', 23)), 
		output_exponent_bits=op.get_attr("output_exponent_bits"), 
		channel_blocking_type=int(os.getenv('QUANTEMU_CBLOCK_TYPE_GRAD', 0)), 
		input_channels_per_block=int(os.getenv('QUANTEMU_CBLOCK_SIZE_GRAD', 4096)), 
		round_mode=int(os.getenv('QUANTEMU_RMODE_GRAD', 0)), 
		quantize_gradients=op.get_attr("quantize_gradients"), 
		quantize_gradients_only=op.get_attr("quantize_gradients_only"))]
  else :
    return [grad]
