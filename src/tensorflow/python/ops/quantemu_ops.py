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
def quantize_emu(
         input, 
         data_format='channels_last', 
         allocate_copy=0, 
         data_type=0, 
         precision=23, 
         exponent_bits=5, 
         channel_blocking_type=0, 
         channels_per_block=4096, 
         round_mode=0): 

  _quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_root_dir_with_all_resources(), '../../core/user_ops/quantemu.so'))
  return _quantemu_module.quantize_emu(
		input, 
		data_format, 
		allocate_copy, 
		data_type, 
		precision, 
		exponent_bits, 
		channel_blocking_type, 
		channels_per_block, 
		round_mode)  

@ops.RegisterGradient("QuantizeEmu")
def _quantize_emu_grad(op, grad):

  # clipping gradient to  -1, 1
  #tf.clip_by_value(grad, -1, 1)
  """
  if op.get_attr("quantize_gradients") is 1 : 
    _quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_root_dir_with_all_resources(), '../../core/user_ops/quantemu.so'))
    return [_quantemu_module.quantize_emu(
		grad, 
		data_format=op.get_attr("data_format"), 
		allocate_copy=int(os.getenv('QUANTEMU_ALLOCATE_COPY_GRAD', 0)), 
		data_type=op.get_attr("data_type"), 
		precision=op.get_attr("precision"),  # Output precision is gradient precision 
		exponent_bits=op.get_attr("output_exponent_bits"), 
		channel_blocking_type=int(os.getenv('QUANTEMU_CBLOCK_TYPE_GRADS', 0)), 
		channels_per_block=int(os.getenv('QUANTEMU_CBLOCK_SIZE_GRADS', 4096)), 
		round_mode=int(os.getenv('QUANTEMU_RMODE_GRADS', 0)))]
  else :
  """
  return [grad]
