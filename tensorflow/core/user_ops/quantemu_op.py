from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export

import sys
import os
import tensorflow as tf

@tf_export("quantize_emu")
def quantize_emu(input, mbits, quantize_grad=False,  mbits_grad=0, lpdata_type=0, exponent_bits=5, block_type=0, block_size=0, round_mode=0):
  _quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'quantemu.so'))
  return _quantemu_module.quantize_emu(input, mbits, quantize_grad,  mbits_grad, lpdata_type, exponent_bits, block_type, block_size, round_mode)

@ops.RegisterGradient("QuantizeEmu")
def _quantize_emu_grad(op, grad):
  if op.get_attr("quantize_grad") == True:
    mbits = op.get_attr("mbits_grad")
    _quantemu_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'quantemu.so'))
    return [_quantemu_module.quantize_emu(grad, mbits, quantize_grad=op.get_attr("quantize_grad"), mbits_grad=op.get_attr("mbits_grad"), lpdata_type=op.get_attr("lpdata_type"), exponent_bits=op.get_attr("exponent_bits"), block_type=op.get_attr("block_type"), block_size=op.get_attr("block_size"), round_mode=op.get_attr("round_mode"))] 
  else :
    return [grad]
