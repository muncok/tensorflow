from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
import os
import enum 

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export

class lp_data (enum.Enum): 
    DFP = 0
    LPFP = 4
    BFP = 5

class channel_block(enum.Enum): 
    NONE = 0
    BLOCK_C = 1
    BLOCK_CHW = 2

@tf_export("quantize_to_dfp")
def quantize_to_dfp (input, output_precision, channel_blocking_type, input_channels_per_block, round_mode):
  
  if channel_block(channel_blocking_type) == channel_block.BLOCK_C : 
    absmax = tf.reshape(tf.reduce_max (tf.abs(input)), [-1])
    quant_max = tf.pow(2, output_precision - 1) - 1
    sfquant = tf.cast(quant_max, dtype=tf.float32)/absmax;
    sfdequant = 1.0/sfquant
    quantized  = tf.round(input * sfquant)
    return  tf.cast(quantized, dtype=tf.float32) * sfdequant

  elif channel_block(channel_blocking_type) == channel_block.BLOCK_CHW : 
    absmax = tf.reshape(tf.reduce_max (tf.abs(input)), [-1])
    quant_max = tf.pow(2, output_precision - 1) - 1
    sfquant = tf.cast(quant_max, dtype=tf.float32)/absmax;
    sfdequant = 1.0/sfquant
    quantized  = tf.round(input * sfquant)
    return  tf.cast(quantized, dtype=tf.float32) * sfdequant

  else : 
    absmax = tf.reshape(tf.reduce_max (tf.abs(input)), [-1])
    quant_max = tf.pow(2, output_precision - 1) - 1
    sfquant = tf.cast(quant_max, dtype=tf.float32)/absmax;
    sfdequant = 1.0/sfquant
    input  = tf.round(input * sfquant)
    #quantized  = tf.round(input * sfquant)
    #quantized  = tf.cast(input * sfquant, dtype=tf.int32)
    #return  tf.cast(tf.round(input * sfquant), dtype=tf.float32) * sfdequant
    input = tf.cast(input, dtype=tf.float32) * sfdequant
    return input

@tf_export("to_dfp")
def to_dfp(
	input, 
	data_format='channels_last', 
	allocate_copy=0, 
	output_data_type=0, 
	output_precision=23, 
	output_exponent_bits=5, 
	channel_blocking_type=0, 
	input_channels_per_block=4096, 
	round_mode=0, 
	quantize_gradients=0, 
	quantize_gradients_only=0):

  if lp_data(output_data_type) == lp_data.DFP :
    return quantize_to_dfp(input, output_precision, channel_blocking_type, input_channels_per_block, round_mode) 
  else :
    return input

@ops.RegisterGradient("to_dfp")
def to_dfp_grad(op, grad):
  if op.get_attr("quantize_gradients") == 1 : 
    if op.get_attr("output_data_type") == lp_data.DFP : 
      return [quantize_to_dfp(
               grad, 
               int(os.getenv('QUANTEMU_PRECISION_GRAD', 23)), 
               int(os.getenv('QUANTEMU_CBLOCK_TYPE_GRAD', 0)), 
               int(os.getenv('QUANTEMU_CBLOCK_SIZE_GRAD', 4096)), 
               int(os.getenv('QUANTEMU_RMODE_GRAD', 0)))]
  else :
    return [grad]
