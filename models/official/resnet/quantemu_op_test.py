import tensorflow as tf
from tensorflow.python.ops import quantemu_ops

class QuantEmuOpTest(tf.test.TestCase):
  def testQuantEmuOp(self):
     # quantemu_ops = tf.load_op_library('./quantemu.so')
     # print(quantemu_ops)
     with self.test_session(use_gpu=True, force_gpu=False) as sess:
       output = quantemu_ops.quantize_emu(
           tf.constant((10,), dtype=tf.float32),
           data_type=9,
           data_format='channels_first',
           precision=8,
           exponent_bits=5,
           channel_blocking_type=0,
           channels_per_block=0,
           round_mode=0 )
       # output = tf.Print(output, [output], message="output and output2 tensor: ")
       # output2 = output * [10]
       # end = tf.Print(output2, [output2], message="output2 tensor: ")
       print(output.eval())

if __name__ == "__main__":
  tf.test.main()
