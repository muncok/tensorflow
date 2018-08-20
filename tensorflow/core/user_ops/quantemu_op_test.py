import tensorflow as tf
from tensorflow.core.user_ops import quantemu_op #quantize_emu

class QuantEmuOpTest(tf.test.TestCase):
  def testQuantEmuOp(self):
     with self.test_session(use_gpu=True, force_gpu=False) as sess:
       output = quantemu_op.quantize_emu([4,5,6,7,8,9] , 8, quantize_grad=False, mbits_grad=0, lpdata_type=0, exponent_bits=5, block_type=0, block_size=0, round_mode=0 )
       output = tf.Print(output, [output], message="output and output2 tensor: ")
       output2 = output * [10] 
       end = tf.Print(output2, [output2], message="output2 tensor: ")
       end.eval()

if __name__ == "__main__":
  tf.test.main()
