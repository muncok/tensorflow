TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o quantemu.cu.o quantemu.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I/usr/local --expt-relaxed-constexpr

g++ -std=c++11 -mf16c -mavx -mavx2  -shared quantemu.cu.o quantemu.cc -o libquantemu.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2  
