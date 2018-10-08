#pip uninstall /tmp/tensorflow_pkg/tensorflow-1.9.0rc0-cp35-cp35m-linux_x86_64.whl
 
bazel build --config=cuda //tensorflow/core/user_ops:quantemu.so
#bazel build --config=cuda //tensorflow/core/user_ops:quantemu_op_py
bazel build --config=cuda //tensorflow/python:quantemu_ops
bazel build --config=cuda //tensorflow/python/tools:freeze_graph
bazel build --config=cuda //tensorflow/python:no_contrib
bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
# bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# bazel build --config=opt --config=cuda //tensorflow/python:quantemu_ops
# bazel build --config=opt --config=cuda //tensorflow/python:no_contrib
# bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
# bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

#pip install /tmp/tensorflow_pkg/tensorflow-1.9.0rc0-cp35-cp35m-linux_x86_64.whl
#--config=opt 
#--config=opt 
#--config=opt 
#--config=opt 
#--config=opt 
#--config=opt 
