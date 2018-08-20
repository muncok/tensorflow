load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "quantemu.so",
    srcs = ["quantemu.cc", "quantemu.h",],
    gpu_srcs = ["quantemu.cu.cc"],
)

py_library(
    name = "quantemu_op_py",
    srcs = ["quantemu_op.py"],
    data = [":quantemu.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"], 
    visibility = ["//visibility:public"],
)

py_test(
    name = "quantemu_op_test",
    size = "small",
    srcs = ["quantemu_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quantemu_op",
        "//tensorflow:tensorflow_py",
    ],
)
