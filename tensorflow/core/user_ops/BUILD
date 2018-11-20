load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "quantemu.so",
    srcs = ["quantemu.cc"],
    gpu_srcs = ["quantemu.cu.cc"],
#    srcs = ["quantemu.cc", "posit_impl.cc"],
#    gpu_srcs = ["quantemu.cu.cc", "posit_impl.cu.cc"],
    deps = [":otherfiles"],
    linkopts = ["-fopenmp"],
)

cc_library(
    name = "otherfiles",
    srcs = ["quantemu.h", "posit_types.h"],
    visibility = ["//visibility:public"],
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
    srcs = ["quantemu_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quantemu_op",
        "//tensorflow:tensorflow_py",
    ],
)
