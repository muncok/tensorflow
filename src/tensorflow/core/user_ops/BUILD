load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "mimiquant.so",
    srcs = ["mimiquant.cc"],
    gpu_srcs = ["mimiquant.cu.cc"],
    deps = [":mimiquantdeps"],
    linkopts = ["-fopenmp"],
)

cc_library(
    name = "mimiquantdeps",
    srcs = ["mimiquant.h", "posit_types.h"],
    visibility = ["//visibility:public"],
)

tf_custom_op_library(
    name = "quantemu.so",
    srcs = ["quantemu.cc"],
    gpu_srcs = ["quantemu.cu.cc"],
    deps = [":otherfiles"],
    linkopts = ["-fopenmp"],
)

cc_library(
    name = "otherfiles",
    srcs = ["quantemu.h", "posit_types.h", "posit_impl.h", "posit_impl_cuda.h"],
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
