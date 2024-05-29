# Merging LUTs

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"]
)

cc_library(
    name = "MergeLUTs",
    srcs = ["MergeLUTs.cpp", "LutMergingUtils.cpp"],
    hdrs = ["MergeLUTs.h", "LutMergingUtils.h"],
    deps = [
        ":pass_inc_gen",
        ":SynthesizeArithmeticLut",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@heir//lib/Graph",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms"
    ]
)

cc_library(
    name = "SynthesizeArithmeticLut",
    srcs = ["SynthesizeArithmeticLut.cpp"],
    hdrs = ["SynthesizeArithmeticLut.h"],
    deps = [
        "@llvm-project//mlir:Pass",
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/sat:model",
        "@com_google_ortools//ortools/constraint_solver:cp"
    ]
)

# MergeLUTs tablegen and headers
gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MergeLUTs",
            ],
            "MergeLUTs.h.inc"
        ),
        (
            ["-gen-pass-doc"],
            "MergeLUTsPasses.md"
        )
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MergeLUTs.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles"
    ]
)