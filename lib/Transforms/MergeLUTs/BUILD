# Merging LUTs

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"]
)

cc_library(
    name = "MergeLUTs",
    srcs = ["MergeLUTs.cpp"],
    hdrs = ["MergeLUTs.h"],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms"
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