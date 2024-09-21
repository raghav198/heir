# Merging LUTs

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"]
)

cc_library(
    name = "ShrinkLutConstants",
    srcs = ["ShrinkLutConstants.cpp"],
    hdrs = ["ShrinkLutConstants.h"],
    deps = [
        ":pass_inc_gen",
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

# MergeLUTs tablegen and headers
gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=ShrinkLutConstants",
            ],
            "ShrinkLutConstants.h.inc"
        ),
        (
            ["-gen-pass-doc"],
            "ShrinkLutConstants.md"
        )
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ShrinkLutConstants.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles"
    ]
)