# Merging LUTs

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"]
)

cc_library(
    name = "CGGICanonicalizeToLuts",
    srcs = ["CGGICanonicalizeToLuts.cpp"],
    hdrs = ["CGGICanonicalizeToLuts.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Graph",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ]
)

# MergeLUTs tablegen and headers
gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=CGGICanonicalizeToLuts",
            ],
            "CGGICanonicalizeToLuts.h.inc"
        ),
        (
            ["-gen-pass-doc"],
            "CGGICanonicalizeToLutsPasses.md"
        )
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CGGICanonicalizeToLuts.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles"
    ]
)