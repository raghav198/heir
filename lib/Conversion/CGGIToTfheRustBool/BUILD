load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CGGIToTfheRustBool",
    srcs = ["CGGIToTfheRustBool.cpp"],
    hdrs = [
        "CGGIToTfheRustBool.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Conversion:Utils",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/TfheRustBool/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=CGGIToTfheRustBool",
            ],
            "CGGIToTfheRustBool.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "CGGIToTfheRustBool.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CGGIToTfheRustBool.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
