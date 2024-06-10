#ifndef LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def CGGICanonicalizeToLuts : Pass<"cggi-canonicalize-luts"> {
  // FIXME: add add summary/description
  let summary = "";
  let description = [{
  }];
  let dependentDialects = ["mlir::heir::cggi::CGGIDialect"];
}

#endif  // LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_