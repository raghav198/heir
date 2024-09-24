#ifndef LIB_TRANSFORMS_UNROLLSECRETLOOPS_UNROLLSECRETLOOPS_TD_
#define LIB_TRANSFORMS_UNROLLSECRETLOOPS_UNROLLSECRETLOOPS_TD_

include "mlir/Pass/PassBase.td"

def UnrollSecretLoops : Pass<"unroll-secret-loops"> {
  // FIXME: add add summary/description
  let summary = "";
  let description = [{
  }];
  let dependentDialects = [

  ];
}

#endif  // LIB_TRANSFORMS_UNROLLSECRETLOOPS_UNROLLSECRETLOOPS_TD_