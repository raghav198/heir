#ifndef LIB_TRANSFORMS_MERGELUTS_MERGELUTS_TD_
#define LIB_TRANSFORMS_MERGELUTS_MERGELUTS_TD_

include "mlir/Pass/PassBase.td"

def MergeLUTs : Pass<"merge-luts", "func::FuncOp"> {
    let summary = "Replace mergeable gates into complex LUTs";
    let description = [{
        Scan the IR for sequences of boolean gates that can be expressed
        as 8-row lookup tables over a linear combination of the inputs.
    }];
}

#endif // LIB_TRANSFORMS_MERGELUTS_MERGELUTS_TD_