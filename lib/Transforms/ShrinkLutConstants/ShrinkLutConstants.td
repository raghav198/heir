#ifndef LIB_TRANSFORMS_SHRINKLUTCONSTANTS_SHRINKLUTCONSTANTS_TD_
#define LIB_TRANSFORMS_SHRINKLUTCONSTANTS_SHRINKLUTCONSTANTS_TD_

include "mlir/Pass/PassBase.td"

def ShrinkLutConstants : Pass<"shrink-lut-constants", "func::FuncOp"> {
    let summary = "Remove constant inputs to LUTs";
    let description = [{
        Shrink the size of some LUTs by removing constant inputs
    }];
}

#endif // LIB_TRANSFORMS_SHRINKLUTCONSTANTS_SHRINKLUTCONSTANTS_TD_