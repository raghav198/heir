#ifndef LIB_TRANSFORMS_CGGICANONICALIZETOLUTS_CGGICANONICALIZETOLUTS_TD_
#define LIB_TRANSFORMS_CGGICANONICALIZETOLUTS_CGGICANONICALIZETOLUTS_TD_

include "mlir/Pass/PassBase.td"

def CGGICanonicalizeToLuts : Pass<"cggi-canonicalize-to-luts", "func::FuncOp"> {
    let summary = "Canonicalize CGGI boolean gates into LUTs";
    let description = [{
        Scan the IR for boolean gates, and replace them with equivalent lookup tables.
        This makes it easier to do LUT merging at a later step.
    }];
}

#endif // LIB_TRANSFORMS_CGGICANONICALIZETOLUTS_CGGICANONICALIZETOLUTS_TD_