#ifndef LIB_TRANSFORMS_MERGELUTS_MERGELUTS_TD_
#define LIB_TRANSFORMS_MERGELUTS_MERGELUTS_TD_

include "mlir/Pass/PassBase.td"

def MergeLUTs : Pass<"merge-luts", "func::FuncOp"> {
    let summary = "Replace mergeable gates into complex LUTs";
    let description = [{
        Scan the IR for sequences of boolean gates that can be expressed
        as 8-row lookup tables over a linear combination of the inputs.
    }];

    let options = [
        Option<"solverTimeout", "solver-timeout", "int", "1500",
               "Set the timeout for the ILP solver in milliseconds">,
    ];

    let statistics = [
        Statistic<"solverMaxFailures", "solver-max-failures", "Maximum number of failures where a solution was found while running the solver">,
        Statistic<"solverMaxBranches", "solver-max-branches", "Maximum number of branches where a solution was found  while running the solver">,
        Statistic<"solverMaxTime", "solver-max-time", "Maximum time taken in microseconds to find a solution while running the solver">,
        Statistic<"solverSolutions", "solver-solutions", "Total number of solutions found while running the solver">,
    ];
}

#endif // LIB_TRANSFORMS_MERGELUTS_MERGELUTS_TD_