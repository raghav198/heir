#ifndef HEIR_LIB_TRANSFORMS_MERGELUTS_LUTMERGINGUTILS_H_
#define HEIR_LIB_TRANSFORMS_MERGELUTS_LUTMERGINGUTILS_H_

#include "mlir/include/mlir/Pass/Pass.h"
#include "lib/Dialect/Comb/IR/CombOps.h"
#include "lib/Graph/Graph.h"
#include "lib/Transforms/MergeLUTs/SynthesizeArithmeticLut.h"

namespace mlir
{
namespace heir
{
struct LutMergeResult {
    llvm::SmallVector<mlir::Value> userInputs;
    mlir::IntegerAttr lookupTable;
    ArithmeticLut arithmeticLookupTable;

    template <std::size_t Index>
    std::tuple_element<Index, LutMergeResult> &get()
    {
        if constexpr (Index == 0) return userInputs;
        if constexpr (Index == 1) return lookupTable;
        if constexpr (Index == 2) return arithmeticLookupTable;
    }
};

graph::Graph<mlir::Operation*> makeLUTGraph(mlir::Operation *root);
int composeLookupTables(
    const llvm::SmallVector<int>& sourceIdxs, int sourceLut, 
    const llvm::SmallVector<int>& destIdxs, int destLut);

mlir::APInt getMergedLookupTable(comb::TruthTableOp user, comb::TruthTableOp lutToMerge, mlir::SetVector<Value> userInputs);

mlir::FailureOr<LutMergeResult> mergeLutsIfPossible(comb::TruthTableOp user, comb::TruthTableOp lutToMerge, mlir::OpBuilder &builder);

} // namespace heir
} // namespace mlir



#endif // HEIR_LIB_TRANSFORMS_MERGELUTS_LUTMERGINGUTILS_H_