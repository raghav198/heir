#include "lib/Transforms/MergeLUTs/MergeLUTs.h"

#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir
{
namespace heir
{

#define GEN_PASS_DEF_MERGELUTS
#include "lib/Transforms/MergeLUTs/MergeLUTs.h.inc"

struct MergeLUTs : public impl::MergeLUTsBase<MergeLUTs> {
    using MergeLUTsBase::MergeLUTsBase;

    MergeLUTs() = default;

    void runOnOperation() override
    {
        
    }
};

} // namespace heir
} // namespace mlir