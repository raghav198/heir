#include "lib/Transforms/ShrinkLutConstants/ShrinkLutConstants.h"

#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project

#include <iostream>

namespace mlir
{
namespace heir
{
#define GEN_PASS_DEF_SHRINKLUTCONSTANTS
#include "lib/Transforms/ShrinkLutConstants/ShrinkLutConstants.h.inc"

struct ShrinkLutConstants : public impl::ShrinkLutConstantsBase<ShrinkLutConstants>
{
    using ShrinkLutConstantsBase::ShrinkLutConstantsBase;
    void runOnOperation() override
    {
        // TODO: implement this
    }
};

} // namespace heir
} // namespace mlir