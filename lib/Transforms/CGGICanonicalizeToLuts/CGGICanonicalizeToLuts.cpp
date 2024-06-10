#include "lib/Transforms/CGGICanonicalizeToLuts/CGGICanonicalizeToLuts.h"

#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project

namespace mlir
{
namespace heir
{

#define GEN_PASS_DEF_CGGICANONICALIZETOLUTS
#include "lib/Transforms/CGGICanonicalizeToLuts/CGGICanonicalizeToLuts.h.inc"

struct CGGICanonicalizeToLuts : public impl::CGGICanonicalizeToLutsBase<CGGICanonicalizeToLuts> {
    using CGGICanonicalizeToLutsBase::CGGICanonicalizeToLutsBase;

    void runOnOperation() override
    {
        MLIRContext *context = &getContext();

        RewritePatternSet patterns(context);
        ConversionTarget targeT(*context);

    }
};

} // namespace heir
} // namespace mlir