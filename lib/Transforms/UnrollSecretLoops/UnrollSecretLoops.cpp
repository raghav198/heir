#include "lib/Transforms/UnrollSecretLoops/UnrollSecretLoops.h"

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/include/mlir/Transforms/Passes.h" // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h" // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_UNROLLSECRETLOOPS
#include "lib/Transforms/UnrollSecretLoops/UnrollSecretLoops.h.inc"

struct UnrollSecretLoops : impl::UnrollSecretLoopsBase<UnrollSecretLoops> {
  using UnrollSecretLoopsBase::UnrollSecretLoopsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    if (getOperation()->walk([](secret::GenericOp generic){
      if (generic.walk([](affine::AffineForOp forOp) {
        if (failed(affine::loopUnrollFull(forOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }).wasInterrupted()) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }).wasInterrupted()) {
      signalPassFailure();
    }

    
    //(void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir