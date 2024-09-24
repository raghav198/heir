#ifndef LIB_TRANSFORMS_UNROLLSECRETLOOPS_UNROLLSECRETLOOPS_H_
#define LIB_TRANSFORMS_UNROLLSECRETLOOPS_UNROLLSECRETLOOPS_H_

#include "mlir/include/mlir/Pass/Pass.h" // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/UnrollSecretLoops/UnrollSecretLoops.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/UnrollSecretLoops/UnrollSecretLoops.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_UNROLLSECRETLOOPS_UNROLLSECRETLOOPS_H_