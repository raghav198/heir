#ifndef HEIR_LIB_TRANSFORMS_MERGELUTS_MERGELUTS_H_
#define HEIR_LIB_TRANSFORMS_MERGELUTS_MERGELUTS_H_

#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir
{
namespace heir
{

#define GEN_PASS_DECL
#include "lib/Transforms/MergeLUTs/MergeLUTs.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/MergeLUTs/MergeLUTs.h.inc"

} // namespace heir
} // namespace mlir

#endif // HEIR_LIB_TRANSFORMS_MERGELUTS_MERGELUTS_H_