#ifndef LIB_TRANSFORMS_CGGICANONICALIZETOLUTS_H_
#define LIB_TRANSFORMS_CGGICANONICALIZETOLUTS_H_

#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir
{
namespace heir
{

#define GEN_PASS_DECL
#include "lib/Transforms/CGGICanonicalizeToLuts/CGGICanonicalizeToLuts.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/CGGICanonicalizeToLuts/CGGICanonicalizeToLuts.h.inc"

} // namespace heir
} // namespace mlir

#endif // LIB_TRANSFORMS_CGGICANONICALIZETOLUTS_H_