#ifndef HEIR_LIB_TRANSFORMS_MERGELUTS_LUTMERGINGUTILS_H_
#define HEIR_LIB_TRANSFORMS_MERGELUTS_LUTMERGINGUTILS_H_

#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir
{
namespace heir
{
int composeLookupTables(
    const llvm::SmallVector<int>& sourceIdxs, int sourceLut, 
    const llvm::SmallVector<int>& destIdxs, int destLut);
} // namespace heir
} // namespace mlir



#endif // HEIR_LIB_TRANSFORMS_MERGELUTS_LUTMERGINGUTILS_H_