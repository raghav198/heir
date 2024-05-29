#ifndef HEIR_LIB_TRANSFORMS_MERGELUTS_SYNTHESIZEARITHMETICLUT_H_
#define HEIR_LIB_TRANSFORMS_MERGELUTS_SYNTHESIZEARITHMETICLUT_H_

#include "mlir/include/mlir/Pass/Pass.h" // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h" // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

#include <vector>

namespace mlir
{
namespace heir
{

struct ArithmeticLut {
    std::vector<int> coefficients;
    int lookupTable;
    int lutSize;

    ArithmeticLut(const std::vector<int>& ones, const std::vector<int>& zeros, const std::vector<int>& coefficients, int lutSize);
    mlir::IntegerAttr buildAttr(mlir::OpBuilder &builder);
};

struct ArithmeticLutSynthesizer {
    static ArithmeticLutSynthesizer& getInstance();
    mlir::FailureOr<ArithmeticLut> synthesize(mlir::IntegerAttr lookupTable, int maxLutSize = 8);
private:
    mlir::DenseMap<mlir::IntegerAttr, mlir::FailureOr<ArithmeticLut>> synthesizedCache;
    mlir::FailureOr<ArithmeticLut> doSynth(mlir::IntegerAttr lookupTable, int maxLutSize);
};

} // namespace heir
} // namespace mlir



#endif // HEIR_LIB_TRANSFORMS_MERGELUTS_SYNTHESIZEARITHMETICLUT_H_