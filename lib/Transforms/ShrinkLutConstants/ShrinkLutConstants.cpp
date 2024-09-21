#include "lib/Transforms/ShrinkLutConstants/ShrinkLutConstants.h"

#include <iostream>
#include <set>

#include "lib/Dialect/Comb/IR/CombOps.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project

namespace mlir {
namespace heir {
#define GEN_PASS_DEF_SHRINKLUTCONSTANTS
#include "lib/Transforms/ShrinkLutConstants/ShrinkLutConstants.h.inc"

struct ShrinkLutConstants
    : public impl::ShrinkLutConstantsBase<ShrinkLutConstants> {
  using ShrinkLutConstantsBase::ShrinkLutConstantsBase;
  void runOnOperation() override {
    auto root = getOperation();
    mlir::OpBuilder builder(&getContext());

    mlir::DenseMap<comb::TruthTableOp, std::set<int>> lutConstantIndices;

    root->walk([&lutConstantIndices](comb::TruthTableOp lut) -> void {
      std::set<int> constantIndices;
      for (auto [i, input] : llvm::enumerate(lut.getLookupTableInputs())) {
        if (mlir::isa<arith::ConstantOp>(input.getDefiningOp()))
          constantIndices.insert(i);
      }
      if (!constantIndices.empty())
        lutConstantIndices.insert({lut, constantIndices});
    });

    for (auto &[lut, indices] : lutConstantIndices) {
      auto originalLutInputs = lut.getLookupTableInputs();
      int reducedSize = originalLutInputs.size() - indices.size();
      std::vector<int> convertedInputBitsFixed(originalLutInputs.size());
      for (auto index : indices) {
        auto constInput = originalLutInputs[index];
        convertedInputBitsFixed[index] =
            mlir::cast<mlir::IntegerAttr>(
                constInput.getDefiningOp<arith::ConstantOp>().getValue())
                .getInt();
      }
      std::vector<int> nonConstantIndices;
      nonConstantIndices.reserve(reducedSize);
      for (int i = 0; i < originalLutInputs.size(); i++) {
        if (indices.find(i) == indices.end()) nonConstantIndices.push_back(i);
      }

      std::vector<mlir::Value> nonConstantInputs;
      nonConstantInputs.reserve(nonConstantIndices.size());
      for (auto i : nonConstantIndices)
        nonConstantInputs.push_back(originalLutInputs[i]);

      int originalLookupTable = lut.getLookupTable().getValue().getZExtValue();
      int reducedLookupTable = 0;

      for (int input = 0; input < (1 << reducedSize); input++) {
        std::vector<int> convertedInputBits = convertedInputBitsFixed;
        for (int i = 0; i < reducedSize; i++)
          convertedInputBits[nonConstantIndices[reducedSize - i - 1]] =
              (input & (1 << i)) >> i;

        int convertedInput = 0;
        for (int i = 0; i < convertedInputBits.size(); i++)
          convertedInput |=
              convertedInputBits[convertedInputBits.size() - i - 1] << i;

        reducedLookupTable |= ((originalLookupTable >> convertedInput) & 1)
                              << input;
      }

      builder.setInsertionPointAfter(lut);
      auto reducedLutOp = builder.create<comb::TruthTableOp>(
          lut->getLoc(), nonConstantInputs,
          builder.getIntegerAttr(
              builder.getIntegerType(1 << reducedSize, false),
              reducedLookupTable));

      lut->replaceAllUsesWith(reducedLutOp);
      lut->erase();
    }
  }
};

}  // namespace heir
}  // namespace mlir