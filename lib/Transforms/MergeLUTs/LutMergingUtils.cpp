#include "lib/Transforms/MergeLUTs/LutMergingUtils.h"

#include <algorithm>
#include <ostream>
#include <vector>

#include "lib/Dialect/Comb/IR/CombOps.h"
#include "lib/Transforms/MergeLUTs/SynthesizeArithmeticLut.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project

#define DEBUG_TYPE "merge-luts"

namespace mlir {
namespace heir {

graph::Graph<mlir::Operation*> makeLUTGraph(mlir::Operation* root) {
  graph::Graph<mlir::Operation*> lutsDependencyGraph;

  root->walk([&lutsDependencyGraph](mlir::Operation* op) {
    // if (!llvm::isa<comb::TruthTableOp>(op)) return WalkResult::advance();
    // for (mlir::Operation *user : op->getUsers()) {
    //   if (!llvm::isa<comb::TruthTableOp>(user)) return WalkResult::advance();
    // }
    // lutsDependencyGraph.addVertex(op);
    // return WalkResult::advance();
    if (llvm::isa<comb::TruthTableOp>(op)) lutsDependencyGraph.addVertex(op);
  });

  for (mlir::Operation* vertex : lutsDependencyGraph.getVertices()) {
    for (mlir::Operation* user : vertex->getUsers()) {
      lutsDependencyGraph.addEdge(vertex, user);
    }
  }

  return lutsDependencyGraph;
}

template <class T>
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const std::vector<T>& vec) {
  os << "[";
  for (auto t : vec) os << t;
  os << "]";

  return os;
}

std::vector<bool> intToBits(int value, int width, bool flip = false) {
  std::vector<bool> bits;
  bits.reserve(width);
  for (int i = 0; i < width; i++)
    bits.push_back((value & (1 << (flip ? i : (width - i - 1)))) != 0);
  return bits;
}

// int bitsToIntFlip(const std::vector<bool>& bits) {
//   int value = 0;
//   int power = 0;
//   for (const auto& bit : bits) {
//     if (bit) value += (1 << power);
//     power++;
//   }
//   return value;
// }

int bitsToInt(const std::vector<bool>& bits) {
  int value = 0;
  int power = 0;

  for (const auto& bit : llvm::reverse(bits)) {
    if (bit) value += (1 << power);
    power++;
  }
  return value;
}

template <class B>
mlir::APInt bitvecToAPInt(const B& bitvec) {
  std::string bitstring;
  int size = 0;
  for (auto b : bitvec) {
    bitstring += (b ? "1" : "0");
    size++;
  }
  return mlir::APInt(size, bitstring, 2);
}

mlir::APInt composeLookupTables(const llvm::SmallVector<int>& sourceIdxs,
                                const mlir::APInt& sourceLut,
                                const llvm::SmallVector<int>& destIdxs,
                                const mlir::APInt& destLut) {
  int composedInputCount =
      std::max(*std::max_element(sourceIdxs.begin(), sourceIdxs.end()),
               *std::max_element(destIdxs.begin(), destIdxs.end())) +
      1;

  llvm::DenseMap<int, int> sourceLocs;
  llvm::DenseMap<int, int> destLocs;

  // std::vector<bool> sourceLutTable = intToBits(sourceLut, 1 <<
  // sourceIdxs.size(), true); std::vector<bool> destLutTable =
  // intToBits(destLut, 1 <<  destIdxs.size(), true);

  for (const auto& [loc, index] : llvm::enumerate(sourceIdxs))
    sourceLocs.insert({index, loc});

  for (const auto& [loc, index] : llvm::enumerate(destIdxs))
    destLocs.insert({index, loc});

  std::vector<bool> composedLookupTable;

  for (int possibleInput = 0; possibleInput < (1 << composedInputCount);
       possibleInput++) {
    auto inputs = intToBits(possibleInput, composedInputCount);
    std::vector<bool> sourceInputs;
    for (auto idx : sourceIdxs) {
      sourceInputs.push_back(inputs[idx]);
    }
    int sourceInput = bitsToInt(sourceInputs);
    bool sourceOutput = (sourceLut.ashr(sourceInput) & 1).getBoolValue();
        // bool sourceOutput = sourceLutTable[bitsToInt(sourceInputs)];

    std::vector<bool> destInputs;
    for (auto idx : destIdxs) {
      if (idx == -1)
        destInputs.push_back(sourceOutput);
      else
        destInputs.push_back(inputs[idx]);
    }

    int destInput = bitsToInt(destInputs);
    bool destOutput = (destLut.ashr(destInput) & 1).getBoolValue();
    composedLookupTable.push_back(destOutput);

    // LLVM_DEBUG({
    //   llvm::dbgs() << "For input: " << possibleInput << "\n";
    //   llvm::dbgs() << "\tsourceInput = " << sourceInput << ", sourceOutput = " << sourceOutput << "\n";
    //   llvm::dbgs() << "\tdestInput = " << destInput << ", destOutput = " << destOutput << "\n";
    // });
  

  }
  LLVM_DEBUG({
    llvm::dbgs() << "Composed bits are: ";
    for (auto b : composedLookupTable) llvm::dbgs() << b;
    llvm::dbgs() << "\n";
  });

  return bitvecToAPInt(llvm::reverse(composedLookupTable));
  // return bitsToIntFlip(composedLookupTable);
}

mlir::APInt getMergedLookupTable(comb::TruthTableOp user,
                                 comb::TruthTableOp lutToMerge,
                                 mlir::SetVector<Value> userInputs) {
  llvm::DenseMap<Value, int> inputIndices;
  for (const auto& [idx, input] : llvm::enumerate(userInputs))
    inputIndices.insert({input, idx});

  mlir::SmallVector<int> sourceIdxs;
  mlir::SmallVector<int> destIdxs;

  for (auto sourceInput : lutToMerge.getLookupTableInputs())
    sourceIdxs.push_back(inputIndices[sourceInput]);

  for (auto destInput : user.getLookupTableInputs()) {
    if (inputIndices.contains(destInput))
      destIdxs.push_back(inputIndices[destInput]);
    else
      destIdxs.push_back(-1);
  }

  mlir::APInt mergedLookupTable = composeLookupTables(
      sourceIdxs, lutToMerge.getLookupTable().getValue(),
      destIdxs, user.getLookupTable().getValue());

  return mergedLookupTable;
}

mlir::FailureOr<LutMergeResult> mergeLutsIfPossible(
    comb::TruthTableOp user, comb::TruthTableOp lutToMerge,
    mlir::OpBuilder& builder) {
  mlir::SetVector<Value> userInputs;
  for (auto input : user.getLookupTableInputs()) {
    if (input.getDefiningOp<comb::TruthTableOp>() == lutToMerge) {
      for (auto sourceInput : lutToMerge.getLookupTableInputs())
        userInputs.insert(sourceInput);
      continue;
    }
    userInputs.insert(input);
  }

  mlir::APInt mergedLookupTable =
      getMergedLookupTable(user, lutToMerge, userInputs);
  auto lookupTable = builder.getIntegerAttr(
      builder.getIntegerType(1 << userInputs.size(), false), mergedLookupTable);

  auto synthesizer = ArithmeticLutSynthesizer::getInstance();
  auto synthesisResult = synthesizer.synthesize(lookupTable);
  if (mlir::failed(synthesisResult)) return mlir::failure();

  return LutMergeResult{.arithmeticLookupTable = *synthesisResult,
                        .lookupTable = lookupTable,
                        .userInputs = userInputs.takeVector()};
}

}  // namespace heir
}  // namespace mlir