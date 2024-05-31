#include "lib/Transforms/MergeLUTs/LutMergingUtils.h"
#include "lib/Dialect/Comb/IR/CombOps.h"
#include "lib/Transforms/MergeLUTs/SynthesizeArithmeticLut.h"
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project

#include <algorithm>
#include <ostream>
#include <vector>

#define DEBUG_TYPE "merge-luts"

namespace mlir
{
namespace heir
{

graph::Graph<mlir::Operation*> makeLUTGraph(mlir::Operation *root)
{
    graph::Graph<mlir::Operation*> lutsDependencyGraph;

    root->walk([&lutsDependencyGraph](mlir::Operation *op) {
        if (llvm::isa<comb::TruthTableOp>(op))
            lutsDependencyGraph.addVertex(op);
    });

    for (mlir::Operation* vertex : lutsDependencyGraph.getVertices())
    {
        for (mlir::Operation* user : vertex->getUsers())
            lutsDependencyGraph.addEdge(vertex, user);
    }

    return lutsDependencyGraph;
}

template <class T>
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const std::vector<T>& vec)
{
    os << "[";
    for (auto t : vec) os << t;
    os << "]";
    
    return os;
}

std::vector<bool> intToBits(int value, int width, bool flip = false)
{
    std::vector<bool> bits;
    bits.reserve(width);
    for (int i = 0; i < width; i++)
        bits.push_back((value & (1 << (flip ? i : (width - i - 1)))) != 0);
    return bits;
}

int bitsToIntFlip(const std::vector<bool>& bits)
{
    int value = 0;
    int power = 0;
    for (const auto& bit : bits)
    {
        if (bit) value += (1 << power);
        power++;
    }
    return value;
}

int bitsToInt(const std::vector<bool>& bits)
{
    int value = 0;
    int power = 0;

    for (const auto& bit : llvm::reverse(bits))
    {
        if (bit) value += (1 << power);
        power++;
    }
    return value;
}

int composeLookupTables(
    const llvm::SmallVector<int>& sourceIdxs, int sourceLut, 
    const llvm::SmallVector<int>& destIdxs, int destLut)
{
    int composedInputCount = std::max(
        *std::max_element(sourceIdxs.begin(), sourceIdxs.end()),
        *std::max_element(destIdxs.begin(), destIdxs.end())) + 1;

    llvm::DenseMap<int, int> sourceLocs;
    llvm::DenseMap<int, int> destLocs;

    std::vector<bool> sourceLutTable = intToBits(sourceLut, 1 << sourceIdxs.size(), true);
    std::vector<bool> destLutTable = intToBits(destLut, 1 <<  destIdxs.size(), true);

    for (const auto& [loc, index] : llvm::enumerate(sourceIdxs))
        sourceLocs.insert({index, loc});

    for (const auto& [loc, index] : llvm::enumerate(destIdxs))
        destLocs.insert({index, loc});

    std::vector<bool> composedLookupTable;

    for (int possibleInput = 0; possibleInput < (1 << composedInputCount); possibleInput++)
    {
        auto inputs = intToBits(possibleInput, composedInputCount);
        std::vector<bool> sourceInputs;
        for (auto idx : sourceIdxs) sourceInputs.push_back(inputs[idx]);
        bool sourceOutput = sourceLutTable[bitsToInt(sourceInputs)];

        std::vector<bool> destInputs;
        for (auto idx : destIdxs)
        {
            if (idx == -1) destInputs.push_back(sourceOutput);
            else destInputs.push_back(inputs[idx]);
        }
        
        composedLookupTable.push_back(destLutTable[bitsToInt(destInputs)]);
    }

    return bitsToIntFlip(composedLookupTable);
}

unsigned int getMergedLookupTable(comb::TruthTableOp user, comb::TruthTableOp lutToMerge, mlir::SetVector<Value> userInputs)
{
    llvm::DenseMap<Value, int> inputIndices;
    for (const auto& [idx, input] : llvm::enumerate(userInputs))
        inputIndices.insert({input, idx});

    mlir::SmallVector<int> sourceIdxs;
    mlir::SmallVector<int> destIdxs;
    
    for (auto sourceInput : lutToMerge.getLookupTableInputs())
        sourceIdxs.push_back(inputIndices[sourceInput]);

    for (auto destInput : user.getLookupTableInputs())
    {
        if (inputIndices.contains(destInput)) destIdxs.push_back(inputIndices[destInput]);
        else destIdxs.push_back(-1);
    }

    unsigned int mergedLookupTable = composeLookupTables(
        sourceIdxs, lutToMerge.getLookupTable().getValue().getZExtValue(),
        destIdxs, user.getLookupTable().getValue().getZExtValue());

    return mergedLookupTable;
}

mlir::FailureOr<LutMergeResult> mergeLutsIfPossible(comb::TruthTableOp user, comb::TruthTableOp lutToMerge, mlir::OpBuilder &builder)
{
    mlir::SetVector<Value> userInputs;
    for (auto input : user.getLookupTableInputs())
    {
        if (input.getDefiningOp<comb::TruthTableOp>() == lutToMerge)
        {
            for (auto sourceInput : lutToMerge.getLookupTableInputs())
                userInputs.insert(sourceInput);
            continue;
        }
        userInputs.insert(input);
    }

    unsigned int mergedLookupTable = getMergedLookupTable(user, lutToMerge, userInputs);
    auto lookupTable = builder.getIntegerAttr(builder.getIntegerType(1 << userInputs.size(), false), mergedLookupTable);

    auto synthesizer = ArithmeticLutSynthesizer::getInstance();
    auto synthesisResult = synthesizer.synthesize(lookupTable);
    if (mlir::failed(synthesisResult))
        return mlir::failure();

    return LutMergeResult{.arithmeticLookupTable = *synthesisResult, .lookupTable = lookupTable, .userInputs = userInputs.takeVector()};
}

} // namespace heir
} // namespace mlir