#include "lib/Transforms/MergeLUTs/MergeLUTs.h"

#include "lib/Dialect/Comb/IR/CombOps.h"
#include "lib/Graph/Graph.h"
#include "lib/Transforms/MergeLUTs/LutMergingUtils.h"
#include "lib/Transforms/MergeLUTs/SynthesizeArithmeticLut.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project

#include <iostream>

#define DEBUG_TYPE "merge-luts"

namespace mlir
{
namespace heir
{

#define GEN_PASS_DEF_MERGELUTS
#include "lib/Transforms/MergeLUTs/MergeLUTs.h.inc"

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

mlir::Operation *nextLutToMerge(mlir::Operation *root, graph::Graph<mlir::Operation *>& lutGraph)
{
    static std::set<mlir::Operation *> alreadyMerged;

    // Return the vertex with the minimum non-zero out-degree, or nullptr if no such vertex exists
    lutGraph = makeLUTGraph(root);
    mlir::Operation *next = nullptr;
    int outDegree = lutGraph.getVertices().size(); // larger than the maximum possible out degree

    for (auto *vertex : lutGraph.getVertices())
    {
        if (alreadyMerged.find(vertex) != alreadyMerged.end()) continue;

        auto outEdges = lutGraph.edgesOutOf(vertex);
        if (outEdges.empty()) continue;
        if (outEdges.size() < outDegree) next = vertex;
    }

    alreadyMerged.insert(next);
    return next;
}

unsigned int getMergedLookupTable(mlir::Operation *user, mlir::Operation *lutToMerge, mlir::SetVector<Value> userInputs)
{
    auto castUser = llvm::cast<comb::TruthTableOp>(user);
    auto castLutToMerge = llvm::cast<comb::TruthTableOp>(lutToMerge);

    llvm::DenseMap<Value, int> inputIndices;
    for (const auto& [idx, input] : llvm::enumerate(userInputs))
        inputIndices.insert({input, idx});

    mlir::SmallVector<int> sourceIdxs;
    mlir::SmallVector<int> destIdxs;
    
    for (auto sourceInput : castLutToMerge.getLookupTableInputs())
        sourceIdxs.push_back(inputIndices[sourceInput]);

    for (auto destInput : castUser.getLookupTableInputs())
    {
        if (inputIndices.contains(destInput)) destIdxs.push_back(inputIndices[destInput]);
        else destIdxs.push_back(-1);
    }

    unsigned int mergedLookupTable = composeLookupTables(
        sourceIdxs, castLutToMerge.getLookupTable().getValue().getZExtValue(),
        destIdxs, castUser.getLookupTable().getValue().getZExtValue());

    return mergedLookupTable;
}

bool performSingleMerge(mlir::Operation *user, mlir::Operation *lutToMerge, mlir::OpBuilder& builder)
{
    auto castLutToMerge = llvm::cast<comb::TruthTableOp>(lutToMerge);
    auto castUser = llvm::cast<comb::TruthTableOp>(user);

    LLVM_DEBUG({
        llvm::dbgs() << "lutToMerge = " << castLutToMerge << "\n";
        llvm::dbgs() << "user = " << castUser << "\n";
        for (auto input : castUser.getLookupTableInputs())
            llvm::dbgs() << "- user LUT input " << input << "\n";

        for (auto input : castUser.getInputs())
            llvm::dbgs() << "- user input " << input << "\n";
    });

    SetVector<Value> userInputs;
    for (auto input : castUser.getLookupTableInputs())
    {
        if (input.getDefiningOp() == lutToMerge)
        {
            for (auto sourceInput : castLutToMerge.getLookupTableInputs())
                userInputs.insert(sourceInput);
            continue;
        }
        userInputs.insert(input);
    }

    unsigned int mergedLookupTable = getMergedLookupTable(user, lutToMerge, userInputs);
    
    auto synthesizer = ArithmeticLutSynthesizer::getInstance();

    auto lookupTable = builder.getIntegerAttr(builder.getIntegerType(1 << userInputs.size(), false), mergedLookupTable);
    auto synthesisResult = synthesizer.synthesize(lookupTable);
    if (mlir::failed(synthesisResult))
    {
        LLVM_DEBUG({ llvm::dbgs() << "Synthesis for " << mergedLookupTable << " failed, skipping...\n"; });
        return false;
    }

    LLVM_DEBUG({
        llvm::dbgs() << "Successfully synthesized LUT: " << synthesisResult->lookupTable << "\n";
        for (auto coeff : synthesisResult->coefficients)
            llvm::dbgs() << "\tCoefficient: " << coeff << "\n";
    });
    
    builder.setInsertionPointAfter(user);
    auto lookupTableOp = builder.create<comb::TruthTableOp>(
        user->getLoc(), userInputs.takeVector(), lookupTable);

    LLVM_DEBUG({
        llvm::dbgs() << "Built new op: " << lookupTableOp << "\n";
        llvm::dbgs() << "Replacing all uses of " << user->getResult(0) << " with new op\n";
    });
    
    user->getResult(0).replaceAllUsesWith({lookupTableOp});
    return true;
}

struct MergeLUTs : public impl::MergeLUTsBase<MergeLUTs> {
    using MergeLUTsBase::MergeLUTsBase;

    void runOnOperation() override
    {
        graph::Graph<mlir::Operation *> lutGraph;
        mlir::Operation *lutToMerge;

        while ((lutToMerge = nextLutToMerge(getOperation(), lutGraph)) != nullptr)
        {
            LLVM_DEBUG({
                for (auto *user : lutGraph.edgesOutOf(lutToMerge))
                    llvm::dbgs() << "Merging " << *lutToMerge << " into " << *user << "\n";
            });

            std::vector<mlir::Operation *> successfulMerges;

            for (auto *user : lutGraph.edgesOutOf(lutToMerge))
            {
                mlir::OpBuilder builder(&getContext());
                if (performSingleMerge(user, lutToMerge, builder)) successfulMerges.push_back(user);
            }

            for (auto *user : successfulMerges)
                user->erase();

            if (successfulMerges.size() == lutGraph.edgesOutOf(lutToMerge).size())
                lutToMerge->erase();
        }
    }
};

} // namespace heir
} // namespace mlir