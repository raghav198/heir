#include "lib/Transforms/MergeLUTs/MergeLUTs.h"

#include "lib/Dialect/Comb/IR/CombOps.h"
#include "lib/Graph/Graph.h"
#include "lib/Transforms/MergeLUTs/LutMergingUtils.h"
#include "lib/Transforms/MergeLUTs/SynthesizeArithmeticLut.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project

#include <iostream>

#define DEBUG_TYPE "merge-luts"

namespace mlir
{
namespace heir
{

#define GEN_PASS_DEF_MERGELUTS
#include "lib/Transforms/MergeLUTs/MergeLUTs.h.inc"

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

bool performSingleMerge(mlir::Operation *user, mlir::Operation *lutToMerge, mlir::OpBuilder& builder)
{
    auto mergeResult = mergeLutsIfPossible(llvm::cast<comb::TruthTableOp>(user), llvm::cast<comb::TruthTableOp>(lutToMerge), builder);
    if (mlir::failed(mergeResult)) return false;

    auto [userInputs, lookupTable, synthesisResult] = *mergeResult;

    builder.setInsertionPointAfter(user);
    auto lookupTableOp = builder.create<comb::TruthTableOp>(
        user->getLoc(), userInputs, lookupTable);


    lookupTableOp->setAttr("coefficients", builder.getDenseI32ArrayAttr(synthesisResult.coefficients));
    lookupTableOp->setAttr("prepped_lut", builder.getIntegerAttr(builder.getIntegerType(synthesisResult.lutSize), synthesisResult.lookupTable));

    LLVM_DEBUG({
        llvm::dbgs() << "Built new op: " << lookupTableOp << "\n";
        llvm::dbgs() << "Replacing all uses of " << user->getResult(0) << " with new op\n";
    });
    
    user->getResult(0).replaceAllUsesWith({lookupTableOp});
    return true;
}

void executeMerge(mlir::Operation *user, const LutMergeResult& mergeResult, mlir::OpBuilder &builder)
{
    LLVM_DEBUG(llvm::dbgs() << "Executing merge on " << user << "\n");
    auto [userInputs, lookupTable, synthesisResult] = mergeResult;

    builder.setInsertionPointAfter(user);
    auto lookupTableOp = builder.create<comb::TruthTableOp>(
        user->getLoc(), userInputs, lookupTable);


    lookupTableOp->setAttr("coefficients", builder.getDenseI32ArrayAttr(synthesisResult.coefficients));
    lookupTableOp->setAttr("prepped_lut", builder.getIndexAttr(synthesisResult.lookupTable));

    LLVM_DEBUG({
        llvm::dbgs() << "Built new op: " << lookupTableOp << "\n";
        llvm::dbgs() << "Replacing all uses of " << user->getResult(0) << " with new op\n";
    });
    
    user->getResult(0).replaceAllUsesWith({lookupTableOp});
}

struct MergeLUTs : public impl::MergeLUTsBase<MergeLUTs> {
    using MergeLUTsBase::MergeLUTsBase;

    void runOnOperation() override
    {
        graph::Graph<mlir::Operation *> lutGraph;
        mlir::Operation *lutToMerge;

        mlir::OpBuilder builder(&getContext());

        while ((lutToMerge = nextLutToMerge(getOperation(), lutGraph)) != nullptr)
        {
            LLVM_DEBUG({
                for (auto *user : lutGraph.edgesOutOf(lutToMerge))
                    llvm::dbgs() << "Merging " << *lutToMerge << " into " << *user << "\n";
            });

            mlir::DenseMap<mlir::Operation *, LutMergeResult> mergeResults;

            for (auto *user : lutGraph.edgesOutOf(lutToMerge))
            {
                LLVM_DEBUG(llvm::dbgs() << "Try merge " << *lutToMerge << " to " << *user << "\n");
                auto result = mergeLutsIfPossible(llvm::cast<comb::TruthTableOp>(user), llvm::cast<comb::TruthTableOp>(lutToMerge), builder);
                if (mlir::succeeded(result)) mergeResults.insert({user, *result});
                else LLVM_DEBUG(llvm::dbgs() << "Merge " << *lutToMerge << " to " << *user << " failed\n");
            }

            LLVM_DEBUG(llvm::dbgs() << "Of " << lutGraph.edgesOutOf(lutToMerge).size() << " edges, " << mergeResults.size() << " succeeded\n");
            if (mergeResults.size() != lutGraph.edgesOutOf(lutToMerge).size()) continue;
            for (auto& [user, result] : mergeResults)
            {
                executeMerge(user, result, builder);
                user->erase();
            }
            lutToMerge->erase();
        }
    }
};

} // namespace heir
} // namespace mlir