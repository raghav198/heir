#include "lib/Transforms/MergeLUTs/MergeLUTs.h"

#include "lib/Dialect/Comb/IR/CombOps.h"
#include "lib/Graph/Graph.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
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

struct MergeLUTs : public impl::MergeLUTsBase<MergeLUTs> {
    using MergeLUTsBase::MergeLUTsBase;

    void runOnOperation() override
    {
        auto lutGraph = makeLUTGraph(getOperation());
        llvm::SmallVector<mlir::Operation *> luts(lutGraph.getVertices().begin(), lutGraph.getVertices().end());
        std::sort(luts.begin(), luts.end(), [&lutGraph](mlir::Operation *a, mlir::Operation *b) {
            return lutGraph.edgesOutOf(a).size() < lutGraph.edgesOutOf(b).size();
        });

        mlir::Operation *lutToMerge = nullptr;

        for (auto *lut : luts)
        {
            if (!lutGraph.edgesOutOf(lut).empty())
            {
                lutToMerge = lut;
                break;
            }
        }

        if (lutToMerge == nullptr) return;

        for (auto *user : lutGraph.edgesOutOf(lutToMerge))
        {
            LLVM_DEBUG({llvm::dbgs() << "Merging into: " << *user << "\n";});
            mlir::OpBuilder builder(&getContext());

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

            

            builder.setInsertionPointAfter(user);

            SetVector<Value> userInputs;
            for (auto input : llvm::cast<comb::TruthTableOp>(user).getLookupTableInputs())
            {
                if (input.getDefiningOp() == lutToMerge)
                {
                    for (auto sourceInput : llvm::cast<comb::TruthTableOp>(lutToMerge).getLookupTableInputs())
                        userInputs.insert(sourceInput);
                    continue;
                }
                userInputs.insert(input);
            }

            auto argType = builder.getIntegerType(1 << userInputs.size());
            auto lookupTable = builder.create<comb::TruthTableOp>(
                user->getLoc(), userInputs.takeVector(), builder.getIntegerAttr(argType, 0));

            LLVM_DEBUG({
                llvm::dbgs() << "Built new op: " << lookupTable << "\n";
                llvm::dbgs() << "Replacing all uses of " << user->getResult(0) << " with new op\n";
            });
            
            user->getResult(0).replaceAllUsesWith({lookupTable});
        }

        for (auto *user : lutGraph.edgesOutOf(lutToMerge))
            user->erase();
        lutToMerge->erase();
    }
};

} // namespace heir
} // namespace mlir