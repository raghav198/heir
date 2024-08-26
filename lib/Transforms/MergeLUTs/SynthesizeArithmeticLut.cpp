#include "lib/Transforms/MergeLUTs/SynthesizeArithmeticLut.h"
#include <algorithm>
#include <iterator>

#include "ortools/constraint_solver/constraint_solver.h"
#include "llvm/include/llvm/Support/Debug.h"

#define DEBUG_TYPE "merge-luts"

namespace mlir
{
namespace heir
{

ArithmeticLut::ArithmeticLut(
    const std::vector<int>& ones, 
    const std::vector<int>& zeros, 
    const std::vector<int>& coefficients,
    int lutSize) : coefficients(coefficients), lookupTable(0), lutSize(lutSize)
{
    std::set<int> oneSet, zeroSet;
    auto positivize = [&](int val) {
        while (val < 0) val += lutSize;
        return val % lutSize;
    };
    
    std::transform(ones.begin(), ones.end(), std::inserter(oneSet, oneSet.begin()), positivize);
    std::transform(zeros.begin(), zeros.end(), std::inserter(zeroSet, zeroSet.begin()), positivize);

    for (auto o : oneSet)
    {
        // verify that ones and zeros are disjoint
        assert(zeroSet.find(o) == zeroSet.end() && "one-set and zero-set must be disjoint!");
        lookupTable += (1 << o);
    }
}

mlir::IntegerAttr ArithmeticLut::buildAttr(mlir::OpBuilder &builder)
{
    auto type = builder.getIntegerType(lutSize, false);
    return builder.getIntegerAttr(type, lookupTable);
}

ArithmeticLutSynthesizer& ArithmeticLutSynthesizer::getInstance()
{
    static ArithmeticLutSynthesizer instance;
    return instance;
}

template <typename T, typename = std::enable_if<
    std::is_same<T, operations_research::IntExpr>::value ||
    std::is_same<T, operations_research::IntVar>::value>>
std::vector<int> resolve(const std::vector<T *>& exprs)
{
    std::vector<int> resolved;
    resolved.reserve(exprs.size());
    for (auto *t : exprs)
    {
        if constexpr (std::is_same<T, operations_research::IntVar>::value)
            resolved.push_back(t->Value());
        else
            resolved.push_back(t->Var()->Value());
    }
    // LLVM_DEBUG({
    //     llvm::dbgs() << "Resolved:\n";
    //     for (auto x : resolved)
    //         llvm::dbgs() << "- " << x << "\n";
    // });
    return resolved;
}
template <class T>
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const std::vector<T>& vec) {
  os << "[";
  for (auto t : vec) os << t;
  os << "]";

  return os;
}
mlir::FailureOr<ArithmeticLut> ArithmeticLutSynthesizer::doSynth(mlir::IntegerAttr lookupTable, int maxLutSize)
{

    if (synthesizedCache.contains(lookupTable))
        return synthesizedCache.lookup(lookupTable);

    std::vector<unsigned int> coefficients;
    std::vector<operations_research::IntVar *> coefficientVars;

    LLVM_DEBUG(llvm::dbgs() << "Synthesizing for table " << lookupTable << "\n");
    

    unsigned int arity = llvm::Log2_64(lookupTable.getType().getIntOrFloatBitWidth());
    mlir::APInt tableValues = lookupTable.getValue();
    // unsigned int tableValues = lookupTable.getUInt();

    LLVM_DEBUG({
        llvm::dbgs() << "\tArity: " << arity << "\n";
        llvm::dbgs() << "\tValues: " << tableValues << "\n";
    });

    operations_research::Solver solver("solver");
    auto *timeLimit = solver.MakeTimeLimit(absl::Milliseconds(1500));
    coefficientVars.reserve(arity);
    
    for (int i = 0; i < arity; i++) coefficientVars.push_back(solver.MakeIntVar(1 - maxLutSize, maxLutSize - 1));
    // for (int i = 0; i < arity; i++) coefficientVars[i]->SetValue(1);

    std::vector<operations_research::IntExpr *> ones;
    std::vector<operations_research::IntExpr *> zeros;

    for (int i = 0; i < (1 << arity); i++)
    {
        std::vector<int> input;
        input.reserve(arity);
        for (int j = 0; j < arity; j++) input.push_back((i & (1 << j)) >> j);

        auto *output = solver.MakeScalProd(coefficientVars, input);


        // LLVM_DEBUG(llvm::dbgs() << "For input " << i << ", output is " << output->Var()->Value() << "\n");

        solver.AddConstraint(solver.MakeLess(output, maxLutSize));
        solver.AddConstraint(solver.MakeGreater(output, -maxLutSize));

        bool expected = ((tableValues.ashr(i)) & 1).getBoolValue();
        // LLVM_DEBUG(llvm::dbgs() << "For input " << input << ", output is " << expected << "\n");
        
        if (expected) ones.push_back(output);
        else zeros.push_back(output);

        // LLVM_DEBUG(llvm::dbgs() << "Expected output " << (tableValues & (1 << i)) << "\n");
    }

    for (auto *o : ones)
    {
        for (auto *z : zeros)
        {
            auto *diff = solver.MakeDifference(o, z);
            solver.AddConstraint(solver.MakeNonEquality(diff, 0));
            solver.AddConstraint(solver.MakeNonEquality(diff, maxLutSize));
            solver.AddConstraint(solver.MakeNonEquality(diff, -maxLutSize));
        }
    }

    operations_research::DecisionBuilder *db = solver.MakePhase(
        coefficientVars, operations_research::Solver::CHOOSE_FIRST_UNBOUND, 
                                    operations_research::Solver::ASSIGN_MIN_VALUE);

    solver.NewSearch(db, timeLimit);
    while (solver.NextSolution())
    {
        std::vector<int> coefficients = resolve(coefficientVars);
        LLVM_DEBUG(llvm::dbgs() << "\tSUCCESS\n");
        std::reverse(coefficients.begin(), coefficients.end());
        return ArithmeticLut(resolve(ones), resolve(zeros), coefficients, maxLutSize);
    }
    solver.EndSearch();
    LLVM_DEBUG(llvm::dbgs() << "\tFAILURE\n");
    return mlir::failure();
}

mlir::FailureOr<ArithmeticLut> ArithmeticLutSynthesizer::synthesize(mlir::IntegerAttr lookupTable, int maxLutSize)
{
    if (!synthesizedCache.contains(lookupTable))
    {
        auto synthesized = doSynth(lookupTable, maxLutSize);
        synthesizedCache.insert({lookupTable, synthesized});
    }
    return synthesizedCache[lookupTable];
}

} // namespace heir
} // namespace mlir
