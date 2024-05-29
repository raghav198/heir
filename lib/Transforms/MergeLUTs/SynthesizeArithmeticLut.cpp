#include "lib/Transforms/MergeLUTs/SynthesizeArithmeticLut.h"
#include <algorithm>

#include "ortools/constraint_solver/constraint_solver.h"
#include "llvm/include/llvm/Support/Debug.h"

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
    std::set<int> oneSet(ones.begin(), ones.end());
    std::set<int> zeroSet(zeros.begin(), zeros.end());

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
    return resolved;
}

mlir::FailureOr<ArithmeticLut> ArithmeticLutSynthesizer::doSynth(mlir::IntegerAttr lookupTable, int maxLutSize)
{

    if (synthesizedCache.contains(lookupTable))
        return synthesizedCache.lookup(lookupTable);

    std::vector<unsigned int> coefficients;
    std::vector<operations_research::IntVar *> coefficientVars;

    llvm::dbgs() << "Synthesizing for table " << lookupTable << "\n";

    unsigned int arity = llvm::Log2_64(lookupTable.getType().getIntOrFloatBitWidth());
    unsigned int tableValues = lookupTable.getUInt();

    llvm::dbgs() << "\tArity: " << arity << "\n";
    llvm::dbgs() << "\tValues: " << tableValues << "\n";

    operations_research::Solver solver("solver");
    coefficientVars.reserve(arity);
    
    for (int i = 0; i < arity; i++) coefficientVars.push_back(solver.MakeIntVar(0, maxLutSize - 1));
    // for (int i = 0; i < arity; i++) coefficientVars[i]->SetValue(1);

    std::vector<operations_research::IntExpr *> ones;
    std::vector<operations_research::IntExpr *> zeros;

    for (int i = 0; i < (1 << arity); i++)
    {
        std::vector<int> input;
        input.reserve(arity);
        for (int j = 0; j < arity; j++) input.push_back((i & (1 << j)) >> j);

        auto *output = solver.MakeScalProd(coefficientVars, input);

        // llvm::dbgs() << "For input " << i << ", output is " << output->Var()->Value() << "\n";

        solver.AddConstraint(solver.MakeLess(output, maxLutSize));
        solver.AddConstraint(solver.MakeGreater(output, -maxLutSize));

        if (tableValues & (1 << i)) ones.push_back(output);
        else zeros.push_back(output);

        // llvm::dbgs() << "Expected output " << (tableValues & (1 << i)) << "\n";
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

    solver.NewSearch(db);
    while (solver.NextSolution())
    {
        std::vector<int> coefficients = resolve(coefficientVars);
        return ArithmeticLut(resolve(ones), resolve(zeros), coefficients, maxLutSize);
    }
    solver.EndSearch();

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
