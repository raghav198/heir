#include "lib/Target/OpenFhePke/OpenFheBinEmitter.h"

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir::heir::openfhe {

void registerToOpenFheBinTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-openfhe-bin",
      "translate the openfhe dialect to C++ code against the OpenFHE bin API",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToOpenFheBin(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, func::FuncDialect, lwe::LWEDialect,
                        openfhe::OpenfheDialect, memref::MemRefDialect>();
      });
}

LogicalResult translateToOpenFheBin(mlir::Operation *op,
                                    llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  OpenFheBinEmitter emitter(os, &variableNames);
  return emitter.translate(*op);
}

LogicalResult OpenFheBinEmitter::translate(Operation &operation) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(operation)
          .Case<memref::LoadOp>([&](auto load) { return printOperation(load); })
          .Case<memref::StoreOp>([&](auto store) { return printOperation(store); })
          .Case<memref::AllocOp>([&](auto alloc) { return printOperation(alloc); })
          .Case<openfhe::LWEMulConstOp>(
              [&](auto mul) { return printOperation(mul); })
          .Case<openfhe::LWEAddOp>(
              [&](auto add) { return printOperation(add); })
          .Case<openfhe::MakeLutOp>(
            [&](auto makeLut) { return printOperation(makeLut); })
          .Case<openfhe::EvalFuncOp>(
            [&](auto evalFunc) { return printOperation(evalFunc); })
          .Default([&](auto &op) {
            return OpenFhePkeEmitter::translate(operation);
          });
  return status;
}

LogicalResult OpenFheBinEmitter::printOperation(memref::LoadOp load) {
  if (failed(emitTypedAssignPrefix(load.getResult()))) {
    return failure();
  }
  os << variableNames->getNameForValue(load.getMemRef()) << "["
     << variableNames->getNameForValue(load.getIndices()[0]) << "];\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(memref::StoreOp store) {
  os << variableNames->getNameForValue(store.getMemRef()) << "[";
  os << variableNames->getNameForValue(store.getIndices()[0]);
  os << "] = " << variableNames->getNameForValue(store.getValueToStore()) << ";\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(memref::AllocOp alloc) {
  auto typeResult = convertType(*alloc->getResultTypes().begin());
  if (failed(typeResult)) return failure();
  os << typeResult.value() << " ";
  os << variableNames->getNameForValue(alloc.getResult()) << "(";
  os << alloc.getResult().getType().getShape()[0] << ");\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::LWEMulConstOp mul) {
  if (failed(emitTypedAssignPrefix(mul.getResult()))) {
    return failure();
  }
  os << variableNames->getNameForValue(mul.getCiphertext()) << ";\n";
  return printEvalMethod(mul.getResult(), mul.getCryptoContext(),
                         {mul.getResult(), mul.getConstant()}, "EvalMulConstEq");
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::LWEAddOp add) {
  if (failed(emitTypedAssignPrefix(add.getResult()))) {
    return failure();
  }
  os << variableNames->getNameForValue(add.getOperand(1)) << ";\n";
  return printEvalMethod(add.getResult(), add.getCryptoContext(),
                         {add.getResult(), add.getOperand(2)}, "EvalAddEq");
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::MakeLutOp makeLut) {
  emitAutoAssignPrefix(makeLut.getOutput());

  os << variableNames->getNameForValue(makeLut.getCryptoContext())
     << ".GenerateLUTviaFunction([](auto m, auto p) {\n";
  os.indent();
  for (auto val : makeLut.getValues()) {
    os << "if (m == " << val << ") return 1;\n";
  }
  os << "return 0;\n";
  os.unindent();
  os << "}, 8);\n";

  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::EvalFuncOp evalFunc) {
  return printEvalMethod(evalFunc.getResult(), evalFunc.getCryptoContext(),
                         {evalFunc.getInput(), evalFunc.getLut()}, "EvalFunc");
}

}  // namespace mlir::heir::openfhe
