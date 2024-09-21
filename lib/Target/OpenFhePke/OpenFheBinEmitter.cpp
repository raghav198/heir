#include "lib/Target/OpenFhePke/OpenFheBinEmitter.h"

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Target/Utils.h"
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "llvm/ADT/TypeSwitch.h"                       // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project

namespace mlir::heir::openfhe {

// clang-format off
constexpr std::string_view prelude = R"cpp(
#include <openfhe.h>  // from @openfhe

#include <algorithm>
#include <utility>
#include <vector>

using namespace lbcrypto;

using BinFHEContextT = std::shared_ptr<BinFHEContext>;
using LWESchemeT = std::shared_ptr<LWEEncryptionScheme>;

constexpr int ptxt_mod = 8;

std::vector<LWECiphertext> encrypt(BinFHEContextT cc, LWEPrivateKey sk,
                                    int value) {
  std::vector<lbcrypto::LWECiphertext> encrypted_bits;
  for (int i = 0; i < 8; i++) {
    int bit = (value & (1 << i)) >> i;
    encrypted_bits.push_back(
        cc->Encrypt(sk, bit, BINFHE_OUTPUT::SMALL_DIM, ptxt_mod));
  }
  return encrypted_bits;
}

int decrypt(BinFHEContextT cc, LWEPrivateKey sk,
            std::vector<LWECiphertext> encrypted) {
  int result = 0;

  std::reverse(encrypted.begin(), encrypted.end());

  for (LWECiphertext encrypted_bit : encrypted) {
    LWEPlaintext bit;
    cc->Decrypt(sk, encrypted_bit, &bit, ptxt_mod);
    result *= 2;
    result += bit;
  }
  return result;
}

LWECiphertext copy(LWECiphertext ctxt) {
  LWECiphertext copied = std::make_shared<LWECiphertextImpl>(ctxt->GetA(), ctxt->GetB());
  return copied;
}

)cpp";
// clang-format on

llvm::SmallVector<std::pair<int, int>> getIntervals(
    const llvm::ArrayRef<int> &values) {
  llvm::SmallVector<std::pair<int, int>> intervals;
  std::pair<int, int> current{values[0], values[0]};
  for (int value : values) {
    if (value == current.second + 1) {
      current.second = value;
    } else if (value != current.second) {
      intervals.push_back(current);
      current = {value, value};
    }
  }
  if (intervals.end()->first != current.first) intervals.push_back(current);
  return intervals;
}

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
          .Case<mlir::ModuleOp>(
              [&](auto module) { return printOperation(module); })
          .Case<memref::LoadOp>([&](auto load) { return printOperation(load); })
          .Case<memref::StoreOp>(
              [&](auto store) { return printOperation(store); })
          .Case<memref::AllocOp>(
              [&](auto alloc) { return printOperation(alloc); })
          .Case<openfhe::GetLWESchemeOp>(
              [&](auto getScheme) { return printOperation(getScheme); })
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

LogicalResult OpenFheBinEmitter::printOperation(mlir::ModuleOp module) {
  os << prelude << "\n";
  for (Operation &op : module) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
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
  os << "] = " << variableNames->getNameForValue(store.getValueToStore())
     << ";\n";
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

LogicalResult OpenFheBinEmitter::printOperation(
    openfhe::GetLWESchemeOp getScheme) {
  auto cryptoContext = getScheme.getCryptoContext();
  emitAutoAssignPrefix(getScheme.getResult());
  os << variableNames->getNameForValue(cryptoContext) << "->GetLWEScheme();\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::LWEMulConstOp mul) {
  return printInPlaceEvalMethod(mul.getResult(), mul.getCryptoContext(),
                                {mul.getCiphertext(), mul.getConstant()},
                                "EvalMultConstEq");
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::LWEAddOp add) {
  return printInPlaceEvalMethod(add.getResult(), add.getCryptoContext(),
                                {add.getOperand(1), add.getOperand(2)},
                                "EvalAddEq");
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::MakeLutOp makeLut) {
  emitAutoAssignPrefix(makeLut.getOutput());

  os << variableNames->getNameForValue(makeLut.getCryptoContext())
     << "->GenerateLUTviaFunction([](NativeInteger m, NativeInteger p) -> NativeInteger {\n";
  os.indent();
  for (auto val : makeLut.getValues()) {
    os << llvm::formatv("if (m == {0}) return 1;\n", val);
  }

  os << "return 0;\n";
  os.unindent();
  os << "}, ptxt_mod);\n";

  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::EvalFuncOp evalFunc) {
  return printEvalMethod(evalFunc.getResult(), evalFunc.getCryptoContext(),
                         {evalFunc.getInput(), evalFunc.getLut()}, "EvalFunc");
}

LogicalResult OpenFheBinEmitter::printInPlaceEvalMethod(
    mlir::Value result, mlir::Value cryptoContext, mlir::ValueRange operands,
    std::string_view op) {
  if (failed(emitTypedAssignPrefix(result))) {
    return failure();
  }
  os << "copy(" << variableNames->getNameForValue(*operands.begin()) << ");\n";
  os << variableNames->getNameForValue(cryptoContext) << "->" << op << "("
     << variableNames->getNameForValue(result) << ", ";
  os << commaSeparatedValues(
      mlir::ValueRange(operands.begin() + 1, operands.end()),
      [&](mlir::Value value) { return variableNames->getNameForValue(value); });
  os << ");\n";
  return success();
}

}  // namespace mlir::heir::openfhe
