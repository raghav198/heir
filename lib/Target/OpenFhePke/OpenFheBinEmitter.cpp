#include "lib/Target/OpenFhePke/OpenFheBinEmitter.h"

#include <numeric>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Target/Utils.h"
#include "llvm/ADT/TypeSwitch.h"                       // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

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

// I'm done apologizing...
struct view_t {
  int offset;
  int stride;
  int size;

  view_t(int offset, int stride, int size) : offset(offset), stride(stride), size(size) {}
  void apply(view_t other) {
      offset += other.offset * stride;
      stride *= other.stride;
      size = other.size;
  }
};

template<class T>
constexpr int dim = 0;

template<class T>
constexpr int dim<std::vector<T>> = 1 + dim<T>;

template<class T>
constexpr int dim<std::vector<T>&> = 1 + dim<T>;

template <class T>
class vector_view;

template <class T>
class vector_view<std::vector<T>> {
  std::vector<T>& data;
  std::vector<T> owned_data;
  std::vector<view_t> views;
public:
  vector_view(std::vector<T>& data, std::vector<view_t> views) : data(data), views(views) {}
  vector_view(std::vector<T>& data) : data(data) {
    for (int i = 0; i < dim<std::vector<T>>; i++) {
      views.emplace_back(0, 1, -1);
    }
  }
  vector_view(std::initializer_list<T> elems) : data(owned_data), owned_data(elems) {
    for (int i = 0; i < dim<decltype(data)>; i++) {
      views.emplace_back(0, 1, -1);
    }
  }

  vector_view(const vector_view<T>& other) : data(other.data), views(other.views) {}

  constexpr int rank() {
    return dim<decltype(data)>;
  }

  size_t size() const {
    if (views[0].size == -1) return data.size();
    return views[0].size;
  }

  auto operator[](size_t index) {
    auto& vec = data[views[0].offset + index * views[0].stride];
    if constexpr (dim<decltype(data)> == 1) {
      return vec;
    } else {
      vector_view<T> new_view(vec, std::vector<view_t>(views.begin() + 1, views.end()));
      return new_view;
    }
  }

  auto operator[](size_t index) const {
    auto& vec = data[views[0].offset + index * views[0].stride];
    if constexpr (dim<decltype(data)> == 1) {
      return vec;
    } else {
      vector_view<T> new_view(vec, std::vector<view_t>(views.begin() + 1, views.end()));
      return new_view;
    }
  }


  auto subview(std::vector<view_t> slices) {
    auto copied = *this;
    for (int i = 0; i < slices.size(); i++) {
      copied.views[i].apply(slices[i]);
    }
    return copied;
  }

  std::vector<T> copy() const {
    std::vector<T> copied;
    for (int i = 0; i < size(); i++) {
      if constexpr (dim<decltype(data)> == 1) {
        copied.push_back(this->operator[](i));
      } else {
        copied.push_back(this->operator[](i).copy());
      }
    }
    return copied;
  }
};

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
                        openfhe::OpenfheDialect, memref::MemRefDialect,
                        scf::SCFDialect>();
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
          .Case<memref::SubViewOp>(
            [&](auto subview) { return printOperation(subview); })
          .Case<memref::CopyOp>(
            [&](auto copy) { return printOperation(copy); })
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
          .Case<scf::IfOp>([&](auto ifOp) { return printOperation(ifOp); })
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

SmallVector<std::string> OpenFheBinEmitter::getStaticDynamicArgs(SmallVector<mlir::Value> dynamicArgs, ArrayRef<long long> staticArgs) {
  SmallVector<std::string> args;
  int dynamicIndex = 0;
  for (long long staticArg : staticArgs) {
    if (staticArg == ShapedType::kDynamic) {
      args.push_back(variableNames->getNameForValue(dynamicArgs[dynamicIndex++]));
    } else {
      args.push_back(std::to_string(staticArg));
    }
  }
  return args;
}

LogicalResult OpenFheBinEmitter::printOperation(memref::SubViewOp subview) {
  SmallVector<std::string> offsets = getStaticDynamicArgs(subview.getOffsets(), subview.getStaticOffsets());
  SmallVector<std::string> strides = getStaticDynamicArgs(subview.getStrides(), subview.getStaticStrides());
  SmallVector<std::string> sizes = getStaticDynamicArgs(subview.getSizes(), subview.getStaticSizes());

  SmallVector<std::string> viewStrings;
  for (int i = 0; i < offsets.size(); i++) {
    SmallString<8> viewString;
    llvm::raw_svector_ostream ss(viewString);
    ss << "view_t(" << offsets[i] << ", " << strides[i] << ", " << sizes[i]
       << ")";
    viewStrings.push_back(viewString.str().str());
  }
  std::string args = std::accumulate(
      std::next(viewStrings.begin()), viewStrings.end(), viewStrings[0],
      [&](const std::string &a, const std::string &b) { return a + ", " + b; });

  emitAutoAssignPrefix(subview.getResult());
  os << variableNames->getNameForValue(subview.getSource()) << ".subview({" << args << "});\n";
  return success();
  
}

LogicalResult OpenFheBinEmitter::printOperation(memref::CopyOp copy) {
  os << variableNames->getNameForValue(copy.getTarget()) << " = ";
  os << variableNames->getNameForValue(copy.getSource()) << ".copy()\n";
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
     << "->GenerateLUTviaFunction([](NativeInteger m, NativeInteger p) -> "
        "NativeInteger {\n";
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

LogicalResult OpenFheBinEmitter::printOperation(scf::IfOp ifOp) {
  if (failed(emitType(ifOp->getResultTypes().front()))) {
    return failure();
  }

  auto &thenBlock = *ifOp.thenBlock();
  auto &elseBlock = *ifOp.elseBlock();

  // Assumes the type is default-constructible (I don't have a good way around
  // this)
  auto resultName = variableNames->getNameForValue(ifOp->getResult(0));
  os << " " << resultName << ";\n";

  os << "if (" << variableNames->getNameForValue(ifOp.getCondition())
     << ") {\n";
  os.indent();
  for (auto &op : thenBlock.getOperations()) {
    if (auto yieldOp = mlir::dyn_cast<scf::YieldOp>(op)) {
      os << resultName << " = "
         << variableNames->getNameForValue(yieldOp->getOperand(0)) << ";\n";
    } else {
      if (failed(translate(op))) return failure();
    }
  }
  os.unindent();
  os << "} else {\n";
  os.indent();
  for (auto &op : elseBlock.getOperations()) {
    if (auto yieldOp = mlir::dyn_cast<scf::YieldOp>(op)) {
      os << resultName << " = "
         << variableNames->getNameForValue(yieldOp->getOperand(0)) << ";\n";
    } else {
      if (failed(translate(op))) return failure();
    }
  }
  os.unindent();
  os << "}\n";

  return success();
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
