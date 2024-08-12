#ifndef LIB_TARGET_OPENFHEBIN_OPENFHEBINEMITTER_H_
#define LIB_TARGET_OPENFHEBIN_OPENFHEBINEMITTER_H_

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Target/OpenFhePke/OpenFhePkeEmitter.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project

namespace mlir::heir::openfhe {

void registerToOpenFheBinTranslation();
LogicalResult translateToOpenFheBin(mlir::Operation *op, llvm::raw_ostream &os);

class OpenFheBinEmitter : public OpenFhePkeEmitter {
 public:
  OpenFheBinEmitter(raw_ostream &os, SelectVariableNames *variableNames) : OpenFhePkeEmitter(os, variableNames) {}
  LogicalResult translate(::mlir::Operation &operation) override;

 private:
   LogicalResult printOperation(memref::LoadOp load);
   LogicalResult printOperation(memref::StoreOp store);
   LogicalResult printOperation(memref::AllocOp alloc);
   LogicalResult printOperation(openfhe::LWEMulConstOp mul);
   LogicalResult printOperation(openfhe::LWEAddOp add);
   LogicalResult printOperation(openfhe::MakeLutOp makeLut);
   LogicalResult printOperation(openfhe::EvalFuncOp evalFunc);
};

}  // namespace mlir::heir::openfhe

#endif  // LIB_TARGET_OPENFHEBIN_OPENFHEBINEMITTER_H_
