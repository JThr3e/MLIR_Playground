//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandalonePasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

//struct MyPipelineOptions : public mlir::PassPipelineOptions {
//};

void registerPipelineThread() {
    mlir::PassPipelineRegistration<>(
        "test-multithread", "Test out mlir processing func ops in a multithreaded fashion",
        [](mlir::OpPassManager &pm) { 
            mlir::OpPassManager &nestedFunctionPM = pm.nest<mlir::func::FuncOp>();
            nestedFunctionPM.addPass(mlir::standalone::createStandaloneFuncWait());
        });
}


int main(int argc, char **argv) {
  //mlir::registerAllPasses();
  mlir::standalone::registerPasses();

  registerPipelineThread();

  mlir::DialectRegistry registry;
  registry.insert<mlir::standalone::StandaloneDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
