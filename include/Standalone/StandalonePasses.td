//===- StandalonePsss.td - Standalone dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_PASS
#define STANDALONE_PASS

include "mlir/Pass/PassBase.td"

def StandaloneSwitchBarFoo: Pass<"standalone-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

def StandaloneEmpty: Pass<"standalone-empty", "::mlir::ModuleOp"> {
  let summary = "Does nothing";
  let description = [{
  }];
}

def StandaloneFuncWait: Pass<"func-wait", "::mlir::func::FuncOp"> {
  let summary = "waits on a function";
  let description = [{
  }];
}

#endif // STANDALONE_PASS
