//===- StandalonePasses.cpp - Standalone passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Standalone/StandalonePasses.h"

#include <iostream>

namespace mlir::standalone {
#define GEN_PASS_DEF_STANDALONEEMPTY
#include "Standalone/StandalonePasses.h.inc"

namespace {

class StandaloneEmpty
    : public impl::StandaloneEmptyBase<StandaloneEmpty> {
public:
  using impl::StandaloneEmptyBase<
      StandaloneEmpty>::StandaloneEmptyBase;
  void runOnOperation() final {
      auto module = getOperation();
      module.walk([&](mlir::Operation *op) {
            std::cout << "Dumping Operation: " << std::endl;
            op->dump();
      });
  }
};

} // namespace
} // namespace mlir::standalone
