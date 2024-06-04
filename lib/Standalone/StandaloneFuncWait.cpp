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
#include <chrono>
#include <thread>

using namespace std::this_thread;     // sleep_for, sleep_until
using namespace std::chrono_literals; // ns, us, ms, s, h, etc.
using std::chrono::system_clock;

namespace mlir::standalone {
#define GEN_PASS_DEF_STANDALONEFUNCWAIT
#include "Standalone/StandalonePasses.h.inc"

namespace {

class StandaloneFuncWait
    : public impl::StandaloneFuncWaitBase<StandaloneFuncWait> {
public:
  using impl::StandaloneFuncWaitBase<
      StandaloneFuncWait>::StandaloneFuncWaitBase;
  void runOnOperation() final {
      auto func = getOperation();
      std::cout << "Running func wait pass on func: " << func.getName().str() << std::endl; 
      sleep_for(2000ms);
      std::cout << "Finished on func: " << func.getName().str() << std::endl;
  }
};

} // namespace
} // namespace mlir::standalone
