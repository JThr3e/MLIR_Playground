//===- StandalonePasses.cpp - Standalone passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Operation.h"
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
#define GEN_PASS_DEF_DEADCODEELIM
#include "Standalone/StandalonePasses.h.inc"

/*
 *
 * Use this to test me:
 * ./build/bin/standalone-opt --test-dce mlir_files/sample_code.mlir
 *
 */

namespace {

class DeadCodeElim
    : public impl::DeadCodeElimBase<DeadCodeElim> {
public:
  using impl::DeadCodeElimBase<
      DeadCodeElim>::DeadCodeElimBase;
  void runOnOperation() final {
      auto mod = getOperation();
      if(mod.getName().has_value()){
        std::cout << "Running dce on module: " << mod.getName().value().str() << std::endl; 
      }else {
        std::cout << "Running DCE on no name module" << std::endl;
      }


      std::vector<Operation *> visited_ops;
      std::vector<Operation *> un_visited_ops;
      mod.walk([&](mlir::func::FuncOp func){
        std::cout << func.getName().str() << std::endl;
        visited_ops.push_back(func.getOperation());
        mlir::Operation &last_op = func.getBody().front().back();
        if(mlir::func::ReturnOp ret_op = mlir::dyn_cast<mlir::func::ReturnOp>(last_op)){
            std::function<void(Operation *)> bfs_rec = [&](Operation * op){
                visited_ops.push_back(op);
                auto vals = op->getOpOperands();
                for(auto &val : vals){
                    auto owner_op = val.get().getDefiningOp();
                    if(owner_op){
                        bfs_rec(owner_op);
                    }
                }
            };
            bfs_rec(ret_op);
            func.walk([&](mlir::Operation * op){
                if(std::find(visited_ops.begin(), visited_ops.end(), op) == visited_ops.end()){
                    un_visited_ops.push_back(op);
                    std::cerr << "marking for deletion! ";
                    op->dump();
                }
            });
            std::reverse(un_visited_ops.begin(), un_visited_ops.end());
            for(auto op : un_visited_ops){
                op->erase();
            }
        }
        std::cerr << "======================" << std::endl;

      });
  }
};

} // namespace
} // namespace mlir::standalone
