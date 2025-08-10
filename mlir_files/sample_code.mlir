module {
    func.func @foo(%arg0 : i32, %arg1: i32) -> (i32){
        %0 = arith.constant 1 : i32
        %1 = arith.constant 2 : i32
        %2 = arith.addi %0, %1 : i32
        %3 = arith.constant 5 : i32
        %4 = arith.addi %arg0, %arg1 : i32
        %5 = arith.addi %4, %2 : i32
        %6 = arith.subi %3, %arg0 : i32
        %7 = arith.addi %4, %3 : i32
        %8 = arith.subi %5, %2 : i32
        return %8 : i32
    }
}
