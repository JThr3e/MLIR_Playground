module {
    func.func @foo() -> (i32){
        %0 = arith.constant 1 : i32
        %1 = arith.constant 2 : i32
        %2 = arith.addi %0, %1 : i32
        return %2 : i32
    }
    func.func @foo1() {return}
    func.func @foo2() {return}
    func.func @foo3() {return}
    func.func @foo4() {return}
    func.func @foo5() {return}
    func.func @foo6() {return}
    func.func @foo7() {return}
    func.func @foo8() {return}
    func.func @foo9() {return}
    func.func @foo10() {return}
    func.func @foo11() {return}
    func.func @foo12() {return}

}
