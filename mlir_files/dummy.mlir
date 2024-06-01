module {
    func.func @bar() {
        %0 = arith.constant 1 : i32
        %res = standalone.foo %0 : i32
        return
    }

    func.func @standalone_types(%arg0: !standalone.custom<"10">) {
        return
    }
}
