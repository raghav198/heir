module {
    func.func @add_ints(%a: !secret.secret<i8>, %b: !secret.secret<i8>) -> (!secret.secret<i8>) {
        %1 = secret.generic ins(%a, %b: !secret.secret<i8>, !secret.secret<i8>) {
            ^bb0(%A: i8, %B: i8) :
                %2 = arith.addi %A, %B: i8
                secret.yield %2: i8
        } -> (!secret.secret<i8>)
        return %1 : !secret.secret<i8>
    }
}