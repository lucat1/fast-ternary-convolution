from codegen.ssa import *

def innermost(activation: Ref, kernel: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, iK: Expr | Ref, cntp1: Ref, cntp2: Ref) -> Block:
    p1actidx = Expr(Expr(iM, Op.times, K), Op.plus, Expr(iK, Op.plus, zero))
    p2actidx = Expr(Expr(iM, Op.times, K), Op.plus, Expr(iK, Op.plus, one))
    p1keridx = Expr(Expr(iN, Op.times, K), Op.plus, Expr(iK, Op.plus, zero))
    p2keridx = Expr(Expr(iN, Op.times, K), Op.plus, Expr(iK, Op.plus, one))
    p1e1 = Load(typ=Type.i64, src=ArrayRef(activation, p1actidx))
    p1e2 = Load(typ=Type.i64, src=ArrayRef(kernel, p1keridx))
    p2e1 = Load(typ=Type.i64, src=ArrayRef(activation, p2actidx))
    p2e2 = Load(typ=Type.i64, src=ArrayRef(kernel, p2keridx))
    p1 = Compute(Type.i64, Expr(p1e1.ref, Op.bxor, p1e2.ref))
    p2 = Compute(Type.i64, Expr(p2e1.ref, Op.band, p2e2.ref))
    popcntp2 = Compute(Type.i32, CallExpr("popcnt64", [p2.ref]))
    p1andp2 = Compute(Type.i64, Expr(p1.ref, Op.band, p2.ref))
    popcntp1p2 = Compute(Type.i32, CallExpr("popcnt64", [p1andp2.ref]))
    sumpopcntp1 = Compute(Type.i32, Expr(cntp1, Op.plus, popcntp2.ref))
    sumpopcntp2 = Compute(Type.i32, Expr(cntp2, Op.plus, popcntp1p2.ref))

    return Block([
        p1e1,
        p1e2,
        p2e1,
        p2e2,
        p1,
        p2,
        popcntp2,
        p1andp2,
        popcntp1p2,
        sumpopcntp1,
        sumpopcntp2,
        Store(dst=cntp1, src=sumpopcntp1.ref),
        Store(dst=cntp2, src=sumpopcntp2.ref),
    ])

def gemm_kernel(activation: Ref, kernel: Ref, output: Ref, N: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, alpha: Ref) -> Block:
    iK = VarRef("iK")
    cntp1 = Load(typ=Type.i32, src=zero)
    cntp2 = Load(typ=Type.i32, src=zero)

    inner = innermost(activation, kernel, K, iM, iN, iK, cntp1.ref, cntp2.ref)

    subcntp1cntp2 = Compute(Type.i64, Expr(cntp1.ref, Op.minus, cntp2.ref))
    curr = Compute(Type.i64, Expr(subcntp1cntp2.ref, Op.minus, cntp2.ref))
    outidx = Expr(Expr(iM, Op.times, N), Op.plus, iN)
    value = Compute(Type.f32, TernaryExpr(Expr(curr.ref, Op.gt, zero), curr.ref, Expr(curr.ref, Op.times, alpha)))

    return Block([
        cntp1,
        cntp2,
        Loop(
            i=iK,
            lower=zero,
            upper=K,
            stride=BITS,
            block=inner
        ),
        subcntp1cntp2,
        curr,
        value,
        Store(
            dst=ArrayRef(output, outidx),
            src=value.ref
        )
    ])

def gemm_kernel_macro() -> Macro:
    activation = VarRef("activation")
    kernel = VarRef("kernel")
    output = VarRef("output")
    N = VarRef("N")
    K = VarRef("K")
    iM = VarRef("iM")
    iN = VarRef("iN")
    alpha = VarRef("alpha")
    computation = gemm_kernel(activation, kernel, output, N, K, iM, iN, alpha)
    return Macro(name="gemm_kernel", args=[activation, kernel, output, N, K, BITS, iM, iN, alpha], computation=computation)
