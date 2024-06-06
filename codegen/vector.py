from codegen.ssa import *
from codegen.straightforward import innermost

def innermost_256(activation: Ref, kernel: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, iK: Expr | Ref, cntp1: Ref, cntp2: Ref) -> Block:
    p1actidx = Expr(Expr(iM, Op.times, K), Op.plus, Expr(iK, Op.plus, zero))
    p2actidx = Expr(Expr(iM, Op.times, K), Op.plus, Expr(iK, Op.plus, four))
    p1keridx = Expr(Expr(iN, Op.times, K), Op.plus, Expr(iK, Op.plus, zero))
    p2keridx = Expr(Expr(iN, Op.times, K), Op.plus, Expr(iK, Op.plus, four))
    a1 = Load(typ=Type.m256i, src=MRef(AVXLoadKind.load_si256, activation, p1actidx))
    k1 = Load(typ=Type.m256i, src=MRef(AVXLoadKind.load_si256, kernel, p1keridx))
    a2 = Load(typ=Type.m256i, src=MRef(AVXLoadKind.load_si256, activation, p2actidx))
    k2 = Load(typ=Type.m256i, src=MRef(AVXLoadKind.load_si256, kernel, p2keridx))

    xor1 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_xor_si256", args=[a1.ref, k1.ref]))
    and1 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_and_si256", args=[a1.ref, k1.ref]))
    xor2 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_xor_si256", args=[a2.ref, k2.ref]))
    and2 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_and_si256", args=[a2.ref, k2.ref]))

    p1 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_unpacklo_epi64", args=[xor1.ref, xor2.ref]))
    p2 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_unpackhi_epi64", args=[and1.ref, and2.ref]))
    p1andp2 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_and_si256", args=[p1.ref, p2.ref]))

    popcntp2 = Compute(typ=Type.i32, expr=CallExpr(fn="popcnt", args=[RefExpr(p2.ref), CallExpr(fn="sizeof", args=[p1.ref])]))
    popcntp1p2 = Compute(typ=Type.i32, expr=CallExpr(fn="popcnt", args=[RefExpr(p1andp2.ref), CallExpr(fn="sizeof", args=[p1andp2.ref])]))
    sumpopcntp2 = Compute(Type.i32, Expr(cntp1, Op.plus, popcntp2.ref))
    sumpopcntp1p2 = Compute(Type.i32, Expr(cntp2, Op.plus, popcntp1p2.ref))

    return Block([
        a1,
        k1,
        a2,
        k2,
        xor1,
        xor2,
        and1,
        and2,
        p1,
        p2,
        popcntp2,
        p1andp2,
        popcntp1p2,
        sumpopcntp2,
        sumpopcntp1p2,
        Store(dst=cntp1, src=sumpopcntp2.ref),
        Store(dst=cntp2, src=sumpopcntp1p2.ref),
    ])

def gemm_kernel_256(activation: Ref, kernel: Ref, output: Ref, N: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, alpha: Ref) -> Block:
    iK = Load(typ=Type.size, src=zero, dst=VarRef("iK"))
    cntp1 = Load(typ=Type.i32, src=zero)
    cntp2 = Load(typ=Type.i32, src=zero)

    inner = innermost_256(activation, kernel, K, iM, iN, iK.ref, cntp1.ref, cntp2.ref)

    subcntp1cntp2 = Compute(Type.i64, Expr(cntp1.ref, Op.minus, cntp2.ref))
    curr = Compute(Type.i64, Expr(subcntp1cntp2.ref, Op.minus, cntp2.ref))
    outidx = Expr(Expr(iM, Op.times, N), Op.plus, iN)
    value = Compute(Type.f32, TernaryExpr(Expr(curr.ref, Op.gt, zero), curr.ref, Expr(curr.ref, Op.times, alpha)))

    return Block([
        iK,
        cntp1,
        cntp2,
        Loop(
            i=iK.ref,
            lower=None,
            check=Cast(Type.i32, iK.ref),
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, Expr(four, Op.times, BITS))),
            op=Op.lte,
            stride=Expr(four, Op.times, BITS),
            block=inner
        ),
        # cleanup loop
        Loop(
            i=iK.ref,
            lower=None,
            upper=K,
            stride=BITS,
            block=innermost(activation, kernel, K, iM, iN, iK.ref, cntp1.ref, cntp2.ref)
        ),
        subcntp1cntp2,
        curr,
        value,
        Store(dst=ArrayRef(output, outidx), src=value.ref)
    ])

def gemm_kernel_macro_256() -> Macro:
    activation = VarRef("activation")
    kernel = VarRef("kernel")
    output = VarRef("output")
    N = VarRef("N")
    K = VarRef("K")
    iM = VarRef("iM")
    iN = VarRef("iN")
    alpha = VarRef("alpha")
    computation = gemm_kernel_256(activation, kernel, output, N, K, iM, iN, alpha)
    return Macro(name="gemm_kernel", args=[activation, kernel, output, N, K, BITS, iM, iN, alpha], computation=computation)

def innermost_512(activation: Ref, kernel: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, iK: Expr | Ref, cntp1: Ref, cntp2: Ref) -> Block:
    p1actidx = Expr(Expr(iM, Op.times, K), Op.plus, Expr(iK, Op.plus, zero))
    p2actidx = Expr(Expr(iM, Op.times, K), Op.plus, Expr(iK, Op.plus, eight))
    p1keridx = Expr(Expr(iN, Op.times, K), Op.plus, Expr(iK, Op.plus, zero))
    p2keridx = Expr(Expr(iN, Op.times, K), Op.plus, Expr(iK, Op.plus, eight))
    a1 = Load(typ=Type.m512i, src=MRef(AVXLoadKind.load_epi64, activation, p1actidx))
    k1 = Load(typ=Type.m512i, src=MRef(AVXLoadKind.load_epi64, kernel, p1keridx))
    a2 = Load(typ=Type.m512i, src=MRef(AVXLoadKind.load_epi64, activation, p2actidx))
    k2 = Load(typ=Type.m512i, src=MRef(AVXLoadKind.load_epi64, kernel, p2keridx))

    xor1 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_xor_epi64", args=[a1.ref, k1.ref]))
    and1 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_and_epi64", args=[a1.ref, k1.ref]))
    xor2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_xor_epi64", args=[a2.ref, k2.ref]))
    and2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_and_epi64", args=[a2.ref, k2.ref]))

    p1 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_unpacklo_epi64", args=[xor1.ref, xor2.ref]))
    p2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_unpackhi_epi64", args=[and1.ref, and2.ref]))
    p1andp2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_and_epi64", args=[p1.ref, p2.ref]))

    popcntp2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_popcnt_epi64", args=[p2.ref]))
    popcntp1p2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_popcnt_epi64", args=[p1andp2.ref]))
    sumpopcntp2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_add_epi64", args=[cntp1, popcntp2.ref]))
    sumpopcntp1p2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_add_epi64", args=[cntp2, popcntp1p2.ref]))

    return Block([
        a1,
        k1,
        a2,
        k2,
        xor1,
        xor2,
        and1,
        and2,
        p1,
        p2,
        popcntp2,
        p1andp2,
        popcntp1p2,
        sumpopcntp2,
        sumpopcntp1p2,
        Store(dst=cntp1, src=sumpopcntp2.ref),
        Store(dst=cntp2, src=sumpopcntp1p2.ref),
    ])

def innermost_512_libpopcnt(activation: Ref, kernel: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, iK: Expr | Ref, cntp1v: Ref, cntp2v: Ref) -> Block:
    p1actidx = Expr(Expr(iM, Op.times, K), Op.plus, Expr(iK, Op.plus, zero))
    p2actidx = Expr(Expr(iM, Op.times, K), Op.plus, Expr(iK, Op.plus, eight))
    p1keridx = Expr(Expr(iN, Op.times, K), Op.plus, Expr(iK, Op.plus, zero))
    p2keridx = Expr(Expr(iN, Op.times, K), Op.plus, Expr(iK, Op.plus, eight))
    a1 = Load(typ=Type.m512i, src=MRef(AVXLoadKind.load_epi64, activation, p1actidx))
    k1 = Load(typ=Type.m512i, src=MRef(AVXLoadKind.load_epi64, kernel, p1keridx))
    a2 = Load(typ=Type.m512i, src=MRef(AVXLoadKind.load_epi64, activation, p2actidx))
    k2 = Load(typ=Type.m512i, src=MRef(AVXLoadKind.load_epi64, kernel, p2keridx))

    xor1 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_xor_epi64", args=[a1.ref, k1.ref]))
    and1 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_and_epi64", args=[a1.ref, k1.ref]))
    xor2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_xor_epi64", args=[a2.ref, k2.ref]))
    and2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_and_epi64", args=[a2.ref, k2.ref]))

    p1 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_unpacklo_epi64", args=[xor1.ref, xor2.ref]))
    p2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_unpackhi_epi64", args=[and1.ref, and2.ref]))
    p1andp2 = Compute(typ=Type.m512i, expr=CallExpr(fn="_mm512_and_si512", args=[p1.ref, p2.ref]))

    cntp1vidx = Expr(iK, Op.div, Expr(eight, Op.times, BITS))
    cntp2vidx = Expr(iK, Op.div, Expr(eight, Op.times, BITS))

    return Block([
        a1,
        k1,
        a2,
        k2,
        xor1,
        xor2,
        and1,
        and2,
        p1,
        p2,
        p1andp2,
        Store(dst=ArrayRef(arr=cntp1v, idx=cntp1vidx), src=p2.ref),
        Store(dst=ArrayRef(arr=cntp2v, idx=cntp2vidx), src=p1andp2.ref),
    ])

def gemm_kernel_512(activation: Ref, kernel: Ref, output: Ref, N: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, alpha: Ref) -> Block:
    iK = Load(typ=Type.size, src=zero, dst=VarRef("iK"))
    cntp1v = Load(typ=Type.m512i, src=CallExpr(fn="_mm512_setzero_si512", args=[]))
    cntp2v = Load(typ=Type.m512i, src=CallExpr(fn="_mm512_setzero_si512", args=[]))

    inner = innermost_512(activation, kernel, K, iM, iN, iK.ref, cntp1v.ref, cntp2v.ref)

    cntp1 = Compute(typ=Type.i32, expr=CallExpr(fn="_mm512_reduce_add_epi64", args=[cntp1v.ref]))
    cntp2 = Compute(typ=Type.i32, expr=CallExpr(fn="_mm512_reduce_add_epi64", args=[cntp2v.ref]))

    subcntp1cntp2 = Compute(Type.i64, Expr(cntp1.ref, Op.minus, cntp2.ref))
    curr = Compute(Type.i64, Expr(subcntp1cntp2.ref, Op.minus, cntp2.ref))
    outidx = Expr(Expr(iM, Op.times, N), Op.plus, iN)
    value = Compute(Type.f32, TernaryExpr(Expr(curr.ref, Op.gt, zero), curr.ref, Expr(curr.ref, Op.times, alpha)))

    return Block([
        iK,
        cntp1v,
        cntp2v,
        Loop(
            i=iK.ref,
            lower=None,
            check=Cast(Type.i32, iK.ref),
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, Expr(eight, Op.times, BITS))),
            op=Op.lte,
            stride=Expr(eight, Op.times, BITS),
            block=inner
        ),
        cntp1,
        cntp2,
        # cleanup loop
        Loop(
            i=iK.ref,
            lower=None,
            upper=K,
            stride=BITS,
            block=innermost(activation, kernel, K, iM, iN, iK.ref, cntp1.ref, cntp2.ref)
        ),
        subcntp1cntp2,
        curr,
        value,
        Store(dst=ArrayRef(output, outidx), src=value.ref)
    ])

def gemm_kernel_macro_512() -> Macro:
    activation = VarRef("activation")
    kernel = VarRef("kernel")
    output = VarRef("output")
    N = VarRef("N")
    K = VarRef("K")
    iM = VarRef("iM")
    iN = VarRef("iN")
    alpha = VarRef("alpha")
    computation = gemm_kernel_512(activation, kernel, output, N, K, iM, iN, alpha)
    return Macro(name="gemm_kernel", args=[activation, kernel, output, N, K, BITS, iM, iN, alpha], computation=computation)

def innermost_256_libpopcnt(activation: Ref, kernel: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, iK: Expr | Ref, cntp1v: Ref, cntp2v: Ref) -> Block:
    p1actidx = Expr(Expr(iM, Op.times, K), Op.plus, Expr(iK, Op.plus, zero))
    p2actidx = Expr(Expr(iM, Op.times, K), Op.plus, Expr(iK, Op.plus, four))
    p1keridx = Expr(Expr(iN, Op.times, K), Op.plus, Expr(iK, Op.plus, zero))
    p2keridx = Expr(Expr(iN, Op.times, K), Op.plus, Expr(iK, Op.plus, four))
    a1 = Load(typ=Type.m256i, src=MRef(AVXLoadKind.load_si256, activation, p1actidx))
    k1 = Load(typ=Type.m256i, src=MRef(AVXLoadKind.load_si256, kernel, p1keridx))
    a2 = Load(typ=Type.m256i, src=MRef(AVXLoadKind.load_si256, activation, p2actidx))
    k2 = Load(typ=Type.m256i, src=MRef(AVXLoadKind.load_si256, kernel, p2keridx))

    xor1 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_xor_si256", args=[a1.ref, k1.ref]))
    and1 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_and_si256", args=[a1.ref, k1.ref]))
    xor2 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_xor_si256", args=[a2.ref, k2.ref]))
    and2 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_and_si256", args=[a2.ref, k2.ref]))

    p1 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_unpacklo_epi64", args=[xor1.ref, xor2.ref]))
    p2 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_unpackhi_epi64", args=[and1.ref, and2.ref]))
    p1andp2 = Compute(typ=Type.m256i, expr=CallExpr(fn="_mm256_and_si256", args=[p1.ref, p2.ref]))

    cntp1vidx = Expr(iK, Op.div, Expr(four, Op.times, BITS))
    cntp2vidx = Expr(iK, Op.div, Expr(four, Op.times, BITS))

    return Block([
        a1,
        k1,
        a2,
        k2,
        xor1,
        xor2,
        and1,
        and2,
        p1,
        p2,
        p1andp2,
        Store(dst=ArrayRef(arr=cntp1v, idx=cntp1vidx), src=p2.ref),
        Store(dst=ArrayRef(arr=cntp2v, idx=cntp2vidx), src=p1andp2.ref),
    ])

def gemm_kernel_256_libpopcnt(activation: Ref, kernel: Ref, output: Ref, N: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, alpha: Ref) -> Block:
    iK = Load(typ=Type.size, src=zero, dst=VarRef("iK"))
    cntp1v = ArrayDeclr(typ=Type.m256i, size=Expr(K, Op.div, Expr(four, Op.times, BITS)))
    cntp2v = ArrayDeclr(typ=Type.m256i, size=Expr(K, Op.div, Expr(four, Op.times, BITS)))

    inner = innermost_256_libpopcnt(activation, kernel, K, iM, iN, iK.ref, cntp1v.ref, cntp2v.ref)

    cntp1 = Compute(Type.i32, CallExpr(fn="popcnt", args=[cntp1v.ref, CallExpr("sizeof", [cntp1v.ref])]))
    cntp2 = Compute(Type.i32, CallExpr(fn="popcnt", args=[cntp2v.ref, CallExpr("sizeof", [cntp2v.ref])]))
    subcntp1cntp2 = Compute(Type.i64, Expr(cntp1.ref, Op.minus, cntp2.ref))
    curr = Compute(Type.i64, Expr(subcntp1cntp2.ref, Op.minus, cntp2.ref))
    outidx = Expr(Expr(iM, Op.times, N), Op.plus, iN)
    value = Compute(Type.f32, TernaryExpr(Expr(curr.ref, Op.gt, zero), curr.ref, Expr(curr.ref, Op.times, alpha)))

    return Block([
        iK,
        cntp1v,
        cntp2v,
        Loop(
            i=iK.ref,
            lower=None,
            check=Cast(Type.i32, iK.ref),
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, Expr(four, Op.times, BITS))),
            op=Op.lte,
            stride=Expr(four, Op.times, BITS),
            block=inner
        ),
        cntp1,
        cntp2,
        # cleanup loop
        Loop(
            i=iK.ref,
            lower=None,
            upper=K,
            stride=BITS,
            block=innermost(activation, kernel, K, iM, iN, iK.ref, cntp1.ref, cntp2.ref)
        ),
        subcntp1cntp2,
        curr,
        value,
        Store(dst=ArrayRef(output, outidx), src=value.ref)
    ])

def gemm_kernel_macro_256_libpopcnt() -> Macro:
    activation = VarRef("activation")
    kernel = VarRef("kernel")
    output = VarRef("output")
    N = VarRef("N")
    K = VarRef("K")
    iM = VarRef("iM")
    iN = VarRef("iN")
    alpha = VarRef("alpha")
    computation = gemm_kernel_256_libpopcnt(activation, kernel, output, N, K, iM, iN, alpha)
    return Macro(name="gemm_kernel", args=[activation, kernel, output, N, K, BITS, iM, iN, alpha], computation=computation)
