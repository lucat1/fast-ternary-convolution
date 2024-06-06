from codegen.ssa import *
from codegen.straightforward import innermost
from codegen.vector import innermost_256, innermost_256_libpopcnt, innermost_512, innermost_512_libpopcnt
from codegen.unrolled import addall

def gemm_kernel_256(activation: Ref, kernel: Ref, output: Ref, N: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, alpha: Ref, uK: int, uK_cleanup: int) -> Block:
    iK = Load(typ=Type.size, src=zero, dst=VarRef("iK"))

    def off(i: int):
        return Expr(Literal(f"{i}"), Op.times, Expr(four, Op.times, BITS))

    def loop_avx(i: int):
        cntp1 = Load(typ=Type.i32, src=zero)
        cntp2 = Load(typ=Type.i32, src=zero)
        comp = innermost_256(activation, kernel, K, iM, iN, Expr(iK.ref, Op.plus, off(i)), cntp1.ref, cntp2.ref)
        return ([cntp1, cntp2], comp)
    _pre_avx, _compute_avx = list(zip(*[loop_avx(i) for i in range(uK)]))
    pre_avx: List[Computation] = [instr for instrs in _pre_avx for instr in instrs]
    compute_avx = [instr for block in _compute_avx for instr in block.instrs]

    def loop(i: int):
        cntp1 = _pre_avx[i][0]
        cntp2 = _pre_avx[i][1]
        comp = innermost(activation, kernel, K, iM, iN, Expr(iK.ref, Op.plus, Expr(Literal(f"{i}"), Op.times, BITS)), cntp1.ref, cntp2.ref)
        return comp
    compute = [instr for i in range(uK_cleanup) for instr in loop(i).instrs]

    cntp1s: List[Compute] = [f for [f, _] in _pre_avx]
    cntp2s: List[Compute] = [f for [_, f] in _pre_avx]
    cntp1, cntp1f = addall(cntp1s)
    cntp2, cntp2f = addall(cntp2s)
    subcntp1cntp2 = Compute(Type.i64, Expr(cntp1.ref, Op.minus, cntp2.ref))
    curr = Compute(Type.i64, Expr(subcntp1cntp2.ref, Op.minus, cntp2.ref))
    outidx = Expr(Expr(iM, Op.times, N), Op.plus, iN)
    value = Compute(Type.f32, TernaryExpr(Expr(curr.ref, Op.gt, zero), curr.ref, Expr(curr.ref, Op.times, alpha)))

    return Block([
        *pre_avx,
        iK,
        Loop(
            i=iK.ref,
            lower=None,
            check=Cast(Type.i32, iK.ref),
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, off(uK))),
            op=Op.lte,
            stride=off(uK),
            block=Block(compute_avx)
        ),
        # unrolled cleanup loop
        Loop(
            i=iK.ref,
            lower=None,
            check=Cast(Type.i32, iK.ref),
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, Expr(Literal(f"{uK_cleanup}"), Op.times, BITS))),
            op=Op.lte,
            stride=Expr(Literal(f"{uK_cleanup}"), Op.times, BITS),
            block=Block(compute)
        ),
        *cntp1f,
        *cntp2f,
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

def gemm_kernel_macro_256(uK: int, uK_cleanup: int) -> Macro:
    assert uK_cleanup <= uK
    activation = VarRef("activation")
    kernel = VarRef("kernel")
    output = VarRef("output")
    N = VarRef("N")
    K = VarRef("K")
    iM = VarRef("iM")
    iN = VarRef("iN")
    alpha = VarRef("alpha")
    computation = gemm_kernel_256(activation, kernel, output, N, K, iM, iN, alpha, uK, uK_cleanup)
    return Macro(name="gemm_kernel", args=[activation, kernel, output, N, K, BITS, iM, iN, alpha], computation=computation)

def gemm_kernel_512(activation: Ref, kernel: Ref, output: Ref, N: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, alpha: Ref, uK: int, uK_cleanup: int) -> Block:
    iK = Load(typ=Type.size, src=zero, dst=VarRef("iK"))

    def off(i: int):
        return Expr(Literal(f"{i}"), Op.times, Expr(eight, Op.times, BITS))

    def loop_avx(i: int):
        cntp1 = Load(typ=Type.m512i, src=CallExpr(fn="_mm512_setzero_si512", args=[]))
        cntp2 = Load(typ=Type.m512i, src=CallExpr(fn="_mm512_setzero_si512", args=[]))
        comp = innermost_512(activation, kernel, K, iM, iN, Expr(iK.ref, Op.plus, off(i)), cntp1.ref, cntp2.ref)
        return ([cntp1, cntp2], comp)
    _pre_avx, _compute_avx = list(zip(*[loop_avx(i) for i in range(uK)]))
    prev: List[Computation] = [instr for instrs in _pre_avx for instr in instrs]
    computev = [instr for block in _compute_avx for instr in block.instrs]

    cntp1sv: List[Compute] = [f for [f, _] in _pre_avx]
    cntp2sv: List[Compute] = [f for [_, f] in _pre_avx]
    def addv(e1: Ref | Expr, e2: Ref | Expr):
        return CallExpr(fn="_mm512_add_epi64", args=[e1, e2])
    cntp1v, cntp1fv = addall(cntp1sv, typ=Type.m512i, add_expr=addv)
    cntp2v, cntp2fv = addall(cntp2sv, typ=Type.m512i, add_expr=addv)
    cntp1vs = Compute(typ=Type.i32, expr=CallExpr(fn="_mm512_reduce_add_epi64", args=[cntp1v.ref]))
    cntp2vs = Compute(typ=Type.i32, expr=CallExpr(fn="_mm512_reduce_add_epi64", args=[cntp2v.ref]))

    def loop(i: int):
        cntp1 = Load(typ=Type.i32, src=zero)
        cntp2 = Load(typ=Type.i32, src=zero)
        comp = innermost(activation, kernel, K, iM, iN, Expr(iK.ref, Op.plus, Expr(Literal(f"{i}"), Op.times, BITS)), cntp1.ref, cntp2.ref)
        return ([cntp1, cntp2], comp)
    _pre, _compute = list(zip(*[loop(i) for i in range(uK)]))
    pres: List[Computation] = [instr for instrs in _pre for instr in instrs]
    computes = [instr for block in _compute for instr in block.instrs]

    cntp1ss: List[Compute] = [f for [f, _] in _pre]
    cntp2ss: List[Compute] = [f for [_, f] in _pre]
    cntp1s, cntp1fs = addall(cntp1ss)
    cntp2s, cntp2fs = addall(cntp2ss)

    cntp1 = Compute(typ=Type.i32, expr=Expr(cntp1vs.ref, Op.plus, cntp1s.ref))
    cntp2 = Compute(typ=Type.i32, expr=Expr(cntp2vs.ref, Op.plus, cntp2s.ref))

    subcntp1cntp2 = Compute(Type.i64, Expr(cntp1.ref, Op.minus, cntp2.ref))
    curr = Compute(Type.i64, Expr(subcntp1cntp2.ref, Op.minus, cntp2.ref))
    outidx = Expr(Expr(iM, Op.times, N), Op.plus, iN)
    value = Compute(Type.f32, TernaryExpr(Expr(curr.ref, Op.gt, zero), curr.ref, Expr(curr.ref, Op.times, alpha)))

    return Block([
        *prev,
        *pres,
        iK,
        Loop(
            i=iK.ref,
            lower=None,
            check=Cast(Type.i32, iK.ref),
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, off(uK))),
            op=Op.lte,
            stride=off(uK),
            block=Block(computev)
        ),
        # unrolled cleanup loop
        Loop(
            i=iK.ref,
            lower=None,
            check=Cast(Type.i32, iK.ref),
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, Expr(Literal(f"{uK_cleanup}"), Op.times, BITS))),
            stride=Expr(Literal(f"{uK_cleanup}"), Op.times, BITS),
            block=Block(computes)
        ),
        *cntp1fv,
        *cntp2fv,
        *cntp1fs,
        *cntp2fs,
        cntp1vs,
        cntp2vs,
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

def gemm_kernel_macro_512(uK: int, uK_cleanup: int) -> Macro:
    activation = VarRef("activation")
    kernel = VarRef("kernel")
    output = VarRef("output")
    N = VarRef("N")
    K = VarRef("K")
    iM = VarRef("iM")
    iN = VarRef("iN")
    alpha = VarRef("alpha")
    computation = gemm_kernel_512(activation, kernel, output, N, K, iM, iN, alpha, uK, uK_cleanup)
    return Macro(name="gemm_kernel", args=[activation, kernel, output, N, K, BITS, iM, iN, alpha], computation=computation)

def gemm_kernel_256_libpopcnt(activation: Ref, kernel: Ref, output: Ref, N: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, alpha: Ref, uK: int, uK_cleanup: int) -> Block:
    iK = Load(typ=Type.size, src=zero, dst=VarRef("iK"))
    uKl = Literal(f"{uK}")
    aux_size = Expr(Expr(K, Op.div, Expr(Expr(four, Op.times, BITS), Op.times, uKl)), Op.times, uKl)
    cntp1v = ArrayDeclr(typ=Type.m256i, size=aux_size)
    cntp2v = ArrayDeclr(typ=Type.m256i, size=aux_size)

    def off(i: int):
        return Expr(Literal(f"{i}"), Op.times, Expr(four, Op.times, BITS))

    def loop_avx(i: int):
        return innermost_256_libpopcnt(activation, kernel, K, iM, iN, Expr(iK.ref, Op.plus, off(i)), cntp1v.ref, cntp2v.ref)
    compute_avx = [instr for i in range(uK) for instr in loop_avx(i).instrs]

    def loop(i: int):
        cntp1 = Load(typ=Type.i32, src=zero)
        cntp2 = Load(typ=Type.i32, src=zero)
        comp = innermost(activation, kernel, K, iM, iN, Expr(iK.ref, Op.plus, Expr(Literal(f"{i}"), Op.times, BITS)), cntp1.ref, cntp2.ref)
        return ([cntp1, cntp2], comp)
    _pre, _compute = list(zip(*[loop(i) for i in range(uK)]))
    pre: List[Computation] = [instr for instrs in _pre for instr in instrs]
    compute = [instr for block in _compute for instr in block.instrs]

    cntp1s: List[Compute] = [f for [f, _] in _pre]
    cntp2s: List[Compute] = [f for [_, f] in _pre]
    cntp1vf = Compute(Type.i32, CallExpr(fn="popcnt", args=[cntp1v.ref, CallExpr("sizeof", [cntp1v.ref])]))
    cntp2vf = Compute(Type.i32, CallExpr(fn="popcnt", args=[cntp2v.ref, CallExpr("sizeof", [cntp2v.ref])]))
    cntp1, cntp1f = addall(cntp1s + [cntp1vf])
    cntp2, cntp2f = addall(cntp2s + [cntp2vf])
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
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, off(uK))),
            op=Op.lte,
            stride=off(uK),
            block=Block(compute_avx)
        ),
        cntp1vf,
        cntp2vf,
        *pre,
        # unrolled cleanup loop
        Loop(
            i=iK.ref,
            lower=None,
            check=Cast(Type.i32, iK.ref),
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, Expr(Literal(f"{uK_cleanup}"), Op.times, BITS))),
            op=Op.lte,
            stride=Expr(Literal(f"{uK_cleanup}"), Op.times, BITS),
            block=Block(compute)
        ),
        *cntp1f,
        *cntp2f,
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

def gemm_kernel_macro_256_libpopcnt(uK: int, uK_cleanup: int) -> Macro:
    assert uK_cleanup <= uK
    activation = VarRef("activation")
    kernel = VarRef("kernel")
    output = VarRef("output")
    N = VarRef("N")
    K = VarRef("K")
    iM = VarRef("iM")
    iN = VarRef("iN")
    alpha = VarRef("alpha")
    computation = gemm_kernel_256_libpopcnt(activation, kernel, output, N, K, iM, iN, alpha, uK, uK_cleanup)
    return Macro(name="gemm_kernel", args=[activation, kernel, output, N, K, BITS, iM, iN, alpha], computation=computation)

def gemm_kernel_512_libpopcnt(activation: Ref, kernel: Ref, output: Ref, N: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, alpha: Ref, uK: int, uK_cleanup: int) -> Block:
    iK = Load(typ=Type.size, src=zero, dst=VarRef("iK"))
    uKl = Literal(f"{uK}")
    aux_size = Expr(Expr(K, Op.div, Expr(Expr(eight, Op.times, BITS), Op.times, uKl)), Op.times, uKl)
    cntp1v = ArrayDeclr(typ=Type.m512i, size=aux_size)
    cntp2v = ArrayDeclr(typ=Type.m512i, size=aux_size)

    def off(i: int):
        return Expr(Literal(f"{i}"), Op.times, Expr(eight, Op.times, BITS))

    def loop_avx(i: int):
        return innermost_512_libpopcnt(activation, kernel, K, iM, iN, Expr(iK.ref, Op.plus, off(i)), cntp1v.ref, cntp2v.ref)
    compute_avx = [instr for i in range(uK) for instr in loop_avx(i).instrs]

    def loop(i: int):
        cntp1 = Load(typ=Type.i32, src=zero)
        cntp2 = Load(typ=Type.i32, src=zero)
        comp = innermost(activation, kernel, K, iM, iN, Expr(iK.ref, Op.plus, Expr(Literal(f"{i}"), Op.times, BITS)), cntp1.ref, cntp2.ref)
        return ([cntp1, cntp2], comp)
    _pre, _compute = list(zip(*[loop(i) for i in range(uK)]))
    pre: List[Computation] = [instr for instrs in _pre for instr in instrs]
    compute = [instr for block in _compute for instr in block.instrs]

    cntp1s: List[Compute] = [f for [f, _] in _pre]
    cntp2s: List[Compute] = [f for [_, f] in _pre]
    cntp1vf = Compute(Type.i32, CallExpr(fn="popcnt", args=[cntp1v.ref, CallExpr("sizeof", [cntp1v.ref])]))
    cntp2vf = Compute(Type.i32, CallExpr(fn="popcnt", args=[cntp2v.ref, CallExpr("sizeof", [cntp2v.ref])]))
    cntp1, cntp1f = addall(cntp1s + [cntp1vf])
    cntp2, cntp2f = addall(cntp2s + [cntp2vf])
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
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, off(uK))),
            op=Op.lte,
            stride=off(uK),
            block=Block(compute_avx)
        ),
        cntp1vf,
        cntp2vf,
        *pre,
        # unrolled cleanup loop
        Loop(
            i=iK.ref,
            lower=None,
            check=Cast(Type.i32, iK.ref),
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, Expr(Literal(f"{uK_cleanup}"), Op.times, BITS))),
            op=Op.lte,
            stride=Expr(Literal(f"{uK_cleanup}"), Op.times, BITS),
            block=Block(compute)
        ),
        *cntp1f,
        *cntp2f,
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

def gemm_kernel_macro_512_libpopcnt(uK: int, uK_cleanup: int) -> Macro:
    activation = VarRef("activation")
    kernel = VarRef("kernel")
    output = VarRef("output")
    N = VarRef("N")
    K = VarRef("K")
    iM = VarRef("iM")
    iN = VarRef("iN")
    alpha = VarRef("alpha")
    computation = gemm_kernel_512_libpopcnt(activation, kernel, output, N, K, iM, iN, alpha, uK, uK_cleanup)
    return Macro(name="gemm_kernel", args=[activation, kernel, output, N, K, BITS, iM, iN, alpha], computation=computation)
