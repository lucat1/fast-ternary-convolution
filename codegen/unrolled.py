from typing import Tuple, Callable
from codegen.straightforward import innermost
from codegen.ssa import *

Adder = Callable[[Ref | Expr, Ref |Expr], Expr]

def p(q: List[Compute], typ: Type = Type.i32, add_expr: Adder = lambda x, y: Expr(x, Op.plus, y)) -> Tuple[List[Compute], Optional[Compute]]:
    pairs = []
    remainder = None
    while len(q) > 0:
        e1 = q.pop()
        if len(q) == 0:
            remainder = e1
            break

        e2 = q.pop()
        pairs.append(Compute(typ=typ, expr=add_expr(e1.ref, e2.ref)))
    return (pairs, remainder)

def addall(lst: List[Compute], typ: Type = Type.i32, add_expr: Adder = lambda x, y: Expr(x, Op.plus, y)) -> Tuple[Compute, List[Compute]]:
    pairs, remainder = p(lst, typ, add_expr)
    if len(pairs) == 0 and remainder is not None:
        return (remainder, [])
    if len(pairs) == 1 and remainder is None:
        return (pairs[0], pairs)
    elif len(pairs) == 1 and remainder is not None:
        c = Compute(typ=typ, expr=add_expr(pairs[0].ref, remainder.ref))
        return (c, pairs + [c])
    else:
        root, computes = addall(pairs[:], typ, add_expr)
        if remainder is not None:
            root = Compute(typ=typ, expr=add_expr(root.ref, remainder.ref))
            computes.extend([root])
        result = pairs + computes
        return root, result

def gemm_kernel(activation: Ref, kernel: Ref, output: Ref, N: Ref, K: Ref, iM: Expr | Ref, iN: Expr | Ref, alpha: Ref, uK: int) -> Block:
    iK = Load(typ=Type.size, src=zero, dst=VarRef("iK"))
    outidx = Expr(Expr(iM, Op.times, N), Op.plus, iN)
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
    cntp1, cntp1f = addall(cntp1s)
    cntp2, cntp2f = addall(cntp2s)
    subcntp1cntp2 = Compute(Type.i64, Expr(cntp1.ref, Op.minus, cntp2.ref))
    curr = Compute(Type.i64, Expr(subcntp1cntp2.ref, Op.minus, cntp2.ref))
    value = Compute(Type.f32, TernaryExpr(Expr(curr.ref, Op.gt, zero), curr.ref, Expr(curr.ref, Op.times, alpha)))

    return Block([
        *pre,
        iK,
        Loop(
            i=iK.ref,
            lower=None,
            check=Cast(Type.i32, iK.ref),
            upper=Expr(Cast(Type.i32, K), Op.minus, Cast(Type.i32, Expr(Literal(f"{uK}"), Op.times, BITS))),
            op=Op.lte,
            stride=Expr(Literal(f"{uK}"), Op.times, BITS),
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

def gemm_kernel_macro(uK: int) -> Macro:
    activation = VarRef("activation")
    kernel = VarRef("kernel")
    output = VarRef("output")
    N = VarRef("N")
    K = VarRef("K")
    iM = VarRef("iM")
    iN = VarRef("iN")
    alpha = VarRef("alpha")
    computation = gemm_kernel(activation, kernel, output, N, K, iM, iN, alpha, uK)
    return Macro(name="gemm_kernel", args=[activation, kernel, output, N, K, BITS, iM, iN, alpha], computation=computation)
