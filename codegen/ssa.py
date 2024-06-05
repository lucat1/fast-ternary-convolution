from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Set, cast

class Type(Enum):
    size = "size_t"
    i64 = "int64_t"
    i32 = "int"
    f32 = "float"
    m256i = "__m256i"
    m512i = "__m512i"

class Literal():
    def __init__(self, value: str) -> None:
        self.value = value
    
    def gen(self) -> str:
        return self.value

class Op(Enum):
    plus = '+'
    minus = '-'
    times = '*'
    div = '/'

    bxor = '^'
    band = '&'

    gt = '>'
    lt = '<'
    lte = '<='

class Expr():
    def __init__(self, e1: "Ref | Literal | Expr", op: Optional[Op] = None, e2: Optional["Ref | Literal | Expr"] = None) -> None:
        self.e1 = e1
        self.op = op
        self.e2 = e2
        assert self.op is None or self.e2 is not None

    def gen(self) -> str:
        if self.op is not None and self.e2 is not None:
            return f"(({self.e1.gen()}) {self.op.value} ({self.e2.gen()}))"
        else:
            return self.e1.gen()

class Ref(ABC):
    @abstractmethod
    def gen(self) -> str:
        pass

class VarRef(Ref):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def gen(self) -> str:
        return self.name

class ArrayRef(Ref):
    def __init__(self, arr: Ref, idx: Expr) -> None:
        super().__init__()
        self.arr = arr
        self.idx = idx

    def gen(self) -> str:
        return f"({self.arr.gen()})[{self.idx.gen()}]"

class AVXLoadKind(Enum):
    # TODO: addresses should be aligned
    load_si256 = 'loadu_si256'
    load_epi64 = 'loadu_si512'

class MRef(Ref):
    def __init__(self, load_kind: AVXLoadKind, arr: Ref, idx: Expr) -> None:
        super().__init__()
        self.load_kind = load_kind
        self.arr = arr
        self.idx = idx

    def gen(self) -> str:
        tt = self.load_kind.value.split('_')[1]
        t = 'i' if 'i' in tt else 'd'
        s = tt[-3:]
        return f"_mm{s}_{self.load_kind.value}((__m{s}{t} *) ({self.arr.gen()} + ({self.idx.gen()})))"

class CallExpr(Expr):
    def __init__(self, fn: str, args: List[Ref | Expr]) -> None:
        self.fn = fn
        self.args = args

    def gen(self) -> str:
        return f"{self.fn}({','.join(map(lambda a: a.gen(), self.args))})"

class Cast(Expr):
    def __init__(self, typ: Type, expr: Ref | Expr | Literal) -> None:
        self.typ = typ
        self.expr = expr

    def gen(self) -> str:
        return f"(({self.typ.value}) ({self.expr.gen()}))"

class TernaryExpr(Expr):
    def __init__(self, cond: "Ref | Literal | Expr", e1: "Ref | Literal | Expr", e2: "Ref | Literal | Expr") -> None:
        self.cond = cond
        self.e1 = e1
        self.e2 = e2

    def gen(self) -> str:
        return f"(({self.cond.gen()}) ? ({self.e1.gen()}) : ({self.e2.gen()}))"

class RefExpr(Ref):
    def __init__(self, ref: Ref) -> None:
        super().__init__()
        self.ref = ref

    def gen(self) -> str:
        return f"&{self.ref.gen()}"

class Code(ABC):
    @abstractmethod
    def gen(self) -> str:
        pass

class InstructionKind(Enum):
    load = 0
    compute = 1
    store = 2

class Instruction(Code):
    def __init__(self, kind: InstructionKind, dst: Ref, src: Ref | Expr, typ: Optional[Type] = None) -> None:
        super().__init__()
        self.kind = kind
        self.dst = dst
        self.src = src
        self.typ = typ

    def gen(self) -> str:
        return f"{self.dst.gen()} = ({self.src.gen()});"

    def declr(self) -> Optional[str]:
        if self.typ is not None:
            return f"{self.typ.value} {self.dst.gen()};"

class Declr(Code):
    def __init__(self, typ: Type, dst: Ref, size: Optional[Ref | Expr | Literal] = None) -> None:
        super().__init__()
        self.kind = InstructionKind.load
        self.dst = dst
        self.size = size
        self.typ = typ

    def gen(self) -> str:
        return ""

    def declr(self) -> Optional[str]:
        if self.size is not None:
            return f"{self.typ.value} {self.dst.gen()}[{self.size.gen()}];"
        else:
            return f"{self.typ.value} {self.dst.gen()};"

class Computation():
    # returns true if the current computation is only used for controlling the flow of the program
    @property
    def is_cf(self) -> bool:
        return False

    @property
    def declrs(self) -> List[str]:
        return []

    @property
    def steps(self) -> List[Code]:
        return []

class ArrayDeclr(Computation):
    _i = 0

    def __init__(self, typ: Type, size: Expr | Ref | Literal, dst: Optional[Ref] = None) -> None:
        super().__init__()
        self.typ = typ
        self.size = size
        if dst is not None:
            self.dst = dst
        else:
            ArrayDeclr._i += 1
            self.dst = VarRef(f"declr{ArrayDeclr._i}")

    @property
    def ref(self) -> Ref:
        return self.dst

    @property
    def steps(self) -> List[Code]:
        return [Declr(typ= self.typ, dst=self.dst, size=self.size)]

class Load(Computation):
    _i = 0

    def __init__(self, typ: Type, src: Ref | Expr, dst: Optional[Ref] = None) -> None:
        super().__init__()
        self.typ = typ
        self.src = src
        if dst is not None:
            self.dst = dst
        else:
            Load._i += 1
            self.dst = VarRef(f"load{Load._i}")

    @property
    def ref(self) -> Ref:
        return self.dst

    @property
    def steps(self) -> List[Code]:
        return [Instruction(kind=InstructionKind.load, dst=self.dst, src=self.src, typ=self.typ)]

class Compute(Computation):
    _i = 0

    def __init__(self, typ: Type, expr: Expr, dst: Optional[Ref] = None) -> None:
        super().__init__()
        self.typ = typ
        self.expr = expr

        if dst is not None:
            self.dst = dst
        else:
            Compute._i += 1
            self.dst = VarRef(f"comp{Compute._i}")

    @property
    def ref(self) -> Ref:
        return self.dst

    @property
    def steps(self) -> List[Code]:
        return [Instruction(kind=InstructionKind.compute, dst=self.dst, src=self.expr, typ=self.typ)]

class Store(Computation):
    def __init__(self, dst: Ref, src: Expr | Ref) -> None:
        super().__init__()
        self.dst = dst
        self.src = src

    @property
    def steps(self) -> List[Code]:
        return [Instruction(kind=InstructionKind.store, dst=self.dst, src=self.src)]

class Block(Computation):
    def __init__(self, instrs: List[Computation]) -> None:
        super().__init__()
        self.instrs = instrs
        self.components: List[List[Computation] | Computation] = []
        comp = []
        for instr in self.instrs:
            if instr.is_cf:
                self.components.append(comp)
                self.components.append(instr)
                comp = []
            else:
                comp.append(instr)
        if len(comp) > 0:
            self.components.append(comp)

    def filter(self, comp: List[Computation], kind: InstructionKind) -> List[Instruction]:
        return [step for instr in comp for step in instr.steps if step.kind == kind]

    def declrs(self) -> List[str]:
        def m(i: Instruction) -> str | None:
            return i.declr()

        def f(d: str | None) -> bool:
            return d is not None

        def extract(comp) -> List[str]:
            if isinstance(comp, List):
                loads = list(filter(f, map(m, self.filter(comp, InstructionKind.load))))
                computes = list(filter(f, map(m, self.filter(comp, InstructionKind.compute))))
                stores = list(filter(f, map(m, self.filter(comp, InstructionKind.store))))
                return cast(List[str], loads + computes + stores)
            else:
                return comp.declrs()

        return [ele for comp in self.components for ele in extract(comp)]

    def gen(self) -> str:
        out = ""
        for comp in self.components:
            if isinstance(comp, List):
                loads = list(map(lambda instr: instr.gen(), self.filter(comp, InstructionKind.load)))
                computes = list(map(lambda instr: instr.gen(), self.filter(comp, InstructionKind.compute)))
                stores = list(map(lambda instr: instr.gen(), self.filter(comp, InstructionKind.store)))

                if len(loads) > 0:
                    out += '\n\n'
                    out += '\n'.join(loads)
                if len(computes) > 0:
                    out += '\n\n'
                    out += '\n'.join(computes)
                if len(stores) > 0:
                    out += '\n'
                    out += '\n'.join(stores)
            else:
                out += '\n'
                out += comp.gen()

        return '  '.join(('\n' + out).splitlines(True))[1:]

class Loop(Computation):
    def __init__(self, i: Ref, lower: Expr | Ref | Literal | None, upper: Expr | Ref | Literal, stride: Expr | Ref | Literal, block: Block, op: Op = Op.lt, check: Optional[Expr | Ref | Literal] = None) -> None:
        self.i = i
        self.lower = lower
        self.upper = upper
        self.stride = stride
        self.op = op
        self.block = block
        if check is not None:
            self.check = check
        else:
            self.check = i

    @property
    def is_cf(self) -> bool:
        return True

    def declrs(self) -> List[str]:
        return self.block.declrs() + [f"size_t {self.i.gen()};"]

    @property
    def steps(self) -> List[Code]:
      return []

    def gen(self) -> str:
        block = self.block.gen()
        lower = f"{self.i.gen()} = {self.lower.gen()}" if self.lower is not None else ""
        return f"for({lower}; ({self.check.gen()}) {self.op.value} ({self.upper.gen()}); {self.i.gen()} += {self.stride.gen()})\n{{" + block + '\n}'

class Macro():
    def __init__(self, name: str, args: List[Ref], computation: Computation) -> None:
        self.name = name
        self.computation = computation
        self.args = args

    def gen(self) -> str:
        macro = f"#define {self.name}({', '.join(map(lambda a: a.gen(), self.args))})"
        declrs = set(self.computation.declrs())
        body = '\n'.join(declrs)
        body = '  '.join(('\n' + body).splitlines(True))
        body += '\n' + self.computation.gen()
        lines = [macro] + ("do {" + body + "\n} while(0);").split('\n') 
        longest = max([len(line) for line in lines] + [80])
        lines = [line.ljust(longest+2) + '\\' for line in lines]
        return '\n'.join(lines)

zero = Expr(Literal("0"))
one = Expr(Literal("1"))
four = Literal("4")
eight = Literal("8")
BITS = VarRef("BITS")
