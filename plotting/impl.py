from abc import ABC, abstractmethod
from plotting.datatypes import Function
import pandas as pd

class Cost:
    def __init__(self, iops: int, flops: int, q: int) -> None:
        self._iops = iops
        self._flops = flops
        self._q = q

    @property
    def iops(self) -> int:
        return self._iops

    @property
    def flops(self) -> int:
        return self._flops

    @property
    def q(self) -> int:
        return self._q
    
    def add(self, c: "Cost") -> "Cost":
        return Cost(self.iops + c.iops, self.flops + c.flops, self.q + c.q)

    def __add__(self, c: "Cost") -> "Cost":
        return self.add(c)

    def __iadd__(self, c: "Cost") -> "Cost":
        self = self.add(c)
        return self

    def scalar_mul(self, k: int) -> "Cost":
        return Cost(self.iops * k, self.flops * k, self.q * k)

    def __mul__(self, k: int) -> "Cost":
        return self.scalar_mul(k)

    def __imul__(self, k: int) -> "Cost":
        self = self.scalar_mul(k)
        return self

class Impl(ABC):
    def __init__(self, parameters: pd.Series) -> None:
        super().__init__()
        self.p = parameters

    def conv(self) -> Cost:
        return self.ternarize_im2row().add(self.gemm_prelu())

    @abstractmethod
    def ternarize(self) -> Cost:
        raise NotImplemented

    @abstractmethod
    def im2row(self) -> Cost:
        raise NotImplemented

    def ternarize_im2row(self) -> Cost:
        return self.ternarize().add(self.im2row())

    @abstractmethod
    def gemm(self) -> Cost:
        raise NotImplemented

    def im2row_gemm(self) -> Cost:
        return self.im2row().add(self.gemm())

    @abstractmethod
    def prelu(self) -> Cost:
        raise NotImplemented

    def gemm_prelu(self) -> Cost:
        return self.gemm().add(self.prelu())

    def im2row_gemm_prelu(self) -> Cost:
        return self.im2row().add(self.gemm()).add(self.prelu())

    def cost(self) -> Cost:
        fn = self.p.fn
        if fn == Function.TERNARIZE.value:
            return self.ternarize()
        if fn == Function.IM2ROW.value:
            return self.im2row()
        if fn == Function.TERNA2ROW.value:
            return self.ternarize_im2row()
        elif fn == Function.GEMM.value:
            return self.gemm_prelu()
        elif fn == Function.PRELU.value:
            return self.prelu()
        elif fn == Function.GEMMPRELU.value:
            return self.gemm_prelu()
        elif fn == Function.CONV.value:
            return self.conv()
        else:
            raise Exception(f"Cannot compute cost for function: {fn}")
