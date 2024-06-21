from enum import Enum

class ConvType(Enum):
    TNN = "TNN"
    TBN = "TBN"
    BTN = "BTN"
    BNN = "BNN"

class Function(Enum):
    TERNARIZE = "ternarize"
    BINARIZE = "binarize"
    TERNA2ROW = "terna2row"
    IM2ROW = "im2row"
    ALLOC = "alloc"
    FREE = "free"
    PRELU = "prelu"
    CONV = "conv"
    GEMM = "gemm"
    GEMMPRELU = "gemmprelu"


    def fancy(self) -> str:
        if self == Function.TERNARIZE:
            return 'Ternarize'
        if self == Function.BINARIZE:
            return 'Binarize'
        elif self == Function.IM2ROW:
            return 'Image to Row'
        elif self == Function.TERNA2ROW:
            return 'Ternarize + Image to Row'
        elif self == Function.GEMMPRELU:
            return 'GEMM + PreLU'
        elif self == Function.PRELU:
            return 'PreLU'
        elif self == Function.CONV:
            return 'Convolution'
        elif self == Function.GEMM:
            return 'Ternary General Matrix Multiply'
        else:
            return ''
