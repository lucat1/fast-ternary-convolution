from enum import Enum

class ConvType(Enum):
    TNN = "TNN"
    TBN = "TBN"
    BTN = "BTN"
    BNN = "BNN"

class Function(Enum):
    TERNARIZE = "TERNARIZE"
    BINARIZE = "BINARIZE"
    IMG2ROW = "IMG2ROW"
    ALLOC = "ALLOC"
    ALLOC2 = "ALLOC2"
    FREE = "FREE"
    PRELU = "PRELU"
    CONV = "CONV"
    TNN_GEMM = "TNN_GEMM"
    TBN_GEMM = "TBN_GEMM"
    BTN_GEMM = "BTN_GEMM"
    BNN_GEMM = "BNN_GEMM"

    def fancy(self) -> str:
        if self == Function.TERNARIZE:
            return 'Ternarize'
        if self == Function.BINARIZE:
            return 'Binarize'
        elif self == Function.IMG2ROW:
            return 'Image to Row'
        elif self == Function.PRELU:
            return 'PreLU'
        elif self == Function.CONV:
            return 'Convolution'
        elif self == Function.TNN_GEMM:
            return 'Ternary General Matrix Multiply'
        elif self == Function.TBN_GEMM:
            return 'Ternary-Binary General Matrix Multiply'
        elif self == Function.BTN_GEMM:
            return 'Binary-Ternary General Matrix Multiply'
        elif self == Function.BNN_GEMM:
            return 'Binary General Matrix Multiply'
        else:
            return ''
