import numpy as np

def indirect_convolution(input, weights, indirection, stride=1, padding=0):
    # Extract dimensions
    N, H, W, C = input.shape
    KH, KW = weights.shape[0] // (H * W * C), weights.shape[1] // (H * W * C) # kernel dimensions inferred from weight shape
    OH = (H - KH + 2 * padding) // stride + 1
    OW = (W - KW + 2 * padding) // stride + 1

    # Output tensor
    output = np.zeros((N, OH, OW, weights.shape[0])) # NOH*NOW presumed to be weights.shape[0]

    # Perform convolution
    for n in range(N):
        for oh in range(OH):
            for ow in range(OW):
                # This index maps to the flat version of the kernel-window matrix
                ind_idx = oh * OW + ow
                # indirection contains the start indices of each window in the input tensor
                window_start = indirection[ind_idx]
                # Extract the corresponding window using the indirection indices
                window = input[n].flat[window_start:window_start + KH * KW * C]
                # Perform the dot product with the weights
                output[n, oh, ow] = np.dot(weights[ind_idx], window)

    return output

# Example usage
N, H, W, C = 1, 28, 28, 3 # Input dimensions
KH, KW = 3, 3 # Kernel dimensions
weights_shape = ((H - KH + 1) * (W - KW + 1), KH * KW * C) # Shape of weights

# Random weights and input for testing
weights = np.random.rand(*weights_shape)
input = np.random.rand(N, H, W, C)

# Indirection vector needs to be prepared based on the convolution sliding
indirection = np.array([h * W * C + w * C for h in range(H - KH + 1) for w in range(W - KW + 1)])

# Perform convolution
output = indirect_convolution(input, weights, indirection)
print(output.shape) # Should print (N, OH, OW, weights.shape[0])
