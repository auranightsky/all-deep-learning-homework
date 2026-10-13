

import numpy as np

def conv_forward(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # Replace "pass" statement with your code
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Add zero-padding around the input
    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant"
    )

    # Calculate output dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros((N, F, H_out, W_out))

    # Slide each filter across the input
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH

                    w_start = j * stride
                    w_end = w_start + WW

                    # Extract the current input region
                    region = x_pad[n, :, h_start:h_end, w_start:w_end]

                    # Multiply, sum, and add bias
                    out[n, f, i, j] = (np.sum(region * w[f]) + b[f])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # Replace "pass" statement with your code
    x, w, b, conv_param = cache

    stride = conv_param["stride"]
    pad = conv_param["pad"]

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape

    # Pad the input
    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant"
    )

    # Initialize gradients
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Calculate gradients
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH

                    w_start = j * stride
                    w_end = w_start + WW

                    # Current input region
                    region = x_pad[n, :, h_start:h_end, w_start:w_end]

                    # Upstream gradient for this output
                    grad = dout[n, f, i, j]

                    # Gradient with respect to the bias
                    db[f] += grad

                    # Gradient with respect to the filter
                    dw[f] += region * grad

                    # Gradient with respect to the input region
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += w[f] * grad

    # Remove padding from the input gradient
    if pad > 0:
        dx = dx_pad[:, :, pad:-pad, pad:-pad]
    else:
        dx = dx_pad
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # Replace "pass" statement with your code
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    N, C, H, W = x.shape

    # Calculate output dimensions
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out))

    # Slide the pooling window across each input
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height

                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Extract the current pooling region
                    region = x[n, c, h_start:h_end, w_start:w_end]

                    # Keep only the maximum value
                    out[n, c, i, j] = np.max(region)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # Replace "pass" statement with your code
    x, pool_param = cache

    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)

    # Send each gradient to the maximum element in its region
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height

                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Extract the original pooling region
                    region = x[n, c, h_start:h_end, w_start:w_end]

                    # Find the position of the maximum value
                    max_index = np.argmax(region)
                    max_h, max_w = np.unravel_index(max_index, region.shape)

                    # Pass the gradient to the maximum element
                    dx[n, c, h_start + max_h, w_start + max_w] += dout[n, c, i, j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

