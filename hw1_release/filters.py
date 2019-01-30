import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    kernel = np.flip(np.flip(kernel, 0), 1)

    ### YOUR CODE HERE
    for m in range(Hi):
        for n in range(Wi):
            for i in range(Hk):
                for j in range(Wk):
                    if m + i >= Hk // 2 and n + j >= Wk // 2 and m + i - Hk // 2 < Hi and n + j - Wk // 2 < Wi:
                        #print (m + i, " ", Wk // 2, " ", n + j, " ", Hk // 2)
                        out[m][n] += kernel[i][j] * image[m + i - Hk // 2][n + j - Wk // 2]
                    #if m - i >= 0 and n - j >= 0 and m - i < Hi and n - j < Wi:
                     #   out[m][n] += kernel[i][j] * image[m - i][n - j]
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    out[pad_height : H + pad_height, pad_width : W + pad_width] = np.copy(image)
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    conv_kernel = np.flip(kernel, axis=0)
    conv_kernel = np.flip(conv_kernel, axis=1)
    padded_img = zero_pad(image, Hk // 2, Wk // 2)
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = np.sum(np.multiply(padded_img[i : i + Hk, j : j + Wk], conv_kernel)) 
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    out = conv_fast(f, np.flip(np.flip(g, 0), 1))
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    zero_mean_g = g - np.mean(g)
    out = cross_correlation(f, zero_mean_g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    g = (g - np.mean(g)) / np.std(g)
    padded_img = zero_pad(f, Hk // 2, Wk // 2)
    for i in range(Hi):
        for j in range(Wi):
            patch = np.copy(padded_img[i : i + Hk, j : j + Wk])
            patch = patch - np.mean(patch)
            patch = patch / np.std(patch)
            out[i][j] = np.sum(np.multiply(patch, g)) 
    ### END YOUR CODE

    return out
