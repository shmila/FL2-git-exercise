import numpy as np

def affine_forward(x,w,b):
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x,w,b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0],-1).T.dot(dout)
    db = np.sum(dout,axis=0)
    return dx,dw,db

def relu_forward(x):
    out = np.maximum(0,x)
    cache = x
    return out,cache

def relu_backward(dout, cache):
    x = cache
    dx = (x > 0)*dout
    return dx

def sig_forward(x):
    out = 1./(1.+np.exp(-x))
    cache = x
    return out,cache

def sig_backward(dout,cache):
    x = cache
    dx = dout*(1-dout)
    return dx

# too complicated for non output layer
# def softmax_forward(x,w,b):
#     scores = x.reshape(x.shape[0], -1).dot(w) + b
#     scores -= np.max(scores)
#     scores_exp = np.exp(scores)
#     scores_exp_sum = np.sum(scores_exp, axis=0)
#     out = scores_exp/scores_exp_sum
#     cache = (x, w, b)
#     return out, cache
#
# def softmax_backward(dout,cache):
#     x, w, b = cache

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def affine_activation_forward(x,w,b,act_func=relu_forward):
    a, fc_cache = affine_forward(x, w, b)
    out, act_cache = act_func(a)
    cache = (fc_cache,act_cache)
    return out,cache


def affine_activation_backward(dout, cache,act_func=relu_backward):
    fc_cache, act_cache = cache
    da = act_func(dout, act_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db







