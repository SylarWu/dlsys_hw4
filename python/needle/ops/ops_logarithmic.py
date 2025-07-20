from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        axis = (len(Z.shape) - 1,)
        max_scalar = array_api.max(Z, axis=axis, keepdims=True)
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - max_scalar), axis=axis, keepdims=True)) + max_scalar
        return Z - array_api.broadcast_to(log_sum_exp, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z, = node.inputs
        axis = (len(Z.shape) - 1,)
        sum_out_grad = summation(out_grad, axes=axis)
        return add(
            out_grad,
            negate(
                multiply(
                    exp(node), broadcast_to(reshape(sum_out_grad, sum_out_grad.shape + (1,)), Z.shape)
                )
            )
        )
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_scalar = array_api.max(Z, axis=self.axes, keepdims=True)
        ret = array_api.log(array_api.sum(array_api.exp(Z - max_scalar), axis=self.axes, keepdims=True)) + max_scalar
        return array_api.squeeze(ret, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z, = node.inputs
        axis = tuple(range(0, len(Z.shape))) if self.axes is None else self.axes
        match_shape = tuple([1 if i in axis else n for i, n in enumerate(Z.shape)])
        node = broadcast_to(reshape(node, match_shape), Z.shape)
        out_grad = broadcast_to(reshape(out_grad, match_shape), Z.shape)
        return multiply(out_grad, exp(add(Z, negate(node))))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

