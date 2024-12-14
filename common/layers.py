# coding: utf-8
from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

##modified
class SigmoidWithLoss:
    def __init__(self, penalties=None):
        """
        Penalized Sigmoid with Loss for multilabel classification.

        Args:
        - penalties: A list or numpy array specifying penalties for predicting each label incorrectly.
                     If None, behaves like a standard SigmoidWithLoss.
        """
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # Sigmoid outputs
        self.t = None  # Ground truth labels
        self.penalties = penalties  # Penalty array or None

    def forward(self, x, t):
        """
        Forward pass for Penalized Sigmoid with Loss.

        Args:
        - x: Input logits (batch_size, num_labels).
        - t: Target labels (batch_size, num_labels).

        Returns:
        - Penalized loss value.
        """
        self.t = t
        self.y = 1 / (1 + np.exp(-x))  # Sigmoid activation
        batch_size, num_labels = t.shape

        if self.penalties is None:
            # Default to standard sigmoid loss if no penalties are provided
            penalized_loss = self.t * np.log(self.y + 1e-7) + (1 - self.t) * np.log(1 - self.y + 1e-7)
        else:
            # Use penalties if provided
            penalty_0 = self.penalties[0]  # Penalty for mispredicting 0
            penalty_1 = self.penalties[1]  # Penalty for mispredicting 1
            penalized_loss = (
                penalty_1 * self.t * np.log(self.y + 1e-7) +
                penalty_0 * (1 - self.t) * np.log(1 - self.y + 1e-7)
            )

        self.loss = -np.sum(penalized_loss) / batch_size
        return self.loss

    def backward(self, dout=1):
        """
        Backward pass for Penalized Sigmoid with Loss.

        Args:
        - dout: Upstream gradient.

        Returns:
        - Penalized gradient with respect to input logits.
        """
        batch_size, num_labels = self.t.shape

        if self.penalties is None:
            # Standard gradient calculation if no penalties are provided
            dx = (self.y - self.t) * dout / batch_size
        else:
            # Weighted gradient calculation
            penalty_0 = self.penalties[0]
            penalty_1 = self.penalties[1]
            dx = dout * (
                penalty_1 * self.t * (self.y - 1) +
                penalty_0 * (1 - self.t) * self.y
            ) / batch_size

        return dx




class Dropout:
    '''
    http://arxiv.org/abs/1207.0580
    '''
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        if GPU:
          import cupyx
          cupyx.scatter_add(dW, self.idx, dout)
        else:
          np.add.at(dW, self.idx, dout)
        return None
