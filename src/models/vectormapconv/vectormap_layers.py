import numpy as np
from numpy.random import RandomState
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.nn import Module
from .vectormap_ops import *
import math
import sys


class VectorMapConv(Module):
    r"""Applies a VectorMap Convolution to the incoming data.
    """

    def __init__(self, vectormap_dim, in_channels, out_channels, kernel_size, stride=1,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='vector', seed=None, operation='convolution2d'):

        super(VectorMapConv, self).__init__()

        if in_channels % vectormap_dim != 0:
            raise RuntimeError(
                "VectorMap Tensors must be integer divisible by the vector dimension given."
                "In channels = {} is not divisible by vector dimension = {}".format(in_channels, vectormap_dim)
            )

        if out_channels % vectormap_dim != 0:
            raise RuntimeError(
                "VectorMap Tensors must be integer divisible by the vector dimension given."
                "Out channels = {} is not divisible by vector dimension = {}".format(out_channels, vectormap_dim)
            )

        self.vectormap_dim = vectormap_dim
        self.in_channels = in_channels  // vectormap_dim
        self.out_channels = out_channels // vectormap_dim
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0,1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.winit = {'vector': vectormap_init,
                      'unitary' : unitary_init,
                      'random' : random_init}[self.weight_init]

        
        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation, 
            self.in_channels, self.out_channels, kernel_size )

        self.v_weight = Parameter(torch.Tensor(*(self.vectormap_dim,) + self.w_shape))

        bernoulli_probs = torch.ones(*(self.vectormap_dim, self.vectormap_dim)) * 0.5
        bernoilli = torch.bernoulli(bernoulli_probs)
        ones = torch.ones(*(self.vectormap_dim, self.vectormap_dim))
        self.c_weight = Parameter(ones * bernoilli + (bernoilli - 1.0))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.v_weight, self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):
        return vectormap_conv(input, self.v_weight, self.c_weight, self.bias, 
            self.stride, self.padding, self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', dilation='       + str(self.dilation) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', operation='      + str(self.operation) + ')'


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


class VectorMapBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, vectormap_dim, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(VectorMapBatchNorm2d, self).__init__(num_features, eps=1e-5, momentum=0.1,
                                                   affine=True, track_running_stats=True)
        self.vectormap_dim = vectormap_dim
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.zeros(*(self.vectormap_dim, self.vectormap_dim))) #+
                          #torch.diag( torch.Tensor([1. / torch.sqrt(torch.Tensor([self.vectormap_dim]))]*self.vectormap_dim)))
            self.bias = Parameter(torch.zeros(self.vectormap_dim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.vectormap_dim))
            self.register_buffer('running_var', torch.ones(*(self.vectormap_dim, self.vectormap_dim)))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0,1)
        return_shape = y.shape
        total_planes = return_shape[0]
        dim_planes = total_planes//self.vectormap_dim
        y = torch.cat([*[y[dim_planes * dim : dim_planes * (dim+1), :].contiguous().view(1, -1) for dim in range(self.vectormap_dim)]], dim=0)

        mu = y.mean(dim=1)
        sigma2 = cov(y, rowvar=True)
        W = torch.cholesky(torch.inverse(sigma2))
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
            y = torch.mm(self.running_var, y)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
                    self.running_var = (1-self.momentum)*self.running_var + self.momentum*W
            y = y - mu.view(-1, 1)
            y = torch.mm(W, y)

        y = torch.mm(self.weight, y) + self.bias[:, None]

        y = torch.cat([*[row.view(dim_planes, row.size(0)//dim_planes) for row in y]], dim=0)
        return y.view(return_shape).transpose(0, 1)