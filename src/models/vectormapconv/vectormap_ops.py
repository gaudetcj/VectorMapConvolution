from collections import deque
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState
from scipy.stats import chi
import sys



def vectormap_conv(input, v_weight, c_weight, bias, stride, 
                   padding, groups, dilatation):
    """
    Applies a vectormap convolution to the incoming data:
    """
    
    cat_kernels = [torch.cat([*[v*c for v, c in zip(v_weight, c_weight[0])]], dim=1)]
    for dim in range(1, len(v_weight)):
        v_weight = torch.cat((v_weight[-1:], v_weight[:-1]))
        cat_kernels.append(torch.cat([*[v*c for v, c in zip(v_weight, c_weight[dim])]], dim=1))

    cat_kernel = torch.cat([*cat_kernels], dim=0)

    if   input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, cat_kernel, bias, stride, padding, dilatation, groups)


def vectormap_transpose_conv(input, v_weight, c_weight, bias, stride,
                    padding, output_padding, groups, dilatation):
    """
    Applies a vectormap transposed convolution to the incoming data:
    """

    cat_kernels = [torch.cat([*[v*c for v, c in zip(v_weight, c_weight[0])]], dim=1)]
    for dim in range(1, len(v_weight)):
        v_weight = torch.cat((v_weight[-1:], v_weight[:-1]))
        cat_kernels.append(torch.cat([*[v*c for v, c in zip(v_weight, c_weight[dim])]], dim=1))

    cat_kernel = torch.cat([*cat_kernels], dim=0)


    if   input.dim() == 3:
        convfunc = F.conv_transpose1d
    elif input.dim() == 4:
        convfunc = F.conv_transpose2d
    elif input.dim() == 5:
        convfunc = F.conv_transpose3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, cat_kernel, bias, stride, padding, output_padding, groups, dilatation)


def vectormap_linear(input, v_weight, c_weight, bias=True):
    """
    Applies a vectormap linear transformation to the incoming data
    """

    cat_kernels = torch.cat([[v*c for v, c in zip(v_weight, c_weight[0])]], dim=1)
    for dim in range(1, len(v_weight)):
        v_weight.append(v_weight.pop(0))
        cat_kernels.append(torch.cat([[v*c for v, c in zip(v_weight, c_weight[dim])]], dim=1))

    cat_kernel   = torch.cat(cat_kernels, dim=0)

    if input.dim() == 2 :

        if bias is not None:
            return torch.addmm(bias, input, cat_kernel)
        else: 
            return torch.mm(input, cat_kernel)
    else:
        output = torch.matmul(input, cat_kernel)
        if bias is not None:
            return output+bias
        else:
            return output


def unitary_init(vectormap_dim, in_features, out_features, rng, kernel_size=None, criterion='he'):
    
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features  
        fan_out         = out_features 

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    s = np.sqrt(3.0) * s

    number_of_weights = np.prod(kernel_shape) 
    v_r = np.random.uniform(-s,s,number_of_weights)
    v_i = np.random.uniform(-s,s,number_of_weights)
    v_j = np.random.uniform(-s,s,number_of_weights)
    v_k = np.random.uniform(-s,s,number_of_weights)
    
    # Unitary quaternion
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i]**2 + v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
        v_r[i]/= norm
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    return (v_r, v_i, v_j, v_k)


def random_init(vectormap_dim, in_features, out_features, rng, kernel_size=None, criterion='glorot'):
    
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features  
        fan_out         = out_features 

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape) 
    v_r = np.random.uniform(0.0,1.0,number_of_weights)
    v_i = np.random.uniform(0.0,1.0,number_of_weights)
    v_j = np.random.uniform(0.0,1.0,number_of_weights)
    v_k = np.random.uniform(0.0,1.0,number_of_weights)

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    weight_r = v_r * s
    weight_i = v_i * s
    weight_j = v_j * s
    weight_k = v_k * s
    return (weight_r, weight_i, weight_j, weight_k)


def vectormap_init(vectormap_dim, in_features, out_features, rng, kernel_size=None, criterion='glorot'):
    
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if criterion == 'glorot':
        s = np.sqrt(2. / (vectormap_dim*(fan_in + fan_out)))
    elif criterion == 'he':
        s = np.sqrt(2. / (vectormap_dim*fan_in))
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState(np.random.randint(1, 1234))

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(vectormap_dim, loc=0, scale=s, size=kernel_shape)
    number_of_weights = np.prod(kernel_shape) 

    v_s = np.array([np.random.uniform(-1.0, 1.0, number_of_weights) for _ in range(vectormap_dim - 1)])
    for i in range(0, number_of_weights):
        v_s[:, i] = v_s[:, i] / np.linalg.norm(v_s[:, i])

    v_s = [v.reshape(kernel_shape) for v in v_s]

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight = [modulus * np.cos(phase)]
    for v in v_s:
        weight.append(modulus * v * np.sin(phase))

    return tuple(weight)


def affect_init_conv(v_weight, kernel_size, init_func, rng, init_criterion):
    for dim in range(1, len(v_weight)):
        if v_weight[dim - 1].size() != v_weight[dim].size():
            raise ValueError("""Dimensions {} and {} 
                of the vectormap weights differ""".format(dim - 1, dim))
    
    if 2 >= v_weight[0].dim():
        raise Exception('affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = '
                        + str(v_weight[0].dim()))

    v_s = init_func(
        len(v_weight),
        v_weight[0].size(1),
        v_weight[0].size(0),
        rng=rng,
        kernel_size=kernel_size,
        criterion=init_criterion
    )
    v_s = [torch.from_numpy(v) for v in v_s]
    v_weight.data = torch.stack(v_s).type_as(v_weight.data)

def get_kernel_and_weight_shape(operation, in_channels, out_channels, kernel_size):
    if operation == 'convolution1d':
        if type(kernel_size) is not int:
            raise ValueError(
                """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                must be integer in the case. Found kernel_size = """ + str(kernel_size)
            )
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels) + tuple((ks,))
    else:# in case it is 2d or 3d.
        if   operation == 'convolution2d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size)
        elif operation == 'convolution3d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size, kernel_size)
        elif type(kernel_size) is not int:
            if   operation == 'convolution2d' and len(kernel_size) != 2:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                    must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                )
            elif operation == 'convolution3d' and len(kernel_size) != 3:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                    must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                )
            else:
                ks = kernel_size
        w_shape = (out_channels, in_channels) + (*ks,)
    return ks, w_shape