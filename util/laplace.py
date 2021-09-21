# This function solves a Laplance equation
# "Byung-Woo Hong in Chung-Ang University AI department"
#
# objective function:
#   | I - f |^2 * X + | I - b |^2 * (1 - X) + alpha * | \nabla f |^2 + alpha * | \nabla b |^2
#
# Euler-Lagrange equation:
#   - alpha \Delta u = I \cdot X - u \cdot X
#
# f     : foreground
# b     : background
# I     : input data
# X  : characteristic function (binary)
# alpha : weight for the regularization (L_2^2)
import numpy as np 
import torch
import torch.nn.functional as F


class Laplace:

    def __init__(self, dt=0.25, alpha=1, device='cuda'):

        self.dt     = dt
        self.alpha  = alpha
        self.device = device


    def diffuse(self, I, X, num_iter=1):

        number_image_I  = I.shape[0]
        channel_image_I = I.shape[1]
        height_image_I  = I.shape[2]
        width_image_I   = I.shape[3]
        
        number_image_X  = X.shape[0]
        channel_image_X = X.shape[1]
        height_image_X  = X.shape[2]
        width_image_X   = X.shape[3]

        assert I.ndim == 4,         'dimension of input image I should be 4'
        assert X.ndim == 4,         'dimension of input mask X should be 4'
        assert I.ndim == X.ndim,    'dimension of input image I should match dimension of input mask X'

        p2d = (1, 1, 1, 1)

        _I_expand           = F.pad(I, p2d, 'replicate')
        _X_f_expand         = F.pad(X, p2d, 'replicate') 
        _X_b_expand         = F.pad(1-X, p2d, 'replicate')

        _X_f_expand_up      = torch.roll(_X_f_expand, -1, 2)
        _X_f_expand_down    = torch.roll(_X_f_expand, +1, 2)
        _X_f_expand_left    = torch.roll(_X_f_expand, -1, 3)
        _X_f_expand_right   = torch.roll(_X_f_expand, +1, 3)

        _X_b_expand_up      = torch.roll(_X_b_expand, -1, 2)
        _X_b_expand_down    = torch.roll(_X_b_expand, +1, 2)
        _X_b_expand_left    = torch.roll(_X_b_expand, -1, 3)
        _X_b_expand_right   = torch.roll(_X_b_expand, +1, 3)

        _X_f_border_up      = torch.mul(_X_f_expand, 1-_X_f_expand_up)
        _X_f_border_down    = torch.mul(_X_f_expand, 1-_X_f_expand_down)
        _X_f_border_left    = torch.mul(_X_f_expand, 1-_X_f_expand_left)
        _X_f_border_right   = torch.mul(_X_f_expand, 1-_X_f_expand_right)

        _X_b_border_up      = torch.mul(_X_b_expand, 1-_X_b_expand_up)
        _X_b_border_down    = torch.mul(_X_b_expand, 1-_X_b_expand_down)
        _X_b_border_left    = torch.mul(_X_b_expand, 1-_X_b_expand_left)
        _X_b_border_right   = torch.mul(_X_b_expand, 1-_X_b_expand_right)

        _f_expand           = torch.mul(_I_expand, _X_f_expand)
        _b_expand           = torch.mul(_I_expand, _X_b_expand)
        _f_update           = torch.zeros_like(_f_expand).to(self.device)
        _b_update           = torch.zeros_like(_b_expand).to(self.device)


        for i in range(num_iter):

            _f_expand_up    = torch.roll(_f_expand, -1, 2)
            _f_expand_down  = torch.roll(_f_expand, +1, 2)
            _f_expand_left  = torch.roll(_f_expand, -1, 3)
            _f_expand_right = torch.roll(_f_expand, +1, 3)

            _b_expand_up    = torch.roll(_b_expand, -1, 2)
            _b_expand_down  = torch.roll(_b_expand, +1, 2)
            _b_expand_left  = torch.roll(_b_expand, -1, 3)
            _b_expand_right = torch.roll(_b_expand, +1, 3)

            _f_laplace  = torch.mul(_f_expand_up, 1-_X_f_border_up) + torch.mul(_f_expand, _X_f_border_up) \
                        + torch.mul(_f_expand_down, 1-_X_f_border_down) + torch.mul(_f_expand, _X_f_border_down) \
                        + torch.mul(_f_expand_left, 1-_X_f_border_left) + torch.mul(_f_expand, _X_f_border_left) \
                        + torch.mul(_f_expand_right, 1-_X_f_border_right) + torch.mul(_f_expand, _X_f_border_right) \
                        - 4 * _f_expand 

            _b_laplace  = torch.mul(_b_expand_up, 1-_X_b_border_up) + torch.mul(_b_expand, _X_b_border_up) \
                        + torch.mul(_b_expand_down, 1-_X_b_border_down) + torch.mul(_b_expand, _X_b_border_down) \
                        + torch.mul(_b_expand_left, 1-_X_b_border_left) + torch.mul(_b_expand, _X_b_border_left) \
                        + torch.mul(_b_expand_right, 1-_X_b_border_right) + torch.mul(_b_expand, _X_b_border_right) \
                        - 4 * _b_expand 

            _f_update   = _f_expand + self.alpha * self.dt * _f_laplace
            _b_update   = _b_expand + self.alpha * self.dt * _b_laplace

            _f_diff     = torch.sum(torch.abs(torch.flatten(_f_update) - torch.flatten(_f_expand)))
            _b_diff     = torch.sum(torch.abs(torch.flatten(_b_update) - torch.flatten(_b_expand)))
            
            _f_expand   = _f_update
            _b_expand   = _b_update
        
            # print("[{0:3d}] err(f) : {1}, err(b) : {2}".format(i, _f_diff, _b_diff))
            
        _f = _f_expand[:,:,1:-1,1:-1]
        _b = _b_expand[:,:,1:-1,1:-1]
        
        # _f = torch.mul(_f, X)
        # _b = torch.mul(_b, 1-X)
        
        return (_f, _b)


def laplace_diffuse(I, X, num_iter=1, dt=0.25, alpha=1, device='cuda'):

    number_image_I  = I.shape[0]
    channel_image_I = I.shape[1]
    height_image_I  = I.shape[2]
    width_image_I   = I.shape[3]

    number_image_X  = X.shape[0]
    channel_image_X = X.shape[1]
    height_image_X  = X.shape[2]
    width_image_X   = X.shape[3]

    assert I.ndim == 4,         'dimension of input image I should be 4'
    assert X.ndim == 4,         'dimension of input mask X should be 4'
    assert I.ndim == X.ndim,    'dimension of input image I should match dimension of input mask X'

    p2d = (1, 1, 1, 1)

    _I_expand           = F.pad(I, p2d, 'replicate')
    _X_f_expand         = F.pad(X, p2d, 'replicate') 
    _X_b_expand         = F.pad(1-X, p2d, 'replicate')

    _X_f_expand_up      = torch.roll(_X_f_expand, -1, 2)
    _X_f_expand_down    = torch.roll(_X_f_expand, +1, 2)
    _X_f_expand_left    = torch.roll(_X_f_expand, -1, 3)
    _X_f_expand_right   = torch.roll(_X_f_expand, +1, 3)

    _X_b_expand_up      = torch.roll(_X_b_expand, -1, 2)
    _X_b_expand_down    = torch.roll(_X_b_expand, +1, 2)
    _X_b_expand_left    = torch.roll(_X_b_expand, -1, 3)
    _X_b_expand_right   = torch.roll(_X_b_expand, +1, 3)

    _X_f_border_up      = torch.mul(_X_f_expand, 1-_X_f_expand_up)
    _X_f_border_down    = torch.mul(_X_f_expand, 1-_X_f_expand_down)
    _X_f_border_left    = torch.mul(_X_f_expand, 1-_X_f_expand_left)
    _X_f_border_right   = torch.mul(_X_f_expand, 1-_X_f_expand_right)

    _X_b_border_up      = torch.mul(_X_b_expand, 1-_X_b_expand_up)
    _X_b_border_down    = torch.mul(_X_b_expand, 1-_X_b_expand_down)
    _X_b_border_left    = torch.mul(_X_b_expand, 1-_X_b_expand_left)
    _X_b_border_right   = torch.mul(_X_b_expand, 1-_X_b_expand_right)

    _f_expand           = torch.mul(_I_expand, _X_f_expand)
    _b_expand           = torch.mul(_I_expand, _X_b_expand)
    _f_update           = torch.zeros_like(_f_expand).to(device)
    _b_update           = torch.zeros_like(_b_expand).to(device)


    for i in range(num_iter):

        _f_expand_up    = torch.roll(_f_expand, -1, 2)
        _f_expand_down  = torch.roll(_f_expand, +1, 2)
        _f_expand_left  = torch.roll(_f_expand, -1, 3)
        _f_expand_right = torch.roll(_f_expand, +1, 3)

        _b_expand_up    = torch.roll(_b_expand, -1, 2)
        _b_expand_down  = torch.roll(_b_expand, +1, 2)
        _b_expand_left  = torch.roll(_b_expand, -1, 3)
        _b_expand_right = torch.roll(_b_expand, +1, 3)

        _f_laplace  = torch.mul(_f_expand_up, 1-_X_f_border_up) + torch.mul(_f_expand, _X_f_border_up) \
                    + torch.mul(_f_expand_down, 1-_X_f_border_down) + torch.mul(_f_expand, _X_f_border_down) \
                    + torch.mul(_f_expand_left, 1-_X_f_border_left) + torch.mul(_f_expand, _X_f_border_left) \
                    + torch.mul(_f_expand_right, 1-_X_f_border_right) + torch.mul(_f_expand, _X_f_border_right) \
                    - 4 * _f_expand 

        _b_laplace  = torch.mul(_b_expand_up, 1-_X_b_border_up) + torch.mul(_b_expand, _X_b_border_up) \
                    + torch.mul(_b_expand_down, 1-_X_b_border_down) + torch.mul(_b_expand, _X_b_border_down) \
                    + torch.mul(_b_expand_left, 1-_X_b_border_left) + torch.mul(_b_expand, _X_b_border_left) \
                    + torch.mul(_b_expand_right, 1-_X_b_border_right) + torch.mul(_b_expand, _X_b_border_right) \
                    - 4 * _b_expand 

        _f_update   = _f_expand + alpha * dt * _f_laplace
        _b_update   = _b_expand + alpha * dt * _b_laplace

        _f_diff     = torch.sum(torch.abs(torch.flatten(_f_update) - torch.flatten(_f_expand)))
        _b_diff     = torch.sum(torch.abs(torch.flatten(_b_update) - torch.flatten(_b_expand)))

        _f_expand   = _f_update
        _b_expand   = _b_update

        print("[{0:3d}] err(f) : {1}, err(b) : {2}".format(i, _f_diff, _b_diff))

    _f = _f_expand[:,:,1:-1,1:-1]
    _b = _b_expand[:,:,1:-1,1:-1]

    _f = torch.mul(_f, X)
    _b = torch.mul(_b, 1-X)

    return (_f, _b)