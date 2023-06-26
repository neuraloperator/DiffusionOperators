"""
MIT License
Copyright (c) 2022 Steven Cheng-Xian Li
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from mpmath import *
import numpy as np
import torch

mp.dps = 32
one = mpf(1)
mp.pretty = True

def f(x):
    return sqrt(one-x)

# Derive the taylor and pade' coefficients for MTP, MPA
a = taylor(f, 0, 10)
pade_p, pade_q = pade(a, 5, 5)
a = torch.from_numpy(np.array(a).astype(float))
pade_p = torch.from_numpy(np.array(pade_p).astype(float))
pade_q = torch.from_numpy(np.array(pade_q).astype(float))

def matrix_taylor_polynomial(p, I):
    p_sqrt= I
    p_app = I - p
    p_hat = p_app
    for i in range(10):
      p_sqrt += a[i+1]*p_hat
      p_hat = p_hat.bmm(p_app)
    return p_sqrt

def matrix_pade_approximant(p,I):
    p_sqrt = pade_p[0]*I
    q_sqrt = pade_q[0]*I
    p_app = I - p
    p_hat = p_app
    for i in range(5):
        p_sqrt += pade_p[i+1]*p_hat
        q_sqrt += pade_q[i+1]*p_hat
        p_hat = p_hat.bmm(p_app)
    #There are 4 options to compute the MPA: comput Matrix Inverse or Matrix Linear System on CPU/GPU;
    #It seems that single matrix is faster on CPU and batched matrices are faster on GPU
    #Please check which one is faster before running the code;
    return torch.linalg.solve(q_sqrt, p_sqrt)
    #return torch.linalg.solve(q_sqrt.cpu(), p_sqrt.cpu()).cuda()
    #return torch.linalg.inv(q_sqrt).mm(p_sqrt)
    #return torch.linalg.inv(q_sqrt.cpu()).cuda().bmm(p_sqrt)

def matrix_pade_approximant_inverse(p,I):
    p_sqrt = pade_p[0]*I
    q_sqrt = pade_q[0]*I
    p_app = I - p
    p_hat = p_app
    for i in range(5):
        p_sqrt += pade_p[i+1]*p_hat
        q_sqrt += pade_q[i+1]*p_hat
        p_hat = p_hat.bmm(p_app)
    #There are 4 options to compute the MPA_inverse: comput Matrix Inverse or Matrix Linear System on CPU/GPU;
    #It seems that single matrix is faster on CPU and batched matrices are faster on GPU
    #Please check which one is faster before running the code;
    return torch.linalg.solve(p_sqrt, q_sqrt)
    #return torch.linalg.solve(p_sqrt.cpu(), q_sqrt.cpu()).cuda()
    #return torch.linalg.inv(p_sqrt).mm(q_sqrt)
    #return torch.linalg.inv(p_sqrt.cpu()).cuda().bmm(q_sqrt)

#Differentiable Matrix Square Root by MPA_Lya
class MPA_Lya(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        normM = torch.norm(M,dim=[1,2]).reshape(M.size(0),1,1)
        I = torch.eye(M.size(1), requires_grad=False, device=M.device).reshape(1,M.size(1),M.size(1)).repeat(M.size(0),1,1)
        #This is for MTP calculation
        #M_sqrt = matrix_taylor_polynomial(M/normM,I)
        M_sqrt = matrix_pade_approximant(M / normM, I)
        M_sqrt = M_sqrt * torch.sqrt(normM)
        ctx.save_for_backward(M, M_sqrt, normM,  I)
        return M_sqrt

    @staticmethod
    def backward(ctx, grad_output):
        M, M_sqrt, normM,  I = ctx.saved_tensors
        b = M_sqrt / torch.sqrt(normM)
        c = grad_output / torch.sqrt(normM)
        for i in range(8):
            #In case you might terminate the iteration by checking convergence
            #if th.norm(b-I)<1e-4:
            #    break
            b_2 = b.bmm(b)
            c = 0.5 * (c.bmm(3.0*I-b_2)-b_2.bmm(c)+b.bmm(c).bmm(b))
            b = 0.5 * b.bmm(3.0 * I - b_2)
        grad_input = 0.5 * c
        return grad_input

#Differentiable Inverse Square Root by MPA_Lya_Inv
class MPA_Lya_Inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        normM = torch.norm(M,dim=[1,2]).reshape(M.size(0),1,1)
        I = torch.eye(M.size(1), requires_grad=False, device=M.device).reshape(1,M.size(1),M.size(1)).repeat(M.size(0),1,1)
        #M_sqrt = matrix_taylor_polynomial(M/normM,I)
        M_sqrt_inv = matrix_pade_approximant_inverse(M / normM, I)
        M_sqrt_inv = M_sqrt_inv / torch.sqrt(normM)
        ctx.save_for_backward(M, M_sqrt_inv,  I)
        return M_sqrt_inv

    @staticmethod
    def backward(ctx, grad_output):
        M, M_sqrt_inv,  I = ctx.saved_tensors
        M_inv = M_sqrt_inv.bmm(M_sqrt_inv)
        grad_lya = - M_inv.bmm(grad_output).bmm(M_inv)
        norm_sqrt_inv = torch.norm(M_sqrt_inv)
        b = M_sqrt_inv / norm_sqrt_inv
        c = grad_lya / norm_sqrt_inv
        for i in range(8):
            #In case you might terminate the iteration by checking convergence
            #if th.norm(b-I)<1e-4:
            #    break
            b_2 = b.bmm(b)
            c = 0.5 * (c.bmm(3.0 * I - b_2) - b_2.bmm(c) + b.bmm(c).bmm(b))
            b = 0.5 * b.bmm(3.0 * I - b_2)
        grad_input = 0.5 * c
        return grad_input


if __name__ == '__main__':

    FastMatSqrt = MPA_Lya.apply
    FastInvSqrt = MPA_Lya_Inv.apply

    # For any batched matrices, compute their square root or inverse square root:
    rand_matrix = torch.randn(1,32,32)
    cov = rand_matrix.bmm(rand_matrix.transpose(1,2))
    
    cov_sqrt = FastMatSqrt(cov)
    cov_inv_sqrt = FastInvSqrt(cov)

    cov_inv = torch.linalg.inv(cov)

    # So surely inv_sqrt@inv_sqrt should be == cov_inv
    #import pdb; pdb.set_trace()

    