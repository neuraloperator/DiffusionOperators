import torch
import torch.fft as fft
import numpy as np
import cv2
import math

#Gaussian random fields with Matern-type covariance: C = sigma^2 (-Lap + tau^2 I)^-alpha
#Generates random field samples on the domain [0,L1] x [0,L2]

class PeriodicGaussianRF2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, alpha=2.0, tau=3.0, sigma=None, mean=None, device=None, dtype=torch.float32):

        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - 2.0))

        const1 = (4*(math.pi**2))/(L1**2)
        const2 = (4*(math.pi**2))/(L2**2)
        norm_const = math.sqrt(2.0/(L1*L2))

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2).type(dtype).to(device)

        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1),\
                                torch.arange(start=-s2//2, end=0, step=1)), 0)

        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig = s1*s2*sigma*norm_const*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0,0] = 0.0
        self.sqrt_eig[torch.logical_and(k1 + k2 <= 0.0, torch.logical_or(k1 + k2 != 0.0, k1 <= 0.0))] = 0.0

    def sample(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s1, self.s2, 2, dtype=self.dtype, device=self.device)
        
        xi[...,0] = self.sqrt_eig*xi [...,0]
        xi[...,1] = self.sqrt_eig*xi [...,1]
        
        u = fft.ifft2(torch.view_as_complex(xi), s=(self.s1, self.s2)).imag

        if self.mean is not None:
            u += self.mean
        
        return u


class GaussianRF_idct(object):
    """
    Gaussian random field Non-Periodic Boundary
    mean 0
    covariance operator C = (-Delta + tau^2)^(-alpha)
    Delta is the Laplacian with zero Neumann boundary condition
    """

    def __init__(self, Ln1, Ln2, alpha=2.0, tau=3.0, sigma = 1, device=None):
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.sigma = sigma
        k1 = np.arange(Ln1)
        k2 = np.arange(Ln2)
        K1,K2 = np.meshgrid(k1,k2)
        # Define the (square root of) eigenvalues of the covariance operator
        C = (np.pi**2)*(np.square(K1)+np.square(K2))+tau**2
        C = np.power(C,-alpha/2.0)
        C = (tau**(alpha-1))*C
        # store coefficient
        self.coeff = C

    def sample(self, N, mul=1):
        z_mat = np.zeros((N,self.Ln1, self.Ln2, 2), dtype=np.float32)
        for ix in range(N):
            z_mat[ix,:,:,:] = self._sample2d()
        # convert to torch tensor
        z_mat = torch.from_numpy(z_mat)
        if self.device is not None:
            z_mat = z_mat.to(self.device)
        return z_mat * self.sigma

    def _sample2d(self):
        """
        Single 2D Sample
        :return: GRF numpy.narray (Ln,Ln)
        """
        # # sample from normal discribution
        xr = np.random.standard_normal(size=(self.Ln1,self.Ln2,2))
        # coefficients in fourier domain
        L= np.einsum('ij,ijk->ijk', self.coeff, xr) 
        L= (self.Ln1*self.Ln1)**(1/2)*L
        # apply boundary condition
        L[0,0,:] = 0.0 * L[0,0,:]
        # transform to real domain
        L[:,:,0] = cv2.idct(L[:,:,0])
        L[:,:,1] = cv2.idct(L[:,:,1])
        return L

class IndependentGaussian(object):
    """

    """

    def __init__(self, Ln1, Ln2, sigma=1, device=None):
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.sigma = sigma

    def sample(self, N):
        z = torch.randn((N, self.Ln1, self.Ln2, 2)).normal_(0, self.sigma)
        if self.device is not None:
            z = z.to(self.device)
        return z