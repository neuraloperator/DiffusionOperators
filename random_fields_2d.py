import torch
import torch.fft as fft
import numpy as np
import cv2
import math
import os

from setup_logger import get_logger

logger = get_logger(__name__)

from math_utils import MPA_Lya, MPA_Lya_Inv

FastMatSqrt = MPA_Lya.apply
FastInvSqrt = MPA_Lya_Inv.apply


class PeriodicGaussianRF2d(object):
    def __init__(
        self,
        s1,
        s2,
        L1=2 * math.pi,
        L2=2 * math.pi,
        alpha=2.0,
        tau=3.0,
        sigma=None,
        mean=None,
        device=None,
        dtype=torch.float32,
    ):

        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - 2.0))

        const1 = (4 * (math.pi**2)) / (L1**2)
        const2 = (4 * (math.pi**2)) / (L2**2)
        norm_const = math.sqrt(2.0 / (L1 * L2))

        freq_list1 = torch.cat(
            (
                torch.arange(start=0, end=s1 // 2, step=1),
                torch.arange(start=-s1 // 2, end=0, step=1),
            ),
            0,
        )
        k1 = freq_list1.view(-1, 1).repeat(1, s2).type(dtype).to(device)

        freq_list2 = torch.cat(
            (
                torch.arange(start=0, end=s2 // 2, step=1),
                torch.arange(start=-s2 // 2, end=0, step=1),
            ),
            0,
        )

        k2 = freq_list2.view(1, -1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig = (
            s1
            * s2
            * sigma
            * norm_const
            * ((const1 * k1**2 + const2 * k2**2 + tau**2) ** (-alpha / 2.0))
        )
        self.sqrt_eig[0, 0] = 0.0
        self.sqrt_eig[
            torch.logical_and(
                k1 + k2 <= 0.0, torch.logical_or(k1 + k2 != 0.0, k1 <= 0.0)
            )
        ] = 0.0

    def sample(self, N, xi=None):
        if xi is None:
            xi = torch.randn(
                N, self.s1, self.s2, 2, dtype=self.dtype, device=self.device
            )

        xi[..., 0] = self.sqrt_eig * xi[..., 0]
        xi[..., 1] = self.sqrt_eig * xi[..., 1]

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

    def __init__(self, Ln1, Ln2, alpha=2.0, tau=3.0, sigma=1, device=None):
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.sigma = sigma
        k1 = np.arange(Ln1)
        k2 = np.arange(Ln2)
        K1, K2 = np.meshgrid(k1, k2)
        # Define the (square root of) eigenvalues of the covariance operator
        C = (np.pi**2) * (np.square(K1) + np.square(K2)) + tau**2
        C = np.power(C, -alpha / 2.0)
        C = (tau ** (alpha - 1)) * C
        # store coefficient
        self.coeff = C

    def sample(self, N, mul=1):
        z_mat = np.zeros((N, self.Ln1, self.Ln2, 2), dtype=np.float32)
        for ix in range(N):
            z_mat[ix, :, :, :] = self._sample2d()
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
        xr = np.random.standard_normal(size=(self.Ln1, self.Ln2, 2))
        # coefficients in fourier domain
        L = np.einsum("ij,ijk->ijk", self.coeff, xr)
        L = (self.Ln1 * self.Ln1) ** (1 / 2) * L
        # apply boundary condition
        L[0, 0, :] = 0.0 * L[0, 0, :]
        # transform to real domain
        L[:, :, 0] = cv2.idct(L[:, :, 0])
        L[:, :, 1] = cv2.idct(L[:, :, 1])
        return L


def get_fixed_coords(Ln1, Ln2):
    xs = torch.linspace(0, 1, steps=Ln1 + 1)[0:-1]
    ys = torch.linspace(0, 1, steps=Ln2 + 1)[0:-1]
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.cat([yy.reshape(-1, 1), xx.reshape(-1, 1)], dim=-1)
    return coords


class GaussianRF_RBF(object):
    """ """

    @torch.no_grad()
    def __init__(
        self, Ln1, Ln2, scale=1, eps=0.01, fast_sqrt=False, device=None, cached=True
    ):
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.scale = scale

        # (s^2, 2)
        meshgrid = get_fixed_coords(self.Ln1, self.Ln2)
        # (s^2, s^2)
        C = torch.exp(-torch.cdist(meshgrid, meshgrid) / (2 * scale**2)).to(device)
        # Need to add some regularisation or else the sqrt won't exist
        I = torch.eye(C.size(-1)).to(device)
        self.C = C + (eps**2) * I

        self.L = torch.linalg.cholesky(self.C).to(device)

    @torch.no_grad()
    def sample(self, N):
        # (N, s^2, s^2) x (N, s^2, 1) -> (N, s^2, 2)
        # We can do this in one big torch.bmm, but I am concerned about memory
        # so let's just do it iteratively.
        # L_padded = self.L.repeat(N, 1, 1)
        # z_mat = torch.randn((N, self.Ln1*self.Ln2, 2)).to(self.device)
        # sample = torch.bmm(L_padded, z_mat)
        samples = torch.zeros((N, self.Ln1 * self.Ln2, 2)).to(self.device)
        for ix in range(N):
            # (s^2, s^2) * (s^2, 2) -> (s^2, 2)
            this_z = torch.randn(self.Ln1 * self.Ln2, 2).to(self.device)
            samples[ix] = torch.matmul(self.L, this_z)

        # reshape into (N, s, s, 2)
        sample_rshp = samples.reshape(-1, self.Ln1, self.Ln2, 2)

        return sample_rshp


class IndependentGaussian(object):
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
