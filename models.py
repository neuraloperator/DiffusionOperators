import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy

# from neuralop.models.tfno import FactorizedFNO1d, FactorizedFNO2d
from time_embedding import TimestepEmbedding

class FNO(nn.Module):
    def __init__(self, s=64, modes=32, width=64, embed_dim=512):
        super().__init__()

        self.fno = FactorizedFNO1d(in_channels=3, width=width, modes_height=modes, factorization=None)
        self.time = TimestepEmbedding(embed_dim, s, s)

        pos_embed = torch.linspace(0, 1, s+1)[0:-1].view(1,1,-1)
        self.register_buffer('pos_embed', pos_embed)
    
    def forward(self, x, sigma):
        bsize = x.size(0)

        x = x.unsqueeze(1)
        time_embed = self.time(sigma).view(1,1,-1).repeat(bsize,1,1)
        pos_embed = self.pos_embed.repeat(bsize,1,1)

        x = torch.cat((x, pos_embed, time_embed), 1)


        return self.fno(x).squeeze(1)

class InterpModule(nn.Module):
    def __init__(self, module, s):
        super().__init__()

        self.module = module
        self.s = s
    
    def forward(self, x):
        x = self.module(x).unsqueeze(1)

        return F.interpolate(x, size=self.s, mode='linear')

class InterpModel(nn.Module):
    def __init__(self, model, s):
        super().__init__()

        self.model = model

        pos_embed = torch.linspace(0, 1, s+1)[0:-1].view(1,1,-1)
        self.model.pos_embed = pos_embed

        self.old_time = copy.deepcopy(self.model.time)
        self.model.time = InterpModule(self.old_time, s)
    
    def forward(self, x, sigma):
        return self.model(x, sigma)


class FNO2d(nn.Module):
    def __init__(self, s=64, modes=32, width=64, out_channels = 1, in_channels = 1, embed_dim=512):
        super().__init__()

        self.s = s

        self.fno = FactorizedFNO2d(in_channels=in_channels+3, out_channels=out_channels, width=width, modes_height=modes, modes_width=modes, factorization=None)
        self.time = TimestepEmbedding(embed_dim, 2*s, s**2, pos_dim=1)

        t = torch.linspace(0, 1, s+1)[0:-1]
        X, Y = torch.meshgrid(t, t, indexing='ij') 

        self.register_buffer('pos_embed_x', X)
        self.register_buffer('pos_embed_y', Y)
    
    def forward(self, x, sigma):
        bsize = x.size(0)

       # x = x.unsqueeze(1)
        x = torch.permute(x, (0, 3, 1, 2))
        time_embed = self.time(sigma).view(1,1,self.s, self.s).repeat(bsize,1,1,1)
        pos_embed_x = self.pos_embed_x.repeat(bsize,1,1,1)
        pos_embed_y = self.pos_embed_y.repeat(bsize,1,1,1)

        x = torch.cat((x, pos_embed_x, pos_embed_y, time_embed), 1)


        return torch.permute(self.fno(x).squeeze(1),(0, 2, 3, 1))






class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dim1, dim2,modes1 = None, modes2 = None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim1 = dim1 #output dimensions
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2-1 #if not given take the highest number of modes can be taken
            self.modes2 = dim2//2 
        self.scale = (1 / (2*in_channels))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1 = None,dim2 = None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1, self.dim2//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2))
        return x


class pointwise_op(nn.Module):
    def __init__(self, in_channel, out_channel,dim1, dim2):
        super(pointwise_op,self).__init__()
        self.conv = nn.Conv2d(int(in_channel), int(out_channel), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self,x, dim1 = None, dim2 = None):
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True)
        return x_out

class UNO(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, s , pad = 0, factor = 3/4, embed_dim=512):
        super(UNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.s = s
        #self.time = TimestepEmbedding(embed_dim, 2*self.s, self.s**2, pos_dim=1)
        self.in_d_co_domain = in_d_co_domain # input channel
        self.d_co_domain = d_co_domain 
        self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(self.in_d_co_domain, self.d_co_domain) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.d_co_domain, 2*factor*self.d_co_domain, 48, 48, 24, 24)

        self.conv1 = SpectralConv2d(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32, 16,16)

        self.conv2 = SpectralConv2d(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16,8,8)
        
        self.conv2_1 = SpectralConv2d(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8,4,4)
        
        self.conv2_9 = SpectralConv2d(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16,4,4)
        

        self.conv3 = SpectralConv2d(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32,8,8)

        self.conv4 = SpectralConv2d(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, 48, 48,16,16)

        self.conv5 = SpectralConv2d(4*factor*self.d_co_domain, self.d_co_domain, 64, 64,24,24) # will be reshaped

        self.w0 = pointwise_op(self.d_co_domain,2*factor*self.d_co_domain,48, 48) #
        
        self.w1 = pointwise_op(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32) #
        
        self.w2 = pointwise_op(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16) #
        
        self.w2_1 = pointwise_op(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8)
        
        self.w2_9 = pointwise_op(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16)
        
        self.w3 = pointwise_op(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32) #
        
        self.w4 = pointwise_op(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, 48, 48)
        
        self.w5 = pointwise_op(4*factor*self.d_co_domain, self.d_co_domain, 64, 64) # will be reshaped

        self.fc1 = nn.Linear(2*self.d_co_domain, 4*self.d_co_domain)
        self.fc2 = nn.Linear(4*self.d_co_domain, 2)

    def forward(self, x, sigmas):
        """
        Args:
          x: of shape (bs, res, res, 2)
          sigma: of shape (1,1)

        Input is preprocessed so that x is transformed into the
          shape (bs, res, res, 5), where 2 comes from the grid
          and 1 comes from the time embedding.
        """

        # have a different time embedding per minibatch
        # + as suggested in improved techniques paper, just redefine
        # s(x,sigma) = s(x) / sigma instead.
        
        bsize = x.size(0)
        grid = self.get_grid(x.shape, x.device)
        # print('time_size',self.time(sigma).view(1,self.s, self.s,1).size())
        #time_embed = self.time(sigma).view(1,self.s, self.s,1).repeat(bsize,1,1,1)
        
        x = torch.cat((x, grid), dim=-1)

        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        

        x1_c0 = self.conv0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0 ,D1//2,D2//2)
        x2_c1 = self.w1(x_c0 ,D1//2,D2//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1 ,D1//4,D2//4)
        x2_c2 = self.w2(x_c1 ,D1//4,D2//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )
        #print(x.shape)
        
        x1_c2_1 = self.conv2_1(x_c2,D1//8,D2//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8,D2//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)
        
        x1_c2_9 = self.conv2_9(x_c2_1,D1//4,D2//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4,D2//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1) 

        x1_c3 = self.conv3(x_c2_9,D1//2,D2//2)
        x2_c3 = self.w3(x_c2_9,D1//2,D2//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x2_c4 = self.w4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1,D2)
        x2_c5 = self.w5(x_c4,D1,D2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        

        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]

        x_c5 = x_c5.permute(0, 2, 3, 1)
        
        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)
        
        x_out = torch.tanh(x_out)
        
        return x_out / sigmas.view(-1, 1, 1, 1)
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
