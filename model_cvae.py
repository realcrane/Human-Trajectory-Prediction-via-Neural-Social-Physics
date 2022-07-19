import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pdb
from torch.nn import functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
import yaml

'''MLP model'''
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class CVAE(nn.Module):

    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, fdim, zdim, sigma, past_length, future_length, verbose):
        '''
        Args:
            size parameters: Dimension sizes
            nonlocal_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(CVAE, self).__init__()

        self.zdim = zdim
        #self.nonlocal_pools = nonlocal_pools
        self.sigma = sigma

        # takes in the past
        self.encoder_past = MLP(input_dim = past_length*2, output_dim = fdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim = 2, output_dim = fdim, hidden_size=enc_dest_size)

        self.encoder_latent = MLP(input_dim = 2*fdim, output_dim = 2*zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim = fdim + zdim, output_dim = 2, hidden_size=dec_size)

        # self.non_local_theta = MLP(input_dim = 2*fdim + 2, output_dim = non_local_dim, hidden_size=non_local_theta_size)
        # self.non_local_phi = MLP(input_dim = 2*fdim + 2, output_dim = non_local_dim, hidden_size=non_local_phi_size)
        # self.non_local_g = MLP(input_dim = 2*fdim + 2, output_dim = 2*fdim + 2, hidden_size=non_local_g_size)



    def non_local_social_pooling(self, feat, mask):

        # N,C
        theta_x = self.non_local_theta(feat)

        # C,N
        phi_x = self.non_local_phi(feat).transpose(1,0)

        # f_ij = (theta_i)^T(phi_j), (N,N)
        f = torch.matmul(theta_x, phi_x)

        # f_weights_i =  exp(f_ij)/(\sum_{j=1}^N exp(f_ij))
        f_weights = F.softmax(f, dim = -1)

        # setting weights of non neighbours to zero
        f_weights = f_weights * mask

        # rescaling row weights to 1
        f_weights = F.normalize(f_weights, p=1, dim=1)

        # ith row of all_pooled_f = \sum_{j=1}^N f_weights_i_j * g_row_j
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

        return pooled_f + feat

    def forward(self, x, next_step = None, device=torch.device('cpu')):

        # provide destination iff training
        # assert model.training
        # encode
        ftraj = self.encoder_past(x) #513*16

        if not self.training:
            z = torch.Tensor(x.size(0), self.zdim)
            z.normal_(0, self.sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            ns_features = self.encoder_dest(next_step) #513*16
            features = torch.cat((ftraj, ns_features), dim = 1) #513*32
            latent =  self.encoder_latent(features)#513*32 mean+var

            mu = latent[:, 0:self.zdim] # 2-d array 513*16
            logvar = latent[:, self.zdim:] # 2-d array 513*16

            var = logvar.mul(0.5).exp_() #513*16
            eps = torch.DoubleTensor(var.size()).normal_()#513*16
            eps = eps.to(device)
            z = eps.mul(var).add_(mu) #513*16

        z = z.double().to(device)
        decoder_input = torch.cat((ftraj, z), dim = 1) #516*32
        generated_np = self.decoder(decoder_input) #516*2

        if self.training:
            # prediction in training, no best selection
            # generated_dest_features = self.encoder_dest(generated_dest)
            #
            # prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim = 1)
            #
            # for i in range(self.nonlocal_pools):
            #     # non local social pooling
            #     prediction_features = self.non_local_social_pooling(prediction_features, mask)

            #pred_future = self.predictor(prediction_features)
            return generated_np, mu, logvar

        return generated_np

    # separated for forward to let choose the best destination
    def predict(self, past, generated_dest, mask, initial_pos):
        ftraj = self.encoder_past(past)
        generated_dest_features = self.encoder_dest(generated_dest)
        prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim = 1)

        for i in range(self.nonlocal_pools):
            # non local social pooling
            prediction_features = self.non_local_social_pooling(prediction_features, mask)

        interpolated_future = self.predictor(prediction_features)
        return interpolated_future
