import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pdb
from torch.distributions.normal import Normal
import math
import numpy as np
import copy

def environment(current_step, first_frame, current_vel, semantic_map, k_scope, k_env, k_label_4, F0, device):
    ori_step = current_step + first_frame
    scope_point = ori_step + torch.sign(current_vel) * k_scope
    area = torch.floor(torch.stack((ori_step, scope_point), dim=2)).int()
    F2 = torch.zeros_like(F0)
    for i in range(F2.shape[0]):
        if area[i, 0, 0] == area[i, 0, 1] and area[i, 1, 0] == area[i, 1, 1]:
            continue

        if area[i, 0, 0] == area[i, 0, 1] and area[i, 1, 0] != area[i, 1, 1]:
            environment_vision = semantic_map[area[i, 0, 0],
                                 torch.min(area[i, 1, 0], area[i, 1, 1]): torch.max(area[i, 1, 0], area[i, 1, 1])]
            if len(np.argwhere(environment_vision == 5)) == 0:
                continue
            obstacle = torch.from_numpy(np.mean(np.argwhere(environment_vision == 5), axis=0)).to(device)  # 1
            if area[i, 1, 0] < area[i, 1, 1]:
                dis = torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] = (k_env / dis) * torch.tensor([0, -1]).to(device)
            else:
                dis = k_scope - torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] = (k_env / dis) * torch.tensor([0, 1]).to(device)
            continue

        if area[i, 0, 0] != area[i, 0, 1] and area[i, 1, 0] == area[i, 1, 1]:
            environment_vision = semantic_map[
                                 torch.min(area[i, 0, 0], area[i, 0, 1]): torch.max(area[i, 0, 0], area[i, 0, 1]),
                                 area[i, 1, 0]]
            if len(np.argwhere(environment_vision == 5)) == 0:
                continue
            obstacle = torch.from_numpy(np.mean(np.argwhere(environment_vision == 5), axis=0)).to(device)  # 1
            if area[i, 0, 0] < area[i, 0, 1]:
                dis = torch.norm(obstacle) + 1
                if dis == 0:
                    continue
                F2[i, :] = (k_env / dis) * torch.tensor([-1, 0]).to(device)
            else:
                dis = k_scope - torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] = (k_env / dis) * torch.tensor([1, 0]).to(device)
            continue

        environment_vision = semantic_map[
                             torch.min(area[i, 0, 0], area[i, 0, 1]): torch.max(area[i, 0, 0], area[i, 0, 1]),
                             torch.min(area[i, 1, 0], area[i, 1, 1]): torch.max(area[i, 1, 0], area[i, 1, 1])]
        if len(np.argwhere(environment_vision == 5)) == 0:
            continue
        obstacle = torch.from_numpy(np.mean(np.argwhere(environment_vision == 5), axis=0)).to(device)  # 2
        if area[i, 0, 0] < area[i, 0, 1] and area[i, 1, 0] < area[i, 1, 1]:
            dis = torch.norm(obstacle)
            if dis == 0:
                continue
            F2[i, :] = -(k_env / dis) * (obstacle / dis)
        if area[i, 0, 0] < area[i, 0, 1] and area[i, 1, 0] > area[i, 1, 1]:
            dis = torch.norm(obstacle - torch.tensor([0, k_scope]).to(device))
            if dis == 0:
                continue
            F2[i, :] = (k_env / dis) * ((torch.tensor([0, k_scope]).to(device) - obstacle) / dis)
        if area[i, 0, 0] > area[i, 0, 1] and area[i, 1, 0] < area[i, 1, 1]:
            dis = torch.norm(obstacle - torch.tensor([k_scope, 0]).to(device))
            if dis == 0:
                continue
            F2[i, :] = (k_env / dis) * ((torch.tensor([k_scope, 0]).to(device) - obstacle) / dis)
        if area[i, 0, 0] > area[i, 0, 1] and area[i, 1, 0] > area[i, 1, 1]:
            dis = torch.norm(obstacle - torch.tensor([k_scope, k_scope]).to(device))
            if dis == 0:
                continue
            F2[i, :] = (k_env / dis) * ((torch.tensor([k_scope, k_scope]).to(device) - obstacle) / dis)

    for i in range(F2.shape[0]):
        if area[i, 0, 0] == area[i, 0, 1] and area[i, 1, 0] == area[i, 1, 1]:
            continue

        if area[i, 0, 0] == area[i, 0, 1] and area[i, 1, 0] != area[i, 1, 1]:
            environment_vision = semantic_map[area[i, 0, 0],
                                 torch.min(area[i, 1, 0], area[i, 1, 1]): torch.max(area[i, 1, 0], area[i, 1, 1])]
            if len(np.argwhere(environment_vision == 3)) == 0:
                continue
            obstacle = torch.from_numpy(np.mean(np.argwhere(environment_vision == 3), axis=0)).to(device)  # 1
            if area[i, 1, 0] < area[i, 1, 1]:
                dis = torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] += (k_env / dis) * torch.tensor([0, -1]).to(device)
            else:
                dis = k_scope - torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] += (k_env / dis) * torch.tensor([0, 1]).to(device)
            continue

        if area[i, 0, 0] != area[i, 0, 1] and area[i, 1, 0] == area[i, 1, 1]:
            environment_vision = semantic_map[
                                 torch.min(area[i, 0, 0], area[i, 0, 1]): torch.max(area[i, 0, 0], area[i, 0, 1]),
                                 area[i, 1, 0]]
            if len(np.argwhere(environment_vision == 3)) == 0:
                continue
            obstacle = torch.from_numpy(np.mean(np.argwhere(environment_vision == 3), axis=0)).to(device)  # 1
            if area[i, 0, 0] < area[i, 0, 1]:
                dis = torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] += (k_env / dis) * torch.tensor([-1, 0]).to(device)
            else:
                dis = k_scope - torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] += (k_env / dis) * torch.tensor([1, 0]).to(device)
            continue

        environment_vision = semantic_map[
                             torch.min(area[i, 0, 0], area[i, 0, 1]): torch.max(area[i, 0, 0], area[i, 0, 1]),
                             torch.min(area[i, 1, 0], area[i, 1, 1]): torch.max(area[i, 1, 0], area[i, 1, 1])]
        if len(np.argwhere(environment_vision == 3)) == 0:
            continue
        obstacle = torch.from_numpy(np.mean(np.argwhere(environment_vision == 3), axis=0)).to(device)  # 2
        if area[i, 0, 0] < area[i, 0, 1] and area[i, 1, 0] < area[i, 1, 1]:
            dis = torch.norm(obstacle)
            if dis == 0:
                continue
            F2[i, :] += -(k_env / dis) * (obstacle / dis)
        if area[i, 0, 0] < area[i, 0, 1] and area[i, 1, 0] > area[i, 1, 1]:
            dis = torch.norm(obstacle - torch.tensor([0, k_scope]).to(device))
            if dis == 0:
                continue
            F2[i, :] += (k_env / dis) * ((torch.tensor([0, k_scope]).to(device) - obstacle) / dis)
        if area[i, 0, 0] > area[i, 0, 1] and area[i, 1, 0] < area[i, 1, 1]:
            dis = torch.norm(obstacle - torch.tensor([k_scope, 0]).to(device))
            if dis == 0:
                continue
            F2[i, :] += (k_env / dis) * ((torch.tensor([k_scope, 0]).to(device) - obstacle) / dis)
        if area[i, 0, 0] > area[i, 0, 1] and area[i, 1, 0] > area[i, 1, 1]:
            dis = torch.norm(obstacle - torch.tensor([k_scope, k_scope]).to(device))
            if dis == 0:
                continue
            F2[i, :] += (k_env / dis) * ((torch.tensor([k_scope, k_scope]).to(device) - obstacle) / dis)

    for i in range(F2.shape[0]):
        if area[i, 0, 0] == area[i, 0, 1] and area[i, 1, 0] == area[i, 1, 1]:
            continue

        if area[i, 0, 0] == area[i, 0, 1] and area[i, 1, 0] != area[i, 1, 1]:
            environment_vision = semantic_map[area[i, 0, 0],
                                 torch.min(area[i, 1, 0], area[i, 1, 1]): torch.max(area[i, 1, 0], area[i, 1, 1])]
            if len(np.argwhere(environment_vision == 4)) == 0:
                continue
            obstacle = torch.from_numpy(np.mean(np.argwhere(environment_vision == 4), axis=0)).to(device)  # 1
            if area[i, 1, 0] < area[i, 1, 1]:
                dis = torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] += k_label_4 * (k_env / dis) * torch.tensor([0, -1]).to(device)
            else:
                dis = k_scope - torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] += k_label_4 * (k_env / dis) * torch.tensor([0, 1]).to(device)
            continue

        if area[i, 0, 0] != area[i, 0, 1] and area[i, 1, 0] == area[i, 1, 1]:
            environment_vision = semantic_map[
                                 torch.min(area[i, 0, 0], area[i, 0, 1]): torch.max(area[i, 0, 0], area[i, 0, 1]),
                                 area[i, 1, 0]]
            if len(np.argwhere(environment_vision == 4)) == 0:
                continue
            obstacle = torch.from_numpy(np.mean(np.argwhere(environment_vision == 4), axis=0)).to(device)  # 1
            if area[i, 0, 0] < area[i, 0, 1]:
                dis = torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] += k_label_4 * (k_env / dis) * torch.tensor([-1, 0]).to(device)
            else:
                dis = k_scope - torch.norm(obstacle)
                if dis == 0:
                    continue
                F2[i, :] += k_label_4 * (k_env / dis) * torch.tensor([1, 0]).to(device)
            continue

        environment_vision = semantic_map[
                             torch.min(area[i, 0, 0], area[i, 0, 1]): torch.max(area[i, 0, 0], area[i, 0, 1]),
                             torch.min(area[i, 1, 0], area[i, 1, 1]): torch.max(area[i, 1, 0], area[i, 1, 1])]
        if len(np.argwhere(environment_vision == 4)) == 0:
            continue
        obstacle = torch.from_numpy(np.mean(np.argwhere(environment_vision == 4), axis=0)).to(device)  # 2
        if area[i, 0, 0] < area[i, 0, 1] and area[i, 1, 0] < area[i, 1, 1]:
            dis = torch.norm(obstacle)
            if dis == 0:
                continue
            F2[i, :] += -k_label_4 * (k_env / dis) * (obstacle / dis)
        if area[i, 0, 0] < area[i, 0, 1] and area[i, 1, 0] > area[i, 1, 1]:
            dis = torch.norm(obstacle - torch.tensor([0, k_scope]).to(device))
            if dis == 0:
                continue
            F2[i, :] += k_label_4 * (k_env / dis) * ((torch.tensor([0, k_scope]).to(device) - obstacle) / dis)
        if area[i, 0, 0] > area[i, 0, 1] and area[i, 1, 0] < area[i, 1, 1]:
            dis = torch.norm(obstacle - torch.tensor([k_scope, 0]).to(device))
            if dis == 0:
                continue
            F2[i, :] += k_label_4 * (k_env / dis) * ((torch.tensor([k_scope, 0]).to(device) - obstacle) / dis)
        if area[i, 0, 0] > area[i, 0, 1] and area[i, 1, 0] > area[i, 1, 1]:
            dis = torch.norm(obstacle - torch.tensor([k_scope, k_scope]).to(device))
            if dis == 0:
                continue
            F2[i, :] += k_label_4 * (k_env / dis) * ((torch.tensor([k_scope, k_scope]).to(device) - obstacle) / dis)
    return F2
def stateutils_desired_directions(current_step, generated_dest):

    destination_vectors = generated_dest - current_step #peds*2
    norm_factors = torch.norm(destination_vectors, dim=-1) #peds
    norm_factors = torch.unsqueeze(norm_factors, dim=-1)
    directions = destination_vectors / (norm_factors + 1e-8) #peds*2
    return directions
def f_ab_fun(current_step, coefficients, current_supplement, sigma, device):
    # disp_p_x = torch.zeros(1,2)
    # disp_p_y = torch.zeros(1,2)
    # disp_p_x[0, 0] = 0.1
    # disp_p_y[0, 1] = 0.1

    c1 = current_supplement[:,:-1,:2] #peds*maxpeds*2
    pedestrians = torch.unsqueeze(current_step, dim=1)  # peds*1*2

    v = value_p_p(c1, pedestrians, coefficients, sigma) # peds

    delta = torch.tensor(1e-3).to(device)
    dx = torch.tensor([[[delta, 0.0]]]).to(device) #1*1*2
    dy = torch.tensor([[[0.0, delta]]]).to(device) #1*1*2

    dvdx = (value_p_p(c1, pedestrians + dx, coefficients, sigma) - v) / delta # peds
    dvdy = (value_p_p(c1, pedestrians + dy, coefficients, sigma) - v) / delta # peds

    grad_r_ab = torch.stack((dvdx, dvdy), dim=-1) # peds*2
    out = -1.0 * grad_r_ab

    return out
def value_p_p(c1, pedestrians, coefficients, sigma):
    #potential field function : pf = K*exp(-norm(p-p1))

    d_p_c1 = pedestrians - c1  # peds*maxpeds*2
    d_p_c1_norm = torch.norm(d_p_c1, dim=-1) # peds*maxpeds

    potential = sigma * coefficients * torch.exp(-d_p_c1_norm/sigma) #peds*maxpeds

    out = torch.sum(potential, 1) #peds

    return out

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

class NSP(nn.Module):

    def __init__(self, input_size, embedding_size, rnn_size, output_size, enc_size, dec_size):
        '''
        Args:
            size parameters: Dimension sizes
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(NSP, self).__init__()

        self.max_peds = 25
        self.r_pixel = 100
        self.costheta = np.cos(np.pi / 3)

        # The Goal-Network
        self.cell1 = nn.LSTMCell(embedding_size, rnn_size)
        self.input_embedding_layer1 = nn.Linear(input_size, embedding_size)
        self.output_layer1 = nn.Linear(rnn_size, output_size)

        self.encoder_dest_state = MLP(input_dim = 2, output_dim = output_size, hidden_size=enc_size)
        self.dec_tau = MLP(input_dim = 2*output_size, output_dim = 1, hidden_size=dec_size)

        # The Collision-Network
        self.cell2 = nn.LSTMCell(embedding_size, rnn_size)
        self.input_embedding_layer2 = nn.Linear(input_size, embedding_size)
        self.output_layer2 = nn.Linear(rnn_size, output_size)

        self.encoder_people_state = MLP(input_dim=4, output_dim=output_size, hidden_size=enc_size)
        self.dec_para_people = MLP(input_dim=2 * output_size, output_dim=1, hidden_size=dec_size)


        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward_lstm(self, input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2):
        #input_lstm: peds*4

        # LSTM1
        input_embedded1 = self.relu(self.input_embedding_layer1(input_lstm)) #peds*embedding_size
        h_nodes1, c_nodes1 = self.cell1(input_embedded1, (hidden_states1, cell_states1)) #h_nodes/c_nodes: peds*rnn_size
        outputs1 = self.output_layer1(h_nodes1) #peds*output_size

        # LSTM2
        input_embedded2 = self.relu(self.input_embedding_layer2(input_lstm)) #peds*embedding_size
        h_nodes2, c_nodes2 = self.cell2(input_embedded2, (hidden_states2, cell_states2)) #h_nodes/c_nodes: peds*rnn_size
        outputs2 = self.output_layer2(h_nodes2) #peds*output_size


        return outputs1, h_nodes1, c_nodes1, outputs2, h_nodes2, c_nodes2
    def forward_coefficient_people(self, outputs_features2, supplement, current_step, current_vel, device):
        num_peds = outputs_features2.size()[0]
        curr_supp = torch.zeros((num_peds, 26, 5)).to(device)
        num_peds_adj = []
        for i in range(num_peds):
            peds_con = supplement[i, : int(supplement[i, -1, 1]),:] #peds*5
            person_dir = peds_con[:,:2] - current_step[i,:] #peds*2
            dis = torch.norm(person_dir, dim=1) #peds
            cosangle = torch.matmul(person_dir, current_vel[i,:]) / (dis * torch.norm(current_vel[i,:])) #peds
            bool_sym = (dis < self.r_pixel) & (cosangle > self.costheta) #peds
            peds_vision = peds_con[bool_sym] #peds*5
            num_peds_vision = peds_vision.shape[0]
            curr_supp[i, : num_peds_vision, :] = peds_vision
            curr_supp[i, -1, 1] = num_peds_vision
            num_peds_adj.append(num_peds_vision)


        encoding_part1 = torch.unsqueeze(outputs_features2, dim=1).repeat(1, self.max_peds, 1) #peds*25*16
        features_others = self.encoder_people_state(curr_supp[:, :-1, :-1]) #peds*25*16
        input_coefficients = torch.cat((encoding_part1, features_others), dim=-1) #peds*25*32
        coefficients = torch.squeeze(100*self.sigmoid(self.dec_para_people(input_coefficients)))  # peds*25

        for i in range(num_peds):
            index_2 = int(curr_supp[i, -1, 1])
            coefficients[i, index_2:] = torch.zeros(self.max_peds - index_2)
        return coefficients, curr_supp

    def forward_coefficient_test(self, outputs_features2, supplement, current_step, current_vel, all_first_part,first_frame, device):


        num_peds = outputs_features2.size()[0]
        curr_supp = torch.zeros((num_peds, 26, 5)).to(device)
        curr_state = torch.cat((current_step, current_vel, torch.ones((num_peds,1)).to(device)), dim=1)
        for i in range(num_peds):
            first_part = all_first_part[i]
            peds_con1 = curr_state[first_part, :] #peds*5
            peds_con1[:, :2] = peds_con1[:, :2] + first_frame[first_part, :] - first_frame[i, :] #peds*2
            peds_con2 = supplement[i, int(supplement[i, -1, 0]) : int(supplement[i, -1, 1]), :]  # peds*5
            peds_con = torch.cat((peds_con1, peds_con2), dim=0) #peds*5

            person_dir = peds_con[:,:2] - current_step[i,:] #peds*2
            dis = torch.norm(person_dir, dim=1)  # peds
            cosangle = torch.matmul(person_dir, current_vel[i, :]) / (dis * torch.norm(current_vel[i, :]))  # peds
            bool_sym = (dis < self.r_pixel) & (cosangle > self.costheta)  # peds
            peds_vision = peds_con[bool_sym] #peds*5
            num_peds_vision = peds_vision.shape[0]
            curr_supp[i, : num_peds_vision, :] = peds_vision
            curr_supp[i, -1, 1] = num_peds_vision

        encoding_part1 = torch.unsqueeze(outputs_features2, dim=1).repeat(1, self.max_peds, 1)  # peds*25*16
        features_others = self.encoder_people_state(curr_supp[:, :-1, :-1])  # peds*25*16
        input_coefficients = torch.cat((encoding_part1, features_others), dim=-1)  # peds*25*32
        coefficients = torch.squeeze(100 * self.sigmoid(self.dec_para_people(input_coefficients)))  # peds*25


        for i in range(num_peds):
            index_2 = int(curr_supp[i, -1, 1])
            coefficients[i, index_2:] = torch.zeros(self.max_peds - index_2)
        return coefficients, curr_supp

    def forward_next_step(self, current_step, current_vel, initial_speeds, dest, features_lstm1, coefficients,
                              current_supplement, sigma, semantic_map, first_frame, k_env, device=torch.device('cpu'),
                              k_scope=50, k_label_4= 0.2):
        k_scope = torch.tensor(k_scope).to(device)
        k_label_4 = torch.tensor(k_label_4).to(device)
        delta_t = torch.tensor(0.4).to(device)
        e = stateutils_desired_directions(current_step, dest)  #peds*2

        features_dest = self.encoder_dest_state(dest)
        features_tau = torch.cat((features_lstm1, features_dest), dim = -1)
        tau = self.sigmoid(self.dec_tau(features_tau)) + 0.4


        F0 = 1.0 / tau * (initial_speeds * e - current_vel)  #peds*2
        F1 = f_ab_fun(current_step, coefficients, current_supplement, sigma, device)

        F2 = environment(current_step.detach(), first_frame, current_vel.detach(), semantic_map, k_scope, k_env, k_label_4, F0, device)
        #F2 = torch.DoubleTensor(F2).to(device)

        F = F0 + F1 + F2  #peds*2

        w_v = current_vel + delta_t * F  #peds*2

        # update state
        prediction = current_step + w_v * delta_t  # peds*2

        return prediction, w_v




