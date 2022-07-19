import numpy as np
import copy
import torch
def calculate_v(x_seq):
    length = x_seq.shape[1]
    peds = x_seq.shape[0]
    x_seq_velocity = np.zeros_like(x_seq)
    episa = 1e-6
    for i in range(1, length):
        for j in range(peds):
            position = x_seq[j][i]
            before_position = x_seq[j][i-1]
            position_norm = np.linalg.norm(position)
            before_position_norm = np.linalg.norm(before_position)
            if position_norm < episa:
                velocity = np.array([0,0])
            else:
                if before_position_norm < episa:
                    velocity = np.array([0, 0])
                else:
                    velocity = (position - before_position)/0.4
            x_seq_velocity[j][i] = velocity
    return x_seq_velocity

def translation(x_seq):
    first_frame = x_seq[:, 0, :]
    first_frame_new = first_frame[:, np.newaxis, :] #peds*1*2
    x_seq_translated = x_seq - first_frame_new
    return x_seq_translated

def augment_data(data):
    ks = [1, 2, 3]
    data_ = copy.deepcopy(data)  # data without rotation, used so rotated data can be appended to original df
    #k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
    for k in ks:
        for t in range(len(data)):
            data_rot = rot(data[t], k)
            data_.append(data_rot)
    for t in range(27*4):
        data_flip = fliplr(data_[t])

        data_.append(data_flip)

    return data_

def rot(data_traj, k=1):
    xy = data_traj.copy()

    c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
    R = np.array([[c, s], [-s, c]])
    for i in range(20):
        xy[:, i, :] = np.dot(xy[:, i, :], R)

    return xy

def fliplr(data_traj):
    xy = data_traj.copy()

    R = np.array([[-1, 0], [0, 1]])
    for i in range(20):
        xy[:, i, :] = np.dot(xy[:, i, :], R)

    return xy

def calculate_loss(criterion, future, predictions):

    ADL_traj = criterion(future, predictions)  # better with l2 loss

    return ADL_traj

def calculate_loss_cvae(mean, log_var, criterion, future, predictions):
    # reconstruction loss
    ADL_traj = criterion(future, predictions) # better with l2 loss

    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return KLD, ADL_traj
def translation_goals(goals, x_seq):
    first_frame = x_seq[:, 0, :]
    goals_translated = goals - first_frame #peds*2
    return goals_translated

def select_para(model_complete):

    params_totrain = []
    params_totrain.extend(model_complete.cell2.parameters())
    params_totrain.extend(model_complete.input_embedding_layer2.parameters())
    params_totrain.extend(model_complete.output_layer2.parameters())
    params_totrain.extend(model_complete.encoder_people_state.parameters())
    params_totrain.extend(model_complete.dec_para_people.parameters())
    return params_totrain

def new_point(checkpoint_t_dic, checkpoint_i_dic):
    point = checkpoint_i_dic
    dk_t = list(checkpoint_t_dic.keys())
    dk_i = list(point.keys())
    for k in range(22):
        point[dk_i[k]] = checkpoint_t_dic[dk_t[k]]
    return point

def translation_supp(supplemnt, x_seq):
    first_frame = x_seq[:, 0, :]
    for ped in range(supplemnt.shape[0]):
        for frame in range(20):
            all_other_peds = int(supplemnt[ped, frame, -1, 1])
            supplemnt[ped, frame, :all_other_peds, :2] = supplemnt[ped, frame, :all_other_peds, :2] - first_frame[ped,:]
    return supplemnt