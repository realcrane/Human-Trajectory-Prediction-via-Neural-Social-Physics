import torch
import torch.nn as nn
import torch.optim as optim
import sys
import argparse
sys.path.append("../eth_ucy/")

sys.path.append("../../utils/")
import yaml
from model_cvae import *
from model_nsp_wo import *
from utils import *
import numpy as np
import copy
import pickle
import os
import cv2

parser = argparse.ArgumentParser(description='CVAE')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=3)
parser.add_argument('--config_filename', '-cfn', type=str, default='sdd_nsp_cvae.yaml')
parser.add_argument('--save_file', '-sf', type=str, default='SDD_nsp_cvae.pt')
parser.add_argument('--verbose', '-v', action='store_true')

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)

with open("config/" + args.config_filename, 'r') as file:
    try:
        params = yaml.load(file, Loader = yaml.FullLoader)
    except:
        params = yaml.load(file)
file.close()
print(params)

def train(path, scenes):

    model_cvae.train()
    model_nsp.eval()
    train_loss = 0
    total_kld, total_adl = 0, 0
    criterion = nn.MSELoss()

    shuffle_index = torch.randperm(30)

    for t in shuffle_index:
        scene = scenes[t]
        load_name = path + scene
        with open(load_name, 'rb') as f:
            data = pickle.load(f)
        traj_complete, supplement, first_part = data[0], data[1], data[2]
        traj_complete = np.array(traj_complete)
        if len(traj_complete.shape) == 1:
            continue
        first_frame = traj_complete[:, 0, :2]
        traj_translated = translation(traj_complete[:, :, :2])
        traj_complete_translated = np.concatenate((traj_translated, traj_complete[:, :, 2:]), axis=-1)
        supplement_translated = translation_supp(supplement, traj_complete[:, :, :2])

        traj, supplement = torch.DoubleTensor(traj_complete_translated).to(device), torch.DoubleTensor(
            supplement_translated).to(device)
        first_frame = torch.DoubleTensor(first_frame).to(device)

        semantic_map = cv2.imread(semantic_path_train + semantic_maps_name_train[t])
        semantic_map = np.transpose(semantic_map[:, :, 0])

        y = traj[:, params['past_length']:, :2]  # peds*future_length*2
        dest = y[:, -1, :].to(device)
        future = y.contiguous().to(device)

        future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * 0.4) #peds*2
        future_vel_norm = torch.norm(future_vel, dim=-1) #peds
        initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1) #peds*1

        num_peds = traj.shape[0]
        numNodes = num_peds

        hidden_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
        cell_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
        hidden_states1 = hidden_states1.to(device)
        cell_states1 = cell_states1.to(device)
        hidden_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
        cell_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
        hidden_states2 = hidden_states2.to(device)
        cell_states2 = cell_states2.to(device)

        for m in range(1, params['past_length']):  #
            current_step = traj[:, m, :2]  # peds*2
            current_vel = traj[:, m, 2:]  # peds*2
            input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
            with torch.no_grad():
                outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2  \
                    = model_nsp.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)
        with torch.no_grad():
            coefficients, curr_supp = model_nsp.forward_coefficient_people(outputs_features2, supplement[:, 7, :, :], current_step, current_vel, device)  # peds*maxpeds*2, peds*(max_peds + 1)*4
            prediction, w_v = model_nsp.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                  outputs_features1, coefficients, curr_supp, sigma, semantic_map, first_frame, k_env, device=device)

        x = copy.deepcopy(traj[:, :8, :2])
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2])).to(device) * params['data_scale']
        alpha = (traj[:, 8, :2] - prediction)*params['data_scale']
        alpha_recon, mu, var = model_cvae.forward(x, next_step=alpha, device=device)
        optimizer.zero_grad()
        kld, adl = calculate_loss_cvae(mu, var, criterion, alpha, alpha_recon)
        loss = kld * params["kld_reg"] + adl
        loss.backward()

        train_loss += loss.item()
        total_kld += kld.item()
        total_adl += adl.item()
        optimizer.step()

        for i in range(1, params['future_length']):
            current_step = traj[:, 7+i, :2]  # peds*2
            current_vel = traj[:, 7+i, 2:]  # peds*2
            input_lstm = torch.cat((current_step, current_vel), dim=1)
            with torch.no_grad():
                outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2  \
                    = model_nsp.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

            future_vel = (dest - traj[:, 7+i, :2]) / ((12-i) * 0.4)  # peds*2
            future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1
            with torch.no_grad():
                coefficients, curr_supp = model_nsp.forward_coefficient_people(outputs_features2,
                                                                               supplement[:, 7+i, :, :], current_step,
                                                                               current_vel,
                                                                               device)  # peds*maxpeds*2, peds*(max_peds + 1)*4
                prediction, w_v = model_nsp.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                              outputs_features1, coefficients, curr_supp, sigma,
                                                              semantic_map, first_frame, k_env, device=device)

            x = copy.deepcopy(traj[:, i : 8 + i, :2])
            first_frame_x = copy.deepcopy(x[:, :1, :])
            x = x - first_frame_x
            x = torch.reshape(x,(-1, x.shape[1] * x.shape[2])).to(device) * params['data_scale']
            alpha = (traj[:, 8+i, :2] - prediction) * params['data_scale']
            alpha_recon, mu, var = model_cvae.forward(x, next_step=alpha, device=device)


            optimizer.zero_grad()
            kld, adl = calculate_loss_cvae(mu, var, criterion, alpha, alpha_recon)
            loss = kld * params["kld_reg"] + adl
            loss.backward()

            train_loss += loss.item()
            total_kld += kld.item()
            total_adl += adl.item()
            optimizer.step()

    return train_loss, total_kld, total_adl

def test(path, scenes, generated_goals, best_of_n = 1):
    model_cvae.eval()
    model_nsp.eval()
    all_ade = []
    all_fde = []
    index = 0
    assert best_of_n >= 1 and type(best_of_n) == int

    with torch.no_grad():
        for i, scene in enumerate(scenes):
            load_name = path + scene
            with open(load_name, 'rb') as f:
                data = pickle.load(f)
            traj_complete, supplement, first_part = data[0], data[1], data[2]
            traj_complete = np.array(traj_complete)
            if len(traj_complete.shape) == 1:
                index += 1
                continue
            traj_translated = translation(traj_complete[:, :, :2])
            traj_complete_translated = np.concatenate((traj_translated, traj_complete[:, :, 2:]), axis=-1)
            supplement_translated = translation_supp(supplement, traj_complete[:, :, :2])
            traj, supplement = torch.DoubleTensor(traj_complete_translated).to(device), torch.DoubleTensor(
                supplement_translated).to(device)
            traj_copy = copy.deepcopy(traj)

            semantic_map = cv2.imread(semantic_path_test + semantic_maps_name_test[i])
            semantic_map = np.transpose(semantic_map[:, :, 0])
            y = traj[:, params['past_length']:, :2]  # peds*future_length*2
            y = y.cpu().numpy()
            first_frame = torch.DoubleTensor(traj_complete[:, 0, :2]).to(device)  # peds*2
            num_peds = traj.shape[0]
            ade_20 = np.zeros((20, len(traj_complete)))
            fde_20 = np.zeros((20, len(traj_complete)))

            for j in range(20):
                goals_translated = translation_goals(generated_goals[1][i-index][j,:,:], traj_complete[:,:,:2]) # 20*peds*2
                dest = torch.DoubleTensor(goals_translated).to(device)

                future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * 0.4)  # peds*2
                future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
                initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

                numNodes = num_peds
                hidden_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
                cell_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
                hidden_states1 = hidden_states1.to(device)
                cell_states1 = cell_states1.to(device)
                hidden_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
                cell_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
                hidden_states2 = hidden_states2.to(device)
                cell_states2 = cell_states2.to(device)

                for m in range(1, params['past_length']):  #
                    current_step = traj[:, m, :2]  # peds*2
                    current_vel = traj[:, m, 2:]  # peds*2
                    input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                    outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                        = model_nsp.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

                predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)
                coefficients, curr_supp = model_nsp.forward_coefficient_people(outputs_features2, supplement[:, 7, :, :],
                                                                        current_step, current_vel,
                                                                        device)  # peds*maxpeds*2, peds*(max_peds + 1)*4
                prediction, w_v = model_nsp.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                        outputs_features1, coefficients, curr_supp, sigma, semantic_map,
                                                        first_frame, k_env, device=device)

                x = traj_copy[:, :8, :2]
                x = torch.reshape(x, (-1, x.shape[1] * x.shape[2])).to(device) * params['data_scale']
                alpha_step = torch.zeros(best_of_n, len(traj), 2).to(device)
                for t in range(best_of_n):
                    alpha_recon = model_cvae.forward(x, device=device)
                    alpha_step[t, :, :] = alpha_recon
                alpha_step[-1,:,:] = torch.zeros_like(alpha_step[-1,:,:])
                prediction_correct = alpha_step / params['data_scale'] + prediction
                predictions_norm = torch.norm((prediction_correct - traj[:, 8, :2]), dim=-1)
                values, indices = torch.min(predictions_norm, dim=0)  # peds
                ns_recon_best = prediction_correct[indices, [x for x in range(len(traj))], :]  # peds*2
                predictions[:, 0, :] = ns_recon_best
                current_step = ns_recon_best
                current_vel = (ns_recon_best - traj_copy[:, 7, :2]) / 0.4
                traj_copy[:, 8, :2] = current_step
                traj_copy[:, 8, 2:] = current_vel


                for m in range(1, params['future_length']):
                    input_lstm = torch.cat((current_step, current_vel), dim=1)
                    outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                        = model_nsp.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)
                    future_vel = (dest - prediction) / ((12 - m) * 0.4)  # peds*2
                    future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
                    initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1
                    coefficients, current_supplement = model_nsp.forward_coefficient_test(outputs_features2,
                                                                                      supplement[:, 7+m, :, :],
                                                                                      current_step, current_vel,
                                                                                      first_part, first_frame,
                                                                                      device=device)
                    prediction, w_v = model_nsp.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                                  outputs_features1, coefficients, curr_supp, sigma,
                                                                  semantic_map,
                                                                  first_frame, k_env, device=device)

                    x = traj_copy[:, m: 8 + m, :2]
                    first_frame_x = copy.deepcopy(x[:, :1, :])
                    x = x - first_frame_x
                    x = torch.reshape(x, (-1, x.shape[1] * x.shape[2])).to(device) * params['data_scale']
                    alpha_step = torch.zeros(best_of_n, len(traj), 2).to(device)
                    for t in range(best_of_n):
                        alpha_recon = model_cvae.forward(x, device=device)
                        alpha_step[t, :, :] = alpha_recon
                    alpha_step[-1, :, :] = torch.zeros_like(alpha_step[-1, :, :])
                    prediction_correct = alpha_step / params['data_scale'] + prediction
                    predictions_norm = torch.norm((prediction_correct - traj[:, 8+m, :2]), dim=-1)
                    values, indices = torch.min(predictions_norm, dim=0)  # peds
                    ns_recon_best = prediction_correct[indices, [x for x in range(len(traj))], :]  # peds*2
                    predictions[:, m, :] = ns_recon_best
                    current_step = ns_recon_best
                    current_vel = (ns_recon_best - traj_copy[:, 7+m, :2]) / 0.4
                    traj_copy[:, 8+m, :2] = current_step
                    traj_copy[:, 8+m, 2:] = current_vel

                predictions = predictions.cpu().numpy()
                dest = dest.cpu().numpy()

                # ADE error
                test_ade = np.mean(np.linalg.norm(y - predictions, axis=2), axis=1)  # peds
                test_fde = np.linalg.norm((y[:, -1, :] - predictions[:, -1, :]), axis=1)  # peds
                ade_20[j, :] = test_ade
                fde_20[j, :] = test_fde
            ade_single = np.min(ade_20, axis=0)  # peds
            fde_single = np.min(fde_20, axis=0)  # peds
            all_ade.append(ade_single)
            all_fde.append(fde_single)
            #print('test finish:', i)
        ade = np.mean(np.concatenate(all_ade))
        fde = np.mean(np.concatenate(all_fde))



    return ade, fde


model_cvae = CVAE(params["enc_past_size"], params["enc_dest_size"], params["enc_latent_size"], params["dec_size"], params["fdim"], params["zdim"], params["sigma"], params["past_length"], params["future_length"], args.verbose)
model_nsp = NSP(params["input_size"], params["embedding_size"], params["rnn_size"], params["output_size"],  params["enc_size_nsp"], params["dec_size_nsp"])
model_cvae = model_cvae.double().to(device)
model_nsp = model_nsp.double().to(device)
load_path_nsp = 'saved_models/SDD_nsp_wo.pt'
checkpoint_trained = torch.load(load_path_nsp, map_location=torch.device(device))
model_nsp.load_state_dict(checkpoint_trained['model_state_dict'])
k_env = checkpoint_trained['k_env']
sigma = torch.tensor(100)

optimizer = optim.Adam(model_cvae.parameters(), lr=  params["learning_rate"])

goals_path_test = 'data/SDD/goals_Ynet.pickle'
with open(goals_path_test, 'rb') as f:
    goals_test = pickle.load(f)
path_train = 'data/SDD/train_pickle/'
scenes_train = os.listdir(path_train)
path_test = 'data/SDD/test_pickle/'
scenes_test = os.listdir(path_test)
semantic_path_train = 'data/SDD/train_masks/'
semantic_maps_name_train = os.listdir(semantic_path_train)
semantic_path_test = 'data/SDD/test_masks/'
semantic_maps_name_test = os.listdir(semantic_path_test)


best_test_loss = 2.55 # start saving after this threshold
best_endpoint_loss = 3.5
N = params["n_values"]

for e in range(params['num_epochs']):
    train_loss, kld, adl = train(path_train, scenes_train)
    test_ade, test_fde = test(path_test, scenes_test, goals_test,  best_of_n = N-5)

    print()

    if test_ade < best_test_loss:
        best_test_loss = test_ade
        best_endpoint_loss = test_fde
        print("Epoch: ", e)
        print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_ade))
        if best_test_loss < 2.55:
            save_path = 'saved_models/' + args.save_file
            torch.save({
                        'hyper_params': params,
                        'model_state_dict': model_cvae.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, save_path)
            print("Saved model to:\n{}".format(save_path))


    print('num_epoch', e)
    print("Train Loss", train_loss)
    print("KLD", kld)
    print("ADL", adl)
    print('Current Test ADE', test_ade)
    print('Current Test FDE', test_fde)
    print("Test Best ADE Loss So Far (N = {})".format(N), best_test_loss)
    print("Test Best FDE Loss So Far (N = {})".format(N), best_endpoint_loss)



