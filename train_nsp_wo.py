from model_nsp_wo import *
from torch.autograd import Variable
from utils import *
import torch.optim as optim
import argparse
import os
import pickle
import cv2
import yaml

def train(path, scenes):

    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    shuffle_index = torch.randperm(30)

    for i in shuffle_index:
        scene = scenes[i]
        load_name = path + scene
        with open(load_name, 'rb') as f:
            data = pickle.load(f)
        traj_complete, supplement, first_part = data[0], data[1], data[2]
        traj_complete = np.array(traj_complete)
        if len(traj_complete.shape) == 1:
            continue
        first_frame = traj_complete[:,0,:2]
        traj_translated = translation(traj_complete[:, :, :2])
        traj_complete_translated = np.concatenate((traj_translated, traj_complete[:, :, 2:]), axis=-1)
        supplement_translated = translation_supp(supplement, traj_complete[:, :, :2])

        traj, supplement = torch.DoubleTensor(traj_complete_translated).to(device), torch.DoubleTensor(supplement_translated).to(device)
        first_frame = torch.DoubleTensor(first_frame).to(device)

        semantic_map = cv2.imread(semantic_path_train + semantic_maps_name_train[i])
        semantic_map = np.transpose(semantic_map[:, :, 0])

        y = traj[:, params['past_length']:, :2] #peds*future_length*2
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
            outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2  \
                = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

        predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)

        coefficients, curr_supp = model.forward_coefficient_people(outputs_features2, supplement[:, 7, :, :], current_step, current_vel, device)  # peds*maxpeds*2, peds*(max_peds + 1)*4

        prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                  outputs_features1, coefficients, curr_supp, sigma, semantic_map, first_frame, k_env, device=device)
        predictions[:, 0, :] = prediction

        current_step = prediction #peds*2
        current_vel = w_v #peds*2

        for t in range(params['future_length'] - 1):
            input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
            outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

            future_vel = (dest - prediction) / ((12-t-1) * 0.4)  # peds*2
            future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

            coefficients, curr_supp = model.forward_coefficient_people(outputs_features2, supplement[:, 7+t+1, :, :], current_step, current_vel, device)

            prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                      outputs_features1, coefficients, curr_supp, sigma, semantic_map,
                                                      first_frame, k_env, device=device)
            predictions[:, t+1, :] = prediction

            current_step = prediction  # peds*2
            current_vel = w_v # peds*2
        optimizer.zero_grad()

        loss = calculate_loss(criterion, future, predictions)
        loss.backward()

        total_loss += loss.item()
        optimizer.step()
        #print('k_env:', k_env)
        # print('finish:', scene)
        # print('loss:', loss)

    return total_loss

def test(path, scenes, generated_goals):
    model.eval()
    all_ade = []
    all_fde = []
    index = 0
    all_traj = []
    all_scenes = []

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
            semantic_map = cv2.imread(semantic_path_test + semantic_maps_name_test[i])
            semantic_map = np.transpose(semantic_map[:, :, 0])
            y = traj[:, params['past_length']:, :2]  # peds*future_length*2
            y = y.cpu().numpy()
            first_frame = torch.DoubleTensor(traj_complete[:, 0, :2]).to(device)  # peds*2
            num_peds = traj.shape[0]
            ade_20 = np.zeros((20, len(traj_complete)))
            fde_20 = np.zeros((20, len(traj_complete)))
            predictions_20 = np.zeros((20, num_peds, params['future_length'], 2))
            for j in range(20):
                # index_goals = generated_goals[2].index(scene)
                goals_translated = translation_goals(generated_goals[1][i-index][j,:,:], traj_complete[:,:,:2]) # 20*peds*2

                # correct_num = first_part[0][0] - 1
                # for m in range(len(first_part)):
                #     first_part[m] = (np.array(first_part[m]) - correct_num).tolist()
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
                        = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

                predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)

                coefficients, curr_supp = model.forward_coefficient_people(outputs_features2, supplement[:, 7, :, :], current_step, current_vel, device)  # peds*maxpeds*2, peds*(max_peds + 1)*4

                prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                          outputs_features1, coefficients, curr_supp, sigma, semantic_map,
                                                          first_frame, k_env, device=device)
                predictions[:, 0, :] = prediction

                current_step = prediction  # peds*2
                current_vel = w_v  # peds*2

                for t in range(params['future_length'] - 1):
                    input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                    outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                        = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

                    future_vel = (dest - prediction) / ((12 - t - 1) * 0.4)  # peds*2
                    future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
                    initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

                    coefficients, current_supplement = model.forward_coefficient_test(outputs_features2, supplement[:, 7 + t + 1, :, :],
                                                                    current_step, current_vel, first_part, first_frame,
                                                                    device=device)

                    prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                              outputs_features1, coefficients, curr_supp, sigma,
                                                              semantic_map,
                                                              first_frame, k_env, device=device)

                    predictions[:, t + 1, :] = prediction

                    current_step = prediction  # peds*2
                    current_vel = w_v  # peds*2

                predictions = predictions.cpu().numpy()
                dest = dest.cpu().numpy()

                # ADE error
                test_ade = np.mean(np.linalg.norm(y - predictions, axis = 2), axis=1) # peds
                test_fde = np.linalg.norm((y[:,-1,:] - predictions[:, -1, :]), axis=1) #peds
                ade_20[j, :] = test_ade
                fde_20[j, :] = test_fde
                predictions_20[j] = predictions
            ade_single = np.min(ade_20, axis=0)  # peds
            fde_single = np.min(fde_20, axis=0)  # peds
            all_ade.append(ade_single)
            all_fde.append(fde_single)
            all_traj.append(predictions_20)
            all_scenes.append(scene)

            #print('test finish:', i)
        ade = np.mean(np.concatenate(all_ade))
        fde = np.mean(np.concatenate(all_fde))
    return ade, fde


parser = argparse.ArgumentParser(description='NSP')

parser.add_argument('--gpu_index', '-gi', type=int, default=1)
parser.add_argument('--save_file', '-sf', type=str, default='SDD_nsp_wo_complete_1.pt')

args = parser.parse_args()

CONFIG_FILE_PATH = 'config/sdd_nsp_wo.yaml'  # yaml config file containing all the hyperparameters
with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)


model = NSP(params["input_size"], params["embedding_size"], params["rnn_size"], params["output_size"],  params["enc_size"], params["dec_size"])
model = model.double().to(device)

load_path = 'saved_models/SDD_goals.pt'
checkpoint_trained = torch.load(load_path)
load_path_ini = 'saved_models/SDD_nsp_wo_ini.pt'
checkpoint_ini = torch.load(load_path_ini)
checkpoint_dic = new_point(checkpoint_trained['model_state_dict'], checkpoint_ini['model_state_dict'])
model.load_state_dict(checkpoint_dic)
sigma = torch.tensor(100)

parameter_train = select_para(model)
k_env = torch.tensor(65.0).to(device)
k_env.requires_grad = True

optimizer = optim.Adam([{'params': parameter_train}, {'params': [k_env]}], lr=  params["learning_rate"])
best_ade = 6.55
best_fde = 11

goals_path = 'data/SDD/goals_Ynet.pickle'
with open(goals_path, 'rb') as f:
    goals = pickle.load(f)
path_train = 'data/SDD/train_pickle/'
scenes_train = os.listdir(path_train)
path_test = 'data/SDD/test_pickle/'
scenes_test = os.listdir(path_test)
semantic_path_train = 'data/SDD/train_masks/'
semantic_maps_name_train = os.listdir(semantic_path_train)
semantic_path_test = 'data/SDD/test_masks/'
semantic_maps_name_test = os.listdir(semantic_path_test)

for e in range(params['num_epochs']):
    total_loss = train(path_train, scenes_train)
    test_ade, test_fde = test(path_test, scenes_test, goals)

    print()

    if best_ade > test_ade:
        print("Epoch: ", e)
        print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_ade))
        best_ade = test_ade
        best_fde = test_fde
        save_path = 'saved_models/' + args.save_file
        torch.save({'hyper_params': params,
                    'model_state_dict': model.state_dict(),
                    'k_env': k_env
                        }, save_path)
        print("Saved model to:\n{}".format(save_path))


    print('epoch:', e)
    print('k_env:', k_env)
    print("Train Loss", total_loss)
    print("Test ADE", test_ade)
    print("Test FDE", test_fde)
    print("Test Best ADE Loss So Far", best_ade)
    print("Test Best Min FDE", best_fde)