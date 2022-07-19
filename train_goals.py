import numpy as np
import torch
import yaml
from model_goals import *
from utils import *
import torch.optim as optim
from torch.autograd import Variable
import argparse
import pickle

def train(train_batches):

    model.train()
    total_loss = 0
    criterion = nn.MSELoss()

    shuffle_index = torch.randperm(216)
    for k in shuffle_index[:30]:
        traj = train_batches[k]
        traj = torch.squeeze(torch.DoubleTensor(traj).to(device))
        y = traj[:, params['past_length']:, :2] #peds*future_length*2

        dest = y[:, -1, :].to(device)
        #dest_state = traj[:, -1, :]
        future = y.contiguous().to(device)

        future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * 0.4) #peds*2
        future_vel_norm = torch.norm(future_vel, dim=-1) #peds
        initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1) #peds*1

        num_peds = traj.shape[0]
        numNodes = num_peds
        hidden_states = Variable(torch.zeros(numNodes, params['rnn_size']))
        cell_states = Variable(torch.zeros(numNodes, params['rnn_size']))
        hidden_states = hidden_states.to(device)
        cell_states = cell_states.to(device)

        for m in range(1, params['past_length']):  #
            current_step = traj[:, m, :2]  # peds*2
            current_vel = traj[:, m, 2:]  # peds*2
            input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
            outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

        predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)
        prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                  outputs_features, device=device)
        predictions[:, 0, :] = prediction

        current_step = prediction #peds*2
        current_vel = w_v #peds*2

        for t in range(params['future_length'] - 1):
            input_lstm = torch.cat((current_step, current_vel), dim=1)
            outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

            future_vel = (dest - prediction) / ((12-t-1) * 0.4)  # peds*2
            future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

            prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                      outputs_features, device=device)
            predictions[:, t+1, :] = prediction

            current_step = prediction  # peds*2
            current_vel = w_v # peds*2
        optimizer.zero_grad()

        loss = calculate_loss(criterion, future, predictions)
        loss.backward()

        total_loss += loss.item()
        optimizer.step()

    return total_loss

def test(traj, generated_goals):
    model.eval()

    with torch.no_grad():
        traj = torch.squeeze(torch.DoubleTensor(traj).to(device))
        generated_goals = torch.DoubleTensor(generated_goals).to(device)

        y = traj[:, params['past_length']:, :2]  # peds*future_length*2
        y = y.cpu().numpy()
        dest = generated_goals

        future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * 0.4)  # peds*2
        future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
        initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

        num_peds = traj.shape[0]
        numNodes = num_peds
        hidden_states = Variable(torch.zeros(numNodes, params['rnn_size']))
        cell_states = Variable(torch.zeros(numNodes, params['rnn_size']))
        hidden_states = hidden_states.to(device)
        cell_states = cell_states.to(device)

        for m in range(1, params['past_length']):  #
            current_step = traj[:, m, :2]  # peds*2
            current_vel = traj[:, m, 2:]  # peds*2
            input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
            outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

        predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)

        prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                              outputs_features, device=device)

        predictions[:, 0, :] = prediction

        current_step = prediction #peds*2
        current_vel = w_v #peds*2

        for t in range(params['future_length'] - 1):
            input_lstm = torch.cat((current_step, current_vel), dim=1)
            outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

            future_vel = (dest - prediction) / ((12 - t - 1) * 0.4)  # peds*2
            future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

            prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                  outputs_features, device=device)
            predictions[:, t + 1, :] = prediction

            current_step = prediction  # peds*2
            current_vel = w_v  # peds*2

        predictions = predictions.cpu().numpy()

        test_ade = np.mean(np.linalg.norm(y - predictions, axis = 2), axis=1) # peds
        test_fde = np.linalg.norm((y[:,-1,:] - predictions[:, -1, :]), axis=1) #peds
    return test_ade, test_fde

parser = argparse.ArgumentParser(description='NSP')
parser.add_argument('--gpu_index', '-gi', type=int, default=1)
parser.add_argument('--save_file', '-sf', type=str, default='SDD_goals.pt')
args = parser.parse_args()

CONFIG_FILE_PATH = 'config/sdd_goals.yaml'  # yaml config file containing all the hyperparameters
with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)

train_data_path = 'data/SDD/train.pickle'
with open(train_data_path, 'rb') as f:
    train_data = pickle.load(f)
train_traj = train_data[0]
for traj in train_traj:
    traj -= traj[:, :1, :]
train_traj = augment_data(train_traj)
train_batches = []
for traj in train_traj:
    traj_vel = calculate_v(traj)
    traj_complete_translated = np.concatenate((traj, traj_vel), axis=-1)
    train_batches.append(traj_complete_translated)

test_data_path = 'data/SDD/test.pickle'
with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)
goals_path = 'data/SDD/goals_Ynet.pickle'
with open(goals_path, 'rb') as f:
    goals = pickle.load(f)
list_all_traj_com_trans_test = []
list_all_goals_trans_test = []
for i, (traj, predicted_goals) in enumerate(zip(test_data[0], goals[1])):
    traj_vel = calculate_v(traj)
    traj_translated = translation(traj)
    traj_complete_translated = np.concatenate((traj_translated, traj_vel), axis=-1)  # peds*20*4
    predicted_goals_translated = translation_goals(predicted_goals, traj)  # peds*2
    list_all_traj_com_trans_test.append(traj_complete_translated)
    list_all_goals_trans_test.append(predicted_goals_translated)

traj_complete_translated_test = np.concatenate(list_all_traj_com_trans_test) # peds*20*4
goals_translated_test = np.concatenate(list_all_goals_trans_test, axis=1)  #peds*2

model = NSP(params["input_size"], params["embedding_size"], params["rnn_size"], params["output_size"],  params["enc_dest_state_size"], params["dec_tau_size"])
model = model.double().to(device)

optimizer = optim.Adam(model.parameters(), lr=  params["learning_rate"])

best_ade = 7.85
best_fde = 11.85

for e in range(params['num_epochs']):
    total_loss = train(train_batches)

    ade_20 = np.zeros((20, len(traj_complete_translated_test)))
    fde_20 = np.zeros((20, len(traj_complete_translated_test)))
    for j in range(20):
        test_ade, test_fde= test(traj_complete_translated_test, goals_translated_test[j, :, :])
        ade_20[j, :] = test_ade
        fde_20[j, :] = test_fde
    test_ade = np.mean(np.min(ade_20, axis=0))
    test_fde = np.mean(np.min(fde_20, axis=0))
    print()

    if best_ade > test_ade:
        print("Epoch: ", e)
        print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_ade))
        best_ade = test_ade
        best_fde = test_fde
        save_path = 'saved_models/' + args.save_file
        torch.save({'hyper_params': params,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                        }, save_path)
        print("Saved model to:\n{}".format(save_path))

    print('epoch:', e)
    print("Train Loss", total_loss)
    print("Test ADE", test_ade)
    print("Test FDE", test_fde)
    print("Test Best ADE Loss So Far", best_ade)
    print("Test Best FDE Loss So Far", best_fde)