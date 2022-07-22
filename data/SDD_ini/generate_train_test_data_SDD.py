import os
import math
import numpy as np
import copy
import pickle

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

def calculate_a(vel):
    length = vel.shape[1]
    peds = vel.shape[0]
    x_seq_acce = np.zeros_like(vel)
    episa = 1e-6
    for i in range(2, length):
        for j in range(peds):
            position = vel[j][i]
            before_position = vel[j][i - 1]
            position_norm = np.linalg.norm(position)
            before_position_norm = np.linalg.norm(before_position)
            if position_norm < episa:
                acce = np.array([0, 0])
            else:
                if before_position_norm < episa:
                    acce = np.array([0, 0])
                else:
                    acce = (position - before_position) / 0.4
            x_seq_acce[j][i] = acce
    return x_seq_acce


def outlier_test(seq, seq_remains):
    seq = np.swapaxes(seq, 1, 2)  # peds*20*2
    seq_remains = np.swapaxes(seq_remains, 1, 2)  # peds*20*3
    seq_remains_label = np.max(seq_remains[:,:,-1], axis=1) #peds
    seq_remains = seq_remains[seq_remains_label  == 1,:,:] #peds*20*3
    seq_remains = seq_remains[:,:,:-1] #peds*20*2
    seq_all = np.concatenate([seq, seq_remains]) #peds*20*2
    vel_all = calculate_v(seq_all) #peds*20*2
    acc_all = calculate_a(vel_all) #peds*20*2
    vel_all_norm = np.linalg.norm(vel_all, axis=-1) #peds*20
    acc_all_norm = np.linalg.norm(acc_all, axis=-1) #peds*20
    vel_all_norm_max = np.amax(vel_all_norm) #1
    acc_all_norm_max = np.amax(acc_all_norm) #1
    if vel_all_norm_max > 300:
        bool_out = True
    else:
        if vel_all_norm_max > 100 and vel_all_norm_max <= 300:
            if acc_all_norm_max > 150:
                bool_out = True
            else:
                bool_out = False
        else:
            bool_out = False

    return bool_out


path = 'annotations/'
scenes = os.listdir(path)
obs_len = 8
pred_len = 12
seq_len = obs_len + pred_len
skip = 20
min_ped = 1
max_peds = 120
num_success = 0
# r_pixel = 150
# theta = np.pi/3
# costheta = np.cos(theta)
label_set = 0
for scene in scenes:
    scene_path = path + scene
    all_videos = os.listdir(scene_path)
    for video in all_videos:
        label_set += 1
        video_path = scene_path + '/' + video
        processed_data_path = video_path + '/processed.npy'

        num_peds_in_seq = []
        num_peds_in_seq_remains = []
        seq_list = []
        seq_list_setinfo = []
        seq_list_remains = []
        num_seqs = 0
        index = []
        episa = 1e-6

        data = np.load(processed_data_path, allow_pickle=True)
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        num_sequences = int(
            math.ceil((len(frames) - seq_len + 1) / skip))

        for idx in range(0, num_sequences * skip + 1, skip):
            if len(frame_data[idx:idx + seq_len]) < 1:
                continue
            curr_seq_data = np.concatenate(
                frame_data[idx:idx + seq_len], axis=0)
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len))
            curr_seq_setinfo = np.zeros((len(peds_in_curr_seq), 2, seq_len))
            num_peds_considered = 0
            selected_id = []

            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                             ped_id, :]


                if curr_ped_seq[0,-1] != 1:
                    continue
                #curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                ped_front = frames.index(curr_ped_seq[0, 0]) - idx
                ped_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                if curr_ped_seq.shape[0] != seq_len:
                    continue
                curr_ped_seq_setinfo = np.transpose(curr_ped_seq[:, :2]) #2*20
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:4]) #2*20

                _idx = num_peds_considered
                curr_seq[_idx, :, ped_front:ped_end] = curr_ped_seq
                curr_seq_setinfo[_idx, :, ped_front:ped_end] = curr_ped_seq_setinfo
                curr_seq_setinfo[_idx, -1, -1] = label_set


                num_peds_considered += 1
                selected_id.append(ped_id)

            if num_peds_considered > min_ped:
                num_peds_in_seq.append(num_peds_considered)
                seq_list.append(curr_seq[:num_peds_considered]) #peds*2*20
                seq_list_setinfo.append(curr_seq_setinfo[:num_peds_considered])
                peds_in_curr_seq_list = peds_in_curr_seq.tolist()
                for i in range(num_peds_considered):
                    peds_in_curr_seq_list.remove(selected_id[i])
                    index.append(num_seqs)

                peds_remains = len(peds_in_curr_seq_list)

                curr_seq_remains = np.zeros((peds_remains, 3, seq_len))

                for i in range(seq_len):
                    curr_frame = frame_data[idx+i]
                    for j in range(len(curr_frame[:,1])):
                        if curr_frame[j,1] in peds_in_curr_seq_list:
                            index_peds = peds_in_curr_seq_list.index(curr_frame[j,1])
                            curr_seq_remains[index_peds, :, i] = curr_frame[j, 2:]

                num_peds_in_seq_remains.append(peds_remains)
                if peds_remains > 0:
                    seq_list_remains.append(curr_seq_remains)
                num_seqs = num_seqs + 1

                bool_outlier = outlier_test(curr_seq[:num_peds_considered], curr_seq_remains)

                if bool_outlier:
                    del num_peds_in_seq[-1]
                    del seq_list[-1]
                    del seq_list_setinfo[-1]
                    del num_peds_in_seq_remains[-1]
                    if peds_remains > 0:
                        del seq_list_remains[-1]
                    for i in range(num_peds_considered):
                        del index[-1]
                    num_seqs = num_seqs - 1

        if len(seq_list) > 0:
            seq_list = np.concatenate(seq_list, axis=0)  #peds*2*20
            seq_list_setinfo = np.concatenate(seq_list_setinfo, axis=0) #peds*2*20
            seq_list_remains = np.concatenate(seq_list_remains, axis=0) #peds*3*20

            seq_list = np.swapaxes(seq_list, 1, 2) #peds*20*2
            seq_list_setinfo = np.swapaxes(seq_list_setinfo, 1, 2) #peds*20*2
            seq_list_remains = np.swapaxes(seq_list_remains, 1, 2)#peds*20*3
            seq_list_vel = calculate_v(seq_list)
            seq_list_remains_vel = calculate_v(seq_list_remains[:,:,:-1])#peds*20*2
            #seq_list_translated = translation(seq_list)

            seq_list_complete = np.concatenate([seq_list, seq_list_vel], axis=-1)
            seq_list_remains_complete = np.zeros((seq_list_remains.shape[0], 20, 5))
            seq_list_remains_complete[:,:,:-1] = np.concatenate([seq_list_remains[:,:,:2], seq_list_remains_vel], axis=-1)
            seq_list_remains_complete[:,:,-1] = seq_list_remains[:,:,-1]
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            cum_start_idx_remains = [0] + np.cumsum(num_peds_in_seq_remains).tolist()
            seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
            seq_start_end_remains = [(start, end) for start, end in zip(cum_start_idx_remains, cum_start_idx_remains[1:])]
            index_start_end = [seq_start_end[index[i]] for i in range(seq_list.shape[0])]
            index_start_end_remains = [seq_start_end_remains[index[i]] for i in range(seq_list.shape[0])]

            supplement = np.zeros((seq_list.shape[0], seq_len, max_peds + 1, 5)) #peds*20*(maxpeds+1)*5
            all_first_part = []
            for i in range(seq_list.shape[0]):
                print('ith:', i)
                first_part = [m for m in range(index_start_end[i][0], index_start_end[i][1])]
                first_part.remove(i)
                all_first_part.append(first_part)
                second_part = [m for m in range(index_start_end_remains[i][0], index_start_end_remains[i][1])]
                for j in range(seq_len):
                    supplement[i, j, :len(first_part), :-1] = seq_list_complete[first_part, j, :]
                    supplement[i, j, :len(first_part), -1] = np.ones_like(supplement[i, j, :len(first_part), -1])

                    second_part_frame = copy.deepcopy(second_part)
                    for t in range(len(second_part)):
                        check_norm_1 = np.linalg.norm(seq_list_remains_complete[second_part[t], j, :2])
                        if check_norm_1 < episa:
                            second_part_frame.remove(second_part[t])
                    supplement[i, j, len(first_part): len(first_part) + len(second_part_frame), :] = seq_list_remains_complete[second_part_frame, j, :]

                    supplement[i, j, -1, 0] = len(first_part)
                    supplement[i, j, -1, 1] = len(first_part) + len(second_part_frame)
        else:
            seq_list_complete = ['empty']
            seq_list_setinfo = ['empty']
            supplement = ['empty']
            all_first_part = ['empty']
            print('check for validation')

        save_name = '../SDD/' + scene + video + '.pickle'

        with open(save_name, 'wb') as f:
            pickle.dump([seq_list_complete, supplement, all_first_part, seq_list_setinfo], f)
        num_success += 1
        print('success:', num_success)
