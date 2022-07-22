import numpy as np
import pandas as pd
import os

data = []
class_dic = {'Biker': 0, 'Pedestrian': 1, 'Skater': 2, 'Cart': 3, 'Car': 4, 'Bus': 5}
step = 12
path = 'annotations/'
scenes = os.listdir(path)
SDD_cols = ['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
for scene in scenes:
    scene_path = path  + scene
    all_videos = os.listdir(scene_path)
    for video in all_videos:
        video_path = scene_path + '/' + video
        file_path = video_path + '/' + 'annotations.txt'

        scene_df = pd.read_csv(file_path, header=0, names=SDD_cols, delimiter=' ')
        scene_df['x'] = (scene_df['xmax'] + scene_df['xmin']) / 2
        scene_df['y'] = (scene_df['ymax'] + scene_df['ymin']) / 2
        scene_df = scene_df[scene_df['lost'] == 0]
        scene_df['frame_rest'] = scene_df['frame'] % step
        scene_df = scene_df[scene_df['frame_rest'] == 0]
        scene_df = scene_df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'occluded', 'generated', 'lost', 'frame_rest'])


        scene_np = scene_df.to_numpy()
        max_frame = int(np.max(scene_np[:, 1])/step + 1)
        pre_process_list = [[] for j in range(max_frame)]
        for i in range(scene_np.shape[0]):
            scene_np[i, 2] = class_dic[scene_np[i, 2]]
            pre_process_list[int(scene_np[i,1]/step)].append(scene_np[i,:])
        pre_process_list_copy = pre_process_list.copy()
        for i in range(len(pre_process_list_copy)):
            if len(pre_process_list_copy[i]) == 0:
                pre_process_list.remove([])

        processed_data = np.concatenate(pre_process_list)
        processed_data_save = np.zeros_like(processed_data)
        processed_data_save[:, 0] = processed_data[:, 1]
        processed_data_save[:, 1] = processed_data[:, 0]
        processed_data_save[:, 2:4] = processed_data[:, 3:5]
        processed_data_save[:, 4] = processed_data[:, 2]

        save_path = video_path + '/processed.npy'
        np.save(save_path, processed_data_save)
