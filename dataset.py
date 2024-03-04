import os
import json
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import config

'''
    手部Dataset
'''


class HandData(Dataset):
    def __init__(self, root_path, label_hand_pose_file_list=[]):
        self.hand_pose = []

        for data_file in label_hand_pose_file_list:
            tmp = np.load(os.path.join(root_path, data_file))
            if len(self.hand_pose) > 0:
                self.hand_pose = np.concatenate([self.hand_pose, tmp], axis=0)
            else:
                self.hand_pose = np.concatenate([tmp], axis=0)


    def __getitem__(self, index):
        curr_hand_pose = self.hand_pose[index]  # [15, 3]

        curr_hand_pose = curr_hand_pose.flatten() + config.mean_righthand

        return torch.tensor(curr_hand_pose, dtype=torch.float32)

    def __len__(self):
        return self.hand_pose.shape[0]


if __name__ == '__main__':
    root_path = "./data/"
    batch_size = 1024
    train_dataset = HandData(root_path='./data',
                             label_hand_pose_file_list=['InterHand2.6M_label_hand_pose_train.npy'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = HandData(root_path='./data',
                           label_hand_pose_file_list=['InterHand2.6M_label_hand_pose_val.npy'])
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    for idx, (hand_pose) in enumerate(train_dataloader):
        print(idx, hand_pose.shape)
        # break





