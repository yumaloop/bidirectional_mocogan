import hickle as hkl
import torch
import torch.utils.data as data
import os
import numpy as np
import pandas as pd
from PIL import Image

class CustomMovingMNIST(data.Dataset):
    def __init__(self, video_path="./data/video/", 
                 frame_size=20, 
                 img_size=(64, 64),
                 transform=None):

        self.video_path = video_path
        self.frame_size = frame_size
        self.img_size = img_size
        self.df_train = pd.read_csv("./data/labels.csv", header=None, names=["frame_id", "label"])
        self.data_num = len(self.df_train)

    def __getitem__(self, index):        
        frame_id = str(self.df_train.loc[index, ['frame_id']].values[0])
        label = int(self.df_train.loc[index, ['label']].values[0])

        npy_path = os.path.join(self.video_path, frame_id + ".npy")
        video = np.load(npy_path)
        video = video[np.newaxis, :, :, :]
        video = video / 255.
        
        return video, label

    def __len__(self):
        return self.data_num
