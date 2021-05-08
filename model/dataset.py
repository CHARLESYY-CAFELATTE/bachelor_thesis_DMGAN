from torch.utils.data import Dataset
import torch
from model.functions import *
import os

class Low_start(Dataset):
    def __init__(self,input_dir,target_dir,image_size):
        self.inputpic=os.listdir(input_dir)
        self.targetpic=[]
        self.image_size=image_size

        for i in range(len(self.inputpic)):
            self.targetpic.append(target_dir+'/'+self.inputpic[i])
            self.inputpic[i]=input_dir+'/'+self.inputpic[i]

        assert self.inputpic is not None

    def __getitem__(self,index):
        input_tensor=get_image_Tensor(self.inputpic[index],self.image_size)
        target_tensor=get_image_Tensor(self.targetpic[index],self.image_size)
        return input_tensor,target_tensor

    def __len__(self):
        return len(self.inputpic)
