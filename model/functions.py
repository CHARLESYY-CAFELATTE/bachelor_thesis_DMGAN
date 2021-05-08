import numpy as np
import cv2 as cv
from PIL import Image
import torch
from torch.autograd import Variable

def parse_model_config(path):
    file=open(path,'r')
    lines=file.read().split('\n')
    lines=[x for x in lines if x and not x.startswith('#')]
    lines=[x.rstrip().lstrip() for x in lines]
    module_defs=[]

    for line in lines:

        if line.startswith('['):
            module_defs.append({})
            module_defs[-1]['type']=line[1:-1].rstrip()

        else:
            key,value=line.split('=')
            value=value.strip()
            module_defs[-1][key.rstrip()]=value.strip()

    return module_defs

def get_image_Tensor(image_path,image_size):
    image=cv.imread(image_path)
    image=cv.resize(image,(image_size,image_size))
    PILimg=np.array(Image.fromarray(cv.cvtColor(image,cv.COLOR_BGR2RGB)))
    PILimg=np.transpose(PILimg,(2,0,1))
    PILimg=PILimg/255.
    PILimg=torch.from_numpy(PILimg)
    Tensor=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    PILimg=Variable(PILimg.type(Tensor))

    return PILimg

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_recostruction(image_Tensor,image_size):
    image_Tensor=image_Tensor*255.
    Tensor=torch.ByteTensor
    image_Tensor=Variable(image_Tensor.type(Tensor))
    image_Tensor=image_Tensor.squeeze(0)
    image_array=image_Tensor.numpy()
    image_array=np.transpose(image_array,(1,2,0))
    image=Image.fromarray(image_array)
    image=cv.cvtColor(np.asarray(image),cv.COLOR_RGB2BGR)

    return image