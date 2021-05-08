from __future__ import division

import torch
import os
import sys
import argparse
import cv2 as cv
import time

from model.functions import *
from model.models_nobias import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        default='input_image',
        help="input image or image folder"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default='checkpoints',
        help="path to your model weights file"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default='config',
        help="path to your model setting file"
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help='path to save processed picture'
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="the size of input image"
    )

    opt=parser.parse_args()

    process_image_list=[]

    image_size=opt.image_size

    if os.path.isdir(opt.image):

        image_names=os.listdir(opt.image)

        assert image_names is not None

        for image_name in image_names:
            process_image_list.append(opt.image+'/'+image_name)

    else:

        assert opt.image is not None

        process_image_list.append(opt.image)

    os.makedirs(opt.save_dir,exist_ok=True)

    de=get_device()

    gene_cfg=parse_model_config(opt.config_file+'/gene.cfg')

    Generator=generator(
        gene_cfg=gene_cfg
    ).to(de)

    Generator.load_state_dict(torch.load(opt.weights+'/gene.pth'))

    Generator.eval()

    for process_image in process_image_list:
        image=cv.imread(process_image)
        image=get_image_Tensor(process_image,image_size)
        image=image.unsqueeze(0)
        start=time.clock()
        with torch.no_grad():
            output=Generator(image)
        end=time.clock()
        print(end-start)
        image=image_recostruction(output,image_size)
        filename=process_image.split('/')[-1]
        cv.imwrite(opt.save_dir+'/'+filename,image)
