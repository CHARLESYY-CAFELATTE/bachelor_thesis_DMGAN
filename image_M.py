from __future__ import division

import torch
import os
import sys
import argparse
import cv2 as cv

from model.functions import *
from model.models_nobias import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        default='ideal_output',
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
        default='mosaic',
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

    decision_cfg=parse_model_config(opt.config_file+'/decision.cfg')

    Decision=decision(
        decision_cfg=decision_cfg
    ).to(de)

    Decision.load_state_dict(torch.load(opt.weights+'/decision.pth'))

    Decision.eval()

    for process_image in process_image_list:
        image=cv.imread(process_image)
        image=get_image_Tensor(process_image,image_size)
        image=image.unsqueeze(0)
        with torch.no_grad():
            output=Decision(image)
        image=image_recostruction(output,image_size)
        filename=process_image.split('/')[-1]
        cv.imwrite(opt.save_dir+'/'+filename,image)
