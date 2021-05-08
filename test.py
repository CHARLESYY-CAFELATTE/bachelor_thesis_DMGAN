from __future__ import division

from model.models_nobias import *
from model.functions import *
from model.dataset import *

import os
import argparse

import torch

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--testing_folder",
        type=str,
        default='input_image',
        help="the path to original image"
    )

    parser.add_argument(
        "--target_folder",
        type=str,
        default='ideal_output',
        help='the path to construction image'
    )

    parser.add_argument(
        "--weights",
        type=str,
        default='checkpoints',
        help="path to your model weights folder"
    )

    parser.add_argument(
        '--config_file',
        type=str,
        default='config',
        help='path to model config file'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='The size of the training batch'
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='The size of the input image'
    )

    opt = parser.parse_args()

    batch=opt.batch_size

    de=get_device()

    gene_cfg=parse_model_config(opt.config_file+'/gene.cfg')

    Generator=generator(
        gene_cfg=gene_cfg
    ).to(de)

    Generator.load_state_dict(torch.load(opt.weights+'/gene.pth'))

    decision_cfg=parse_model_config(opt.config_file+'/decision.cfg')

    image_size=opt.image_size

    Decision=decision(
        decision_cfg=decision_cfg,
    ).to(de)

    Decision.load_state_dict(torch.load(opt.weights+'/decision.pth'))

    decode_cfg=parse_model_config(opt.config_file+'/decode.cfg')

    Decode=decode(
        decode_cfg=decode_cfg,
        image_size=image_size
    ).to(de)
    
    Decode.load_state_dict(torch.load(opt.weights+'/decode.pth'))

    Generator.eval()

    Decision.eval()
    
    Decode.eval()

    gene_set=torch.utils.data.DataLoader(
        dataset=Low_start(opt.testing_folder,opt.target_folder,image_size),
        batch_size=opt.batch_size,
        shuffle=True
    )

    for i,(pr_pic,target_pic) in enumerate(gene_set):
        with torch.no_grad():
            output=Generator(pr_pic)
            x,loss=Decision(output,targets=pr_pic)
            x,loss_decode=Decode(output,targets=target_pic)
            loss=loss+loss_decode
        print('testing now,the loss is',float(loss))
