from __future__ import division

from model.models_nobias import *
from model.functions import *
from model.dataset import *

import os
import sys
import argparse

import torch
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traning_folder",
        type=str,
        default='dataset/real',
        help="the path to original image"
    )

    parser.add_argument(
        "--target_folder",
        type=str,
        default='dataset/construction',
        help='the path to construction image'
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='checkpoints',
        help='output checkpoints path'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        default='config',
        help='path to model config file'
    )

    parser.add_argument(
        '--epoch_save_frequency',
        type=int,
        default=50,
        help='How often you want your weights saved'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='The size of the training batch'
    )

    parser.add_argument(
        '--decay_rate',
        type=int,
        default=100,
        help='How offen the training rate dacay'
    )

    parser.add_argument(
        '--testing_batch_size',
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

    parser.add_argument(
        "--testing_folder",
        type=str,
        default='input_image',
        help="the path to original image"
    )

    parser.add_argument(
        "--test_target_folder",
        type=str,
        default='ideal_output',
        help='the path to construction image'
    )

    parser.add_argument(
        "--start",
        type=bool,
        default=True,
        help='start a new train or load weights last time'
    )

    parser.add_argument(
        "--weights",
        type=str,
        default='checkpoints',
        help="path to your model weights folder"
    )

    parser.add_argument(
        "--init_lr",
        type=float,
        default=1e-3,
        help="initial learning rate"
    )

    opt = parser.parse_args()

    os.makedirs(opt.checkpoint_path,exist_ok=True)

    batch=opt.batch_size

    de=get_device()

    gene_cfg=parse_model_config(opt.config_file+'/gene.cfg')

    Generator=generator(
        gene_cfg=gene_cfg
    ).to(de)

    decision_cfg=parse_model_config(opt.config_file+'/decision.cfg')

    image_size=opt.image_size
    Decision=decision(
        decision_cfg=decision_cfg
    ).to(de)

    decode_cfg=parse_model_config(opt.config_file+'/decode.cfg')

    Decode=decode(
        decode_cfg=decode_cfg,
        image_size=image_size
    ).to(de)

    init_lr=opt.init_lr

    lr=init_lr

    optimizer_Decision=optim.Adam([
            {'params':Decision.Network.parameters()},
            {'params':Decision.end.parameters(),'lr':lr*0.1}
        ],
        lr=lr
    )

    optimizer_Generator=optim.Adam([
            {'params':Generator.Network.parameters()},
            {'params':Generator.end.parameters(),'lr':lr*0.1}
        ],
        lr=lr
    )

    data_set=torch.utils.data.DataLoader(
        dataset=Low_start(opt.traning_folder,opt.target_folder,image_size),
        batch_size=opt.batch_size,
        shuffle=True
    )

    test_set=torch.utils.data.DataLoader(
        dataset=Low_start(opt.testing_folder,opt.test_target_folder,image_size),
        batch_size=opt.testing_batch_size,
        shuffle=True
    )

    epoch_count_gene=1

    epoch_count_decision=1

    epoch_count_decode=1

    decay_percetage=0.6

    if opt.start is not True:

        Generator.load_state_dict(torch.load(opt.weights+'/gene.pth'))

        Decision.load_state_dict(torch.load(opt.weights+'/decision.pth'))

        Decode.load_state_dict(torch.load(opt.weights+'/decode.pth'))

        optimizer_Decision=optim.Adam(
            params=Decision.parameters(),
            lr=lr*decay_percetage
        )

        optimizer_Generator=optim.Adam(
            params=Generator.parameters(),
            lr=lr*decay_percetage
        )

    loss_last=1.0

    decay_rate=opt.decay_rate

    assert opt.epoch_save_frequency>0

    assert opt.decay_rate>0

    while epoch_count_gene<1010:

        Decode.eval()

        Decision.train()

        Generator.train()

        for j in range(5):
            for i,(true_pic,target_pic) in enumerate(data_set):

                optimizer_Decision.zero_grad()

                x,loss=Decision(target_pic,targets=true_pic)

                loss.backward()

                optimizer_Decision.step()

                print('epoch:',epoch_count_decision,'batch:',i+1,'training decision, current loss is:',float(loss))

            epoch_count_decision+=1

        for i,(target_pic,true_pic) in enumerate(data_set):

            optimizer_Generator.zero_grad()

            x=Generator(target_pic)

            x,loss=Decision(x,targets=target_pic)

            loss.backward()

            optimizer_Generator.step()

            print('epoch:',epoch_count_gene,'batch:',i+1,'training gene, current loss is:',float(loss))

        epoch_count_gene+=1

        for i,(target_pic,true_pic) in enumerate(data_set):

            optimizer_Generator.zero_grad()

            x=Generator(target_pic)

            x,loss=Decode(x,targets=true_pic)

            loss.backward()

            optimizer_Generator.step()

            print('epoch:',epoch_count_decode,'batch:',i+1,'training gene, current loss is:',float(loss))

        epoch_count_decode+=1

        if epoch_count_gene % opt.epoch_save_frequency == 0:

            Generator.eval()

            Decision.eval()

            for i,(pr_pic,target_pic) in enumerate(test_set):

                with torch.no_grad():
                    output=Generator(pr_pic)
                    x,loss=Decision(output,targets=pr_pic)
                    x,loss_decode=Decode(output,targets=target_pic)
                    loss=loss+loss_decode
                print('testing',str(i),float(loss))

                if float(loss)<loss_last:
                    torch.save(Generator.state_dict(),opt.checkpoint_path+'/gene.pth')
                    torch.save(Decode.state_dict(),opt.checkpoint_path+'/decode.pth')
                    torch.save(Decision.state_dict(),opt.checkpoint_path+'/decision.pth')
                    loss_last=float(loss)

                else:
                    torch.save(Generator.state_dict(),opt.checkpoint_path+'/gene_train.pth')
                    torch.save(Decode.state_dict(),opt.checkpoint_path+'/decode.pth')
                    torch.save(Decision.state_dict(),opt.checkpoint_path+'/decision_train.pth')

        if epoch_count_gene % decay_rate == 0:

            lr=lr*decay_percetage

            optimizer_Decision=optim.Adam([
                    {'params':Decision.Network.parameters()},
                    {'params':Decision.end.parameters(),'lr':lr*0.1}
                ],
                lr=lr
            )

            optimizer_Generator=optim.Adam([
                    {'params':Generator.Network.parameters()},
                    {'params':Generator.end.parameters(),'lr':lr*0.1}
                ],
                lr=lr
            )
