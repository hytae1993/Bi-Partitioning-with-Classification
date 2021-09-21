from __future__ import print_function

import os
import math

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.UNet import Encoder, maskDecoder, foreDecoder, backDecoder
from model.classAgnostic import resnet50shared, decoder

from scheduler_learning_rate import *

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

class picture(object):
    def __init__(self, config, val_loader):
        super(picture, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.seed = config.seed
        self.lr = config.lr
        self.mr = config.mr
        self.ms = config.ms
        self.ir = config.ir
        self.cw = config.cw
        self.config = config
        
        self.val_loader = val_loader

        self.encoder = None
        self.maskDecoder = None
        self.foreDecoder = None
        self.backDecoder = None
        self.classifier = None


    def build_model(self):
        classifer_model_name = '../../model_excel_SAVED/savedModel/Chan/' + 'dogCat_classifier' + '_sigBack_mr_' + str(self.mr) + '_ms_' + str(self.ms) + '_ir_' + str(self.ir) + '_cw_' + str(self.cw)
        mask_model_name = '../../model_excel_SAVED/savedModel/Chan/' + 'dogCat_maskDecoder' + '_sigBack_mr_' + str(self.mr) + '_ms_' + str(self.ms) + '_ir_' + str(self.ir) + '_cw_' + str(self.cw)
        # forDecoder_model_name = '../../model_excel_SAVED/savedModel/Mumford/' + 'dogCat_foreDecoder' + '_sigBack_mr_' + str(self.mr) + '_ms_' + str(self.ms) + '_ir_' + str(self.ir) + '_cw_' + str(self.cw)
        # backDecoder_model_name = '../../model_excel_SAVED/savedModel/Mumford/' + 'dogCat_backDecoder' + '_sigBack_mr_' + str(self.mr) + '_ms_' + str(self.ms) + '_ir_' + str(self.ir) + '_cw_' + str(self.cw)

        self.maskDecoder = decoder(last=1)
        self.foreDecoder = decoder(last=3)
        self.backDecoder = decoder(last=3)
        self.classifier = resnet50shared()

        # self.encoder = Encoder()
        # self.maskDecoder = maskDecoder()

        # encoder = torch.load(enocder_model_name)
        maskdecoder = torch.load(mask_model_name)
        classifier = torch.load(classifer_model_name)
        # foredecoder = torch.load(forDecoder_model_name)
        # backdecoder = torch.load(backDecoder_model_name)

        # self.encoder.load_state_dict(encoder)
        self.maskDecoder.load_state_dict(maskdecoder)
        self.classifier.load_state_dict(classifier)
        # self.foreDecoder.load_state_dict(foredecoder)
        # self.backDecoder.load_state_dict(backdecoder)

        # self.encoder = self.encoder.to(self.device)
        self.maskDecoder = self.maskDecoder.to(self.device)
        self.classifier = self.classifier.to(self.device)
        # self.foreDecoder = self.foreDecoder.to(self.device)
        # self.backDecoder = self.backDecoder.to(self.device)

    def convert_image_np(self, inp, image):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array((0.485, 0.456, 0.406))
        std = np.array((0.229, 0.224, 0.225))
        if image:
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

        return inp


    def run(self, data_loader):
        # self.encoder.eval()
        self.maskDecoder.eval()
        self.classifier.eval()

        for batch_num, (data, maskTarget, target) in enumerate(data_loader):
            target = target.to(self.device)
            maskTarget = maskTarget.to(self.device)
            data = data.to(self.device)

            with torch.no_grad():
                originOutput, layer = self.classifier(data)
                mask = self.maskDecoder(layer)

                # block, bottleNeck = self.encoder(data)
                # mask = self.maskDecoder(block, bottleNeck)


                data = data.cpu()
                mask = mask.cpu()
                maskTarget = maskTarget.cpu()

                in_grid = self.convert_image_np(
                    torchvision.utils.make_grid(data, nrow=1), True)
            
                mask_grid = self.convert_image_np(
                    torchvision.utils.make_grid(1-mask, nrow=1), False)

                target_grid = self.convert_image_np(
                    torchvision.utils.make_grid(maskTarget, nrow=1), False)

                fig = plt.figure(figsize=(3,15))

                ax1 = plt.subplot(511,aspect = 'equal')
                ax2 = plt.subplot(512,aspect = 'equal')
                ax3 = plt.subplot(513,aspect = 'equal')
                ax4 = plt.subplot(514,aspect = 'equal')
                ax5 = plt.subplot(515,aspect = 'equal')
                # plt.subplots_adjust(left=0, bottom=0, right=0.5, top=0.5, wspace=0, hspace=0)
                # plt.tight_layout()
                # fig.tight_layout()
                # ax1 = fig.add_subplot(4,1,1)
                # ax2 = fig.add_subplot(4,1,2)
                # ax3 = fig.add_subplot(4,1,3)
                # ax4 = fig.add_subplot(4,1,4)

                ax1.axes.get_xaxis().set_visible(False)
                ax1.axes.get_yaxis().set_visible(False)
                ax2.axes.get_xaxis().set_visible(False)
                ax2.axes.get_yaxis().set_visible(False)
                ax3.axes.get_xaxis().set_visible(False)
                ax3.axes.get_yaxis().set_visible(False)
                ax4.axes.get_xaxis().set_visible(False)
                ax4.axes.get_yaxis().set_visible(False)
                ax5.axes.get_xaxis().set_visible(False)
                ax5.axes.get_yaxis().set_visible(False)

                ax1.imshow(in_grid)
                ax2.imshow(target_grid, cmap='gray', vmin=0, vmax=1)
                ax3.imshow(mask_grid, cmap='gray', vmin=0, vmax=1)
                ax4.imshow(in_grid * mask_grid)
                ax5.imshow(in_grid * (1-mask_grid))

                

                try:
                    fig.savefig('../../result/access/pascal/paper_image/Chanvese/mr_' + str(self.mr) + '_ms_' + str(self.ms) + '_ir_' + str(self.ir) + '/{}.png'.format(batch_num),bbox_inches='tight')
                
                except FileNotFoundError:
                    os.makedirs('../../result/access/pascal/paper_image/Chanvese/mr_' + str(self.mr) + '_ms_' + str(self.ms) + '_ir_' + str(self.ir) + '/')
                    fig.savefig('../../result/access/pascal/paper_image/Chanvese/mr_' + str(self.mr) + '_ms_' + str(self.ms) + '_ir_' + str(self.ir) + '/{}.png'.format(batch_num),bbox_inches='tight')


    def runner(self):

        self.build_model()

        self.run(self.val_loader)

