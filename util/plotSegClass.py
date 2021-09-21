import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import PIL
import os
from utils import get_image, get_binarize_mask, inverse_normalize

class plot:
    def __init__(self, train_loader, val_loader, classifier, maskDecoder, foreDecoder, backDecoder, device, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.classifier = classifier
        self.maskDecoder = maskDecoder
        self.foreDecoder = foreDecoder
        self.backDecoder = backDecoder
        self.device = device
        self.config = config

    def convert_image_np(self, inp, image):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array((0.485, 0.456, 0.406))
        std = np.array((0.229, 0.224, 0.225))
        if image:
            # inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

        return inp

    def visualize_stn(self, loader):
        with torch.no_grad():
            data, target, _ = next(iter(loader))

            # data, target = next(iter(loader))
            # target = torch.zeros(16, 3, 224, 224)

            data = data.to(self.device)
            target = target.to(self.device)
            data = data[:16]
            ## inverse normalization
            # data = inverse_normalize(data)
            target = target[:16]

            input_tensor = data.cpu()
            target_tensor = target.cpu()
            mask, foreground, background = get_image(data, self.classifier, self.maskDecoder, self.foreDecoder, self.backDecoder)
            # mask, foreground, background = get_image(data, self.encoder, self.maskDecoder, self.maskDecoder, self.maskDecoder)      # channen-vesse로 해서 fore, back matrix 그림 없음
            binarized_mask = get_binarize_mask(1-mask)

            in_grid = self.convert_image_np(
                torchvision.utils.make_grid(input_tensor, nrow=4), True)
            
            mask_grid = self.convert_image_np(
                torchvision.utils.make_grid(1-mask, nrow=4), False)
            
            fore_matrix_grid = self.convert_image_np(
                torchvision.utils.make_grid(foreground, nrow=4), True)

            back_matrix_grid = self.convert_image_np(
                torchvision.utils.make_grid(background, nrow=4), True)
            
            target_grid = self.convert_image_np(
                torchvision.utils.make_grid(target_tensor, nrow=4), True)

            fore_grid = in_grid * mask_grid
            back_grid = in_grid * (1-mask_grid)

            binarized_mask_grid = self.convert_image_np(
                torchvision.utils.make_grid(binarized_mask, nrow=4), False)

            binarized_fore_grid = in_grid * binarized_mask_grid
            binarized_back_grid = in_grid * (1 - binarized_mask_grid)

            mask_fore_grid = target_grid * in_grid
            mask_back_grid = (1-target_grid) * in_grid

            plt.close('all')
            fig = plt.figure(figsize=(16,12))
            fig.tight_layout()
            ax1 = fig.add_subplot(4,3,2)

            ax2 = fig.add_subplot(4,3,4)
            ax3 = fig.add_subplot(4,3,5)
            ax4 = fig.add_subplot(4,3,6)

            ax5 = fig.add_subplot(4,3,7)
            ax6 = fig.add_subplot(4,3,8)
            ax7 = fig.add_subplot(4,3,9)

            ax8 = fig.add_subplot(4,3,10)
            ax9 = fig.add_subplot(4,3,11)
            ax10 = fig.add_subplot(4,3,12)

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
            ax6.axes.get_xaxis().set_visible(False)
            ax6.axes.get_yaxis().set_visible(False)
            ax7.axes.get_xaxis().set_visible(False)
            ax7.axes.get_yaxis().set_visible(False)
            ax8.axes.get_xaxis().set_visible(False)
            ax8.axes.get_yaxis().set_visible(False)
            ax9.axes.get_xaxis().set_visible(False)
            ax9.axes.get_yaxis().set_visible(False)
            ax10.axes.get_xaxis().set_visible(False)
            ax10.axes.get_yaxis().set_visible(False)

            ax1.imshow(in_grid)
            ax1.set_title('original')

            maskPic = ax2.imshow(mask_grid, cmap='gray', vmin=0, vmax=1)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(maskPic, cax=cax, orientation='vertical')
            ax2.set_title('mask')

            ax3.imshow(fore_grid)
            ax3.set_title('foreground')

            ax4.imshow(back_grid)
            ax4.set_title('background')

            binarizedMaskPic = ax5.imshow(binarized_mask_grid, cmap='gray', vmin=0, vmax=1)
            divider2 = make_axes_locatable(ax5)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(binarizedMaskPic, cax=cax2, orientation='vertical')
            ax5.set_title('binarized mask')

            ax6.imshow(binarized_fore_grid)
            ax6.set_title('binarized foreground')

            ax7.imshow(binarized_back_grid)
            ax7.set_title('binarized background')

            ax8.imshow(target_grid)
            ax8.set_title('mask truth')

            ax9.imshow(mask_fore_grid)
            ax9.set_title('foreground truth')

            ax10.imshow(mask_back_grid)
            ax10.set_title('background truth')

            # plt.title('{}'.format(count))
            plt.tight_layout()     

            return fig

    def visualize_loss(self, trainResult, valResult):
        plt.clf()
        figure, axarr = plt.subplots(1, 4, figsize=(32,8))

        axarr[0].plot(trainResult[0], 'r-', label='train foreground chanvese')
        axarr[0].plot(trainResult[1], 'g-', label='train background chanvese')
        axarr[0].plot(valResult[0], 'b-', label='val foreground chanvese')
        axarr[0].plot(valResult[1], 'c-', label='val background chanvese')
        axarr[0].set_title('chanvese/tempt loss')
        axarr[0].legend(loc='upper left')
        
        axarr[1].plot(trainResult[4], 'r-', label='train mask region reg')
        axarr[1].plot(trainResult[5], 'g-', label='train mask smooth reg')
        axarr[1].plot(trainResult[6], 'b-', label='train fore reg')
        axarr[1].plot(trainResult[7], 'c-', label='train back reg')
        axarr[1].plot(valResult[4], 'y-', label='val mask region reg')
        axarr[1].plot(valResult[5], 'k-', label='val mask smooth reg')
        axarr[1].plot(valResult[6], 'indigo', label='val fore reg')
        axarr[1].plot(valResult[7], 'm-', label='val back reg')
        axarr[1].set_title('chanvese/tempt regular')
        axarr[1].legend(loc='upper left')

        axarr[2].plot(trainResult[8], 'r-', label='train fore loss')
        axarr[2].plot(valResult[8], 'b-', label='val fore loss')
        axarr[2].set_title('class loss')
        axarr[2].legend(loc='upper left')
        twin = axarr[2].twinx()
        twin.plot(trainResult[9], 'g-', label='train back loss')
        twin.plot(valResult[9], 'c-', label='val back loss')
        twin.legend(loc='upper right')

        axarr[3].plot(trainResult[2], 'r-', label='train loss')
        axarr[3].fill_between(range(len(trainResult[2])),np.array(trainResult[2])-np.array(trainResult[3]), np.array(trainResult[2])+np.array(trainResult[3]),alpha=.1, color='r')
        axarr[3].plot(valResult[2], 'b-', label='val loss')
        axarr[3].fill_between(range(len(valResult[2])),np.array(valResult[2])-np.array(valResult[3]), np.array(valResult[2])+np.array(valResult[3]),alpha=.1, color='b')
        axarr[3].legend(loc='upper left')
        axarr[3].set_title('total loss and accuracy')
        twin = axarr[3].twinx()
        twin.plot(trainResult[10], 'g-', label='train acc')
        twin.plot(valResult[10], 'c-', label='val acc')
        twin.legend(loc='upper right')

        plt.tight_layout()    

        return figure

   
    def plotResult(self, epoch, trainResult, valResult):

        if epoch != 0: 
            # visualize train data
            trainPic1 = self.visualize_stn(self.train_loader)
            try:
                trainPic1.savefig('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/train/result_{}.png'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw, epoch))
            except FileNotFoundError:
                os.makedirs('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/train'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw))
                trainPic1.savefig('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/train/result_{}.png'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw, epoch))
                

            # visualize validation data
            trainPic3 = self.visualize_stn(self.val_loader)
            try:
                trainPic3.savefig('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/val/result_{}.png'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw, epoch))
            except FileNotFoundError:
                os.makedirs('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/val'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw, epoch))
                trainPic3.savefig('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/val/result_{}.png'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw, epoch))
                
            

            # visualize loss graph
            loss = self.visualize_loss(trainResult, valResult)
            try:
                loss.savefig('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/graph/loss.png'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw))
            except FileNotFoundError:
                os.makedirs('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/graph'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw))
                loss.savefig('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/graph/loss.png'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw))

        elif epoch == 0:
            # visualize train data
            trainPic1 = self.visualize_stn(self.train_loader)
            try:
                trainPic1.savefig('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/train/result_{}.png'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw, epoch))
            except FileNotFoundError:
                os.makedirs('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/train'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw))
                trainPic1.savefig('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/train/result_{}.png'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw, epoch))
                

            # visualize validation data
            trainPic3 = self.visualize_stn(self.val_loader)
            try:
                trainPic3.savefig('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/val/result_{}.png'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw, epoch))
            except FileNotFoundError:
                os.makedirs('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/val'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw, epoch))
                trainPic3.savefig('../../../result/masterMask/access/pascal/stepLR/chanvese/tempt/mr_{}_ms_{}_ir_{}/cw_{}/pic/val/result_{}.png'.format(self.config.mr, self.config.ms, self.config.ir, self.config.cw, epoch))