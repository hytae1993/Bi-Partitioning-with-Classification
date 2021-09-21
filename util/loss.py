import torch
from torch.autograd import Variable
from torch.nn import Module
from torch.nn.functional import conv2d
import torch.nn.functional as F
import numpy as np



class Loss: 
    
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.area_loss_coef = 8
        self.smoothness_loss_coef = 0.5
        self.preserver_loss_coef = 0.3
        self.area_loss_power = 0.3
    
    def tv(self, image):
        x_loss = torch.mean((torch.abs(image[:,:,1:,:] - image[:,:,:-1,:])))
        y_loss = torch.mean((torch.abs(image[:,:,:,1:] - image[:,:,:,:-1])))

        return (x_loss + y_loss)

    def regionLoss(self, image):
        mask_mean = F.avg_pool2d(image, image.size(2), stride=1).squeeze().mean()

        return mask_mean

    def one_hot(self,targets):
        depth = self.num_classes
        if targets.is_cuda:
            return Variable(torch.zeros(targets.size(0), depth).cuda().scatter_(1, targets.long().view(-1, 1).data, 1))
        else:
            return Variable(torch.zeros(targets.size(0), depth).scatter_(1, targets.long().view(-1, 1).data, 1))

    def segmentConstantLoss(self, image, mask):
        # pixel-wise constant segmentation loss
        foreground = (1-mask) * image
        background = mask * image

        foregroundCenter = foreground.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True) / (1-mask).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
        backgroundCenter = background.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True) / mask.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)

        foregroundLoss = ((1-mask) * ((image - foregroundCenter) ** 2)).mean()
        backgroundLoss = (mask * (image - backgroundCenter) ** 2).mean()

        return foregroundLoss, backgroundLoss
  
    def segmentSmoothLoss(self, image, mask, foreCenter, backCenter):
        # pixel-wise smooth segmentation loss

        foregroundLoss = ((1-mask) * ((image - foreCenter)**2)).mean()
        backgroundLoss = ((mask) * ((image - backCenter)**2)).mean()


        return foregroundLoss, backgroundLoss


