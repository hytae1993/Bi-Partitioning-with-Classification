from __future__ import print_function

import os
import math
from pickle import FALSE

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.classAgnostic import resnet50shared, decoder
from util.loss import Loss
from util.laplace import Laplace
from util.utils import saveModelCA, saveExcelCA, find_jaccard_overlap, get_binarize_mask

from progress_bar import progress_bar
from scheduler_learning_rate import *

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from util.plotSegClass import plot

class segClass(object):
    def __init__(self, config, training_loader, val_loader):
        super(segClass, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.seed = config.seed
        self.nEpochs = config.epoch
        self.lr = config.lr
        self.mr = config.mr
        self.ms = config.ms
        self.ir = config.ir
        self.cw = config.cw
        self.loss = Loss(len(config.classes))
        self.num_class = len(config.classes)
        self.log_interval = config.log
        self.config = config
        
        self.train_loader = training_loader
        self.val_loader = val_loader

        self.encoder = None
        self.maskDecoder = None
        self.foreDecoder = None
        self.backDecoder = None
        self.classifier = None

        self.plot = None

        self.optimizer = {}

        self.mseCriterion = None
        self.crossCriterion = None
        self.l1Criterion = None

        self.train_loss = []
        self.val_loss = []

    def _to_ohe(self, labels):
        ohe = torch.zeros((labels.size(0), self.num_class), requires_grad=False)
        for i, label in enumerate(labels):
            ohe[i, label] = 1

        # ohe = torch.autograd.Variable(ohe)
        ohe = ohe.detach()

        return ohe

    def build_model(self):
        self.maskDecoder = decoder(last=1)
        # self.foreDecoder = decoder(last=3)
        # self.backDecoder = decoder(last=3)
        self.classifier = resnet50shared(pretrained=True)
        self.classifier.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Linear(512, self.num_class)
        )
        self.maskDecoder = self.maskDecoder.to(self.device)
        # self.foreDecoder = self.foreDecoder.to(self.device)
        # self.backDecoder = self.backDecoder.to(self.device)
        self.classifier = self.classifier.to(self.device)

        self.plot = plot(self.train_loader, self.val_loader, self.classifier, self.maskDecoder, self.foreDecoder, self.backDecoder, self.device, self.config)

        self.mseCriterion = torch.nn.MSELoss()
        self.crossCriterion = torch.nn.CrossEntropyLoss()
        self.l1Criterion = torch.nn.L1Loss()

        if self.CUDA:
            cudnn.benchmark = True
            self.mseCriterion.cuda()
            self.crossCriterion.cuda()
            self.l1Criterion.cuda()

        self.optimizer['maskDecoder'] = torch.optim.SGD(self.maskDecoder.parameters(), lr=self.lr, weight_decay=1e-4)
        # self.optimizer['foreDecoder'] = torch.optim.SGD(self.foreDecoder.parameters(), lr=self.lr, weight_decay=1e-4)
        # self.optimizer['backDecoder'] = torch.optim.SGD(self.backDecoder.parameters(), lr=self.lr, weight_decay=1e-4)
        self.optimizer['classifier'] = torch.optim.SGD(self.classifier.parameters(), lr=self.lr, weight_decay=1e-4)

        # self.encoderScheduler = torch.optim.lr_scheduler.StepLR(self.encoderOptimizer, step_size=100, gamma=0.1)
        # self.decoderScheduler = torch.optim.lr_scheduler.StepLR(self.decoderOptimizer, step_size=100, gamma=0.1)

    def run(self, epoch, data_loader, work):
        if work == 'train':
            self.maskDecoder.train()
            # self.foreDecoder.train()
            # self.backDecoder.train()
            self.classifier.train()
        elif work == 'val':
            self.maskDecoder.eval()
            # self.foreDecoder.eval()
            # self.backDecoder.eval()
            self.classifier.eval()

        lossList = []
        foregroundLoss = 0
        backgroundLoss = 0
        maskRegionRegular = 0
        maskSmoothRegular = 0
        foreRegular = 0
        backRegular = 0
        foreClassLoss = 0
        backClassLoss = 0
        acc = 0
        iou = 0

        iter = 0
        num_data = 0

        for batch_num, (data, maskTarget, target) in enumerate(data_loader):
        # for batch_num, (data, target) in enumerate(data_loader):
            iter += 1
            num_data += data.size(0)
            target = target.to(self.device)
            target = target.squeeze()
            maskTarget = maskTarget.to(self.device)
            data = data.to(self.device)

            if work == 'train':

                originOutput, layer = self.classifier(data)
                original = self.crossCriterion(originOutput, target)
                self.optimizer['classifier'].zero_grad()
                original.backward()
                self.optimizer['classifier'].step()

                layers = [l.clone().detach() for l in layer]

                mask = self.maskDecoder(layers)
                # foreCenter = self.foreDecoder(layers)
                # backCenter = self.backDecoder(layers)

                foreground = (1-mask) * data
                background = mask * data   

                # foregroundSegLoss, backgroundSegLoss = self.loss.segmentSmoothLoss(data, mask, foreCenter, backCenter)
                foregroundSegLoss, backgroundSegLoss = self.loss.segmentConstantLoss(data, mask)

                maskRegion = self.loss.regionLoss(1-mask)
                maskSmooth = self.loss.tv(1-mask)
                # foreRegularization = self.loss.tv(foreCenter)
                # backRegularization = self.loss.tv(backCenter)

                foreOutput,_ = self.classifier(foreground) # batch_num * class_num * 1 * 1
                backOutput,_ = self.classifier(background)
                backOutput = nn.Sigmoid()(backOutput)

                foreProb = self.crossCriterion(foreOutput, target)
                backProb = 1
                for i in range(0, backOutput.shape[1]):
                    backProb *= backOutput[:,i]
                backProb = -torch.log(backProb + 0.0005).mean()

                mumfordLoss = foregroundSegLoss + backgroundSegLoss
                # regularization = self.mr * maskRegion + self.ms * maskSmooth + self.ir * foreRegularization + self.ir * backRegularization
                regularization = self.mr * maskRegion + self.ms * maskSmooth
                classLoss = foreProb + backProb

                loss = self.cw * mumfordLoss + classLoss + regularization

                self.optimizer['classifier'].zero_grad()
                self.optimizer['maskDecoder'].zero_grad()
                # self.optimizer['foreDecoder'].zero_grad()
                # self.optimizer['backDecoder'].zero_grad()
                loss.backward()
                self.optimizer['classifier'].step()
                self.optimizer['maskDecoder'].step()
                # self.optimizer['foreDecoder'].step()
                # self.optimizer['backDecoder'].step()
                IoU = 0

            elif work == 'val':
                # continue
                with torch.no_grad():

                    originOutput, layer = self.classifier(data)
                    original = self.crossCriterion(originOutput, target)

                    mask = self.maskDecoder(layer)
                    # foreCenter = self.foreDecoder(layer)
                    # backCenter = self.backDecoder(layer)

                    foreground = (1-mask) * data 
                    background = mask * data   

                    # foregroundSegLoss, backgroundSegLoss = self.loss.segmentSmoothLoss(data, mask, foreCenter, backCenter)
                    foregroundSegLoss, backgroundSegLoss = self.loss.segmentConstantLoss(data, mask)

                    maskRegion = self.loss.regionLoss(1-mask)
                    maskSmooth = self.loss.tv(1-mask)
                    # foreRegularization = self.loss.tv(foreCenter)
                    # backRegularization = self.loss.tv(backCenter)

                    foreOutput,_ = self.classifier(foreground) # batch_num * class_num * 1 * 1
                    backOutput,_ = self.classifier(background)
                    backOutput = nn.Sigmoid()(backOutput)

                    foreProb = self.crossCriterion(foreOutput, target)
                    backProb = 1
                    for i in range(0, backOutput.shape[1]):
                        backProb *= backOutput[:,i]
                    backProb = -torch.log(backProb + 0.0005).mean()

                    mumfordLoss = foregroundSegLoss + backgroundSegLoss
                    # regularization = self.mr * maskRegion + self.ms * maskSmooth + self.ir * foreRegularization + self.ir * backRegularization
                    regularization = self.mr * maskRegion + self.ms * maskSmooth
                    classLoss = foreProb + backProb

                    loss = self.cw * mumfordLoss + classLoss + regularization

                    thresholded = get_binarize_mask(1-mask)
                    IoU = find_jaccard_overlap(maskTarget, thresholded)

            foregroundLoss += (foregroundSegLoss.item() * data.size(0))  
            backgroundLoss += (backgroundSegLoss.item() * data.size(0)) 
            maskRegionRegular += (maskRegion.item() * data.size(0))
            maskSmoothRegular += (maskSmooth.item() * data.size(0))
            # foreRegular += (foreRegularization.item() * data.size(0))
            # backRegular += (backRegularization.item() * data.size(0))
            foreClassLoss += (foreProb.item() * data.size(0))
            backClassLoss += (backProb.item() * data.size(0))
            lossList.append(loss.item())

            pred = foreOutput.max(1, keepdim=True)[1]
            maskCorrect = pred.eq(target.view_as(pred)).sum().item()
            acc += maskCorrect 

            if work == 'val':
                iou += (IoU.item() * data.size(0))
                    
            progress_bar(batch_num, len(data_loader), 'IOU: {:.4f}'.format(iou/num_data))
            # progress_bar(batch_num, len(data_loader),  '{} Mumford: {:.4f}, Reg: {:.4f}, Class: {:.4f}' .format(work, mumfordLoss, regularization, classLoss))

        return foregroundLoss/num_data, backgroundLoss/num_data, np.mean(lossList), np.std(lossList), maskRegionRegular/num_data, maskSmoothRegular/num_data, \
            foreRegular/num_data, backRegular/num_data, foreClassLoss/num_data, backClassLoss/num_data, 100.*acc/num_data

    def runner(self):

        for i in range(11):
            self.train_loss.append([])
            self.val_loss.append([])

        self.build_model()

        # visualize initialize data
        self.plot.plotResult(epoch=0, trainResult=None, valResult=None)

        # scheduler = scheduler_learning_rate_sigmoid_double(self.optimizer, self.nEpochs, [0.01, 0.1], [0.1, 0.00001], [10, 10], [0,0])
        # scheduler.plot()
        # exit(1)

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))

            # scheduler.step()
            # print(self.optimizer['classifier'].param_groups[0]['lr'])

            trainResult = self.run(epoch, self.train_loader, 'train')
            valResult = self.run(epoch, self.val_loader, 'val')

            for i in range(11):
                self.train_loss[i].append(trainResult[i])
                self.val_loss[i].append(valResult[i])

            if epoch % self.log_interval == 0 or epoch == 1:
                self.plot.plotResult(epoch, self.train_loss, self.val_loss)

            if epoch == self.nEpochs:
                saveModelCA(self.classifier, self.maskDecoder, self.foreDecoder, self.backDecoder, self.config)
            #     saveExcelCA(self.train_loss, self.val_loss, self.config)

