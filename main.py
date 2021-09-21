from __future__ import print_function

import argparse
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import numpy as np

from segClass import segClass
from pascalLoader import pascalDataset

import random

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch segmentation')
# hyper-parameters
parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log', type=int, default=1)
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=32, help='testing batch size')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight Decay. Default=0.0001')
parser.add_argument('--mr', type=float, default=0.01, help='mask region')
parser.add_argument('--ms', type=float, default=0.01, help='mask smooth')
parser.add_argument('--ir', type=float, default=0.01, help='foreground center, background center totalvariation')
parser.add_argument('--mt', type=float, default=1, help='class loss weight per mumford shee loss')
parser.add_argument('--classes', '-c', type=str, nargs='*', default=['dog', 'cat'], help='-c aeroplane car')
parser.add_argument('--dp', type=float, default=0, help='dropout probability')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--iter', type=int, default=5, help='laplace heat equation iter')


# model configuration
parser.add_argument('--model', '-m', type=str, default='segClass2', help='choose which model is going to use')
parser.add_argument('--size', type=int, default=224, help='image size for resize')
parser.add_argument('--title', type=str, default='dogCat/calculate_iter5/')
# parser.add_argument('--num', type=int, default=2, help='number of classes')

args = parser.parse_args()

class main:
    def __init__(self):
        self.model = None
        self.train_loader = None
        self.val_loader = None


    def dataLoad(self):
        # ===========================================================
        # Set train dataset & test dataset
        # ===========================================================
        print('===> Loading datasets')

        def seed_worker(self):
            np.random.seed(args.seed)
            random.seed(args.seed)
        
        self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                        root='../../../dataset/dogcat/train',
                                # root='../../dataset/GTSRB/Training',
                        transform=transforms.Compose([
                            transforms.Resize((args.size, args.size)),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                        ])), 
                        batch_size=args.batchSize, shuffle=True, num_workers=2, drop_last=True, worker_init_fn=seed_worker)
        # 테스트 데이터
        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                        root='../../../dataset/dogcat/val', 
                        transform=transforms.Compose([
                            transforms.Resize((args.size, args.size)),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                        ])), 
                        batch_size=args.batchSize, shuffle=True, num_workers=2, drop_last=False, worker_init_fn=seed_worker)

        # labels = args.classes
        # trainSet = pascalDataset(labels=labels, split='train')
        # self.train_loader = torch.utils.data.DataLoader(trainSet,
        #                                 batch_size=args.batchSize, 
        #                                 shuffle=True,
        #                                 num_workers=2,
        #                                 drop_last=True,
        #                                 collate_fn=trainSet.collate_fn,
        #                                 pin_memory=True,
        #                                 )
        # valSet = pascalDataset(labels=labels, split='val')
        # self.val_loader = torch.utils.data.DataLoader(valSet,
        #                                 batch_size=args.batchSize, 
        #                                 shuffle=True,
        #                                 num_workers=2,
        #                                 drop_last=True,
        #                                 collate_fn=valSet.collate_fn,
        #                                 pin_memory=True,
        #                                 )

    def modelCall(self):
        if args.model == 'segClass':
            self.model = segClass(args, self.train_loader, self.val_loader)
        
        else:
            raise Exception("the model does not exist")

    


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

    main = main()
    main.dataLoad()
    main.modelCall()
    
    main.model.runner()