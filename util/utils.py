import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

SMOOTH = 1e-6

def find_jaccard_overlap(outputs, labels, eps=1e-5):
    outputs = outputs.int()
    labels = labels.int()
    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)
    
    intersection = torch.logical_and(labels, outputs)
    union = torch.logical_or(labels, outputs)
    iou_score = torch.sum(intersection) / torch.sum(union)

    return iou_score

def get_image(input, classifier, maskDecoder, foreDecoder, backDecoder):
    with torch.no_grad():
        _, layers = classifier(input)
        mask = maskDecoder(layers)
        mask, foreground, background = mask.cpu(), mask.cpu(), mask.cpu()

        return mask, mask, mask

def get_imageMA(input, encoder, maskDecoder, foreDecoder, backDecoder):
    with torch.no_grad():
        block, bottleNeck = encoder(input)
        mask = maskDecoder(block, bottleNeck)
        foreCenter = foreDecoder(block, bottleNeck)
        backCenter = backDecoder(block, bottleNeck)
        mask, foreground, background = mask.cpu(), foreCenter.cpu(), backCenter.cpu()

        return mask, foreground, background

def get_binarize_mask(mask):
    with torch.no_grad():
        out = (mask>0.5).float()

        return out

def inverse_normalize(tensor):
    mean = [0.3337, 0.3064, 0.3171]
    std = [0.2672, 0.2564, 0.2629]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        
    return tensor

def threshold(tensor):
    zero_threshold = nn.Threshold(0.5, 0)
    one_threshold = nn.Threshold(-0.5, -1)

    output = zero_threshold(tensor)
    output *= -1

    output = one_threshold(output)
    output *= -1

    return output

def saveModelCA(classifier, maskDecoder, foreDecoder, backDecoder, args):
    try:
        torch.save(maskDecoder.state_dict(), './savedModel/Chan_class/dogCat_maskDecoder_sigBack_mr_{}_ms_{}_ir_{}_cw_{}'.format(args.mr, args.ms, args.ir, args.cw))
        # torch.save(foreDecoder.state_dict(), './savedModel/Mumford/dogCat_foreDecoder_sigBack_mr_{}_ms_{}_ir_{}_cw_{}'.format(args.mr, args.ms, args.ir, args.cw))
        # torch.save(backDecoder.state_dict(), './savedModel/Mumford/dogCat_backDecoder_sigBack_mr_{}_ms_{}_ir_{}_cw_{}'.format(args.mr, args.ms, args.ir, args.cw))
        torch.save(classifier.state_dict(), './savedModel/Chan_class/dogCat_classifier_sigBack_mr_{}_ms_{}_ir_{}_cw_{}'.format(args.mr, args.ms, args.ir, args.cw))
    except FileNotFoundError:
        os.makedirs('./savedModel/Chan_class')
        torch.save(maskDecoder.state_dict(), './savedModel/Chan_class/dogCat_maskDecoder_sigBack_mr_{}_ms_{}_ir_{}_cw_{}'.format(args.mr, args.ms, args.ir, args.cw))
        # torch.save(foreDecoder.state_dict(), './savedModel/Mumford/dogCat_foreDecoder_sigBack_mr_{}_ms_{}_ir_{}_cw_{}'.format(args.mr, args.ms, args.ir, args.cw))
        # torch.save(backDecoder.state_dict(), './savedModel/Mumford/dogCat_backDecoder_sigBack_mr_{}_ms_{}_ir_{}_cw_{}'.format(args.mr, args.ms, args.ir, args.cw))
        torch.save(classifier.state_dict(), './savedModel/Chan_class/dogCat_classifier_sigBack_mr_{}_ms_{}_ir_{}_cw_{}'.format(args.mr, args.ms, args.ir, args.cw))

def saveExcelCA(trainResult, valResult, args):
    trainResult = np.transpose(trainResult)
    train_df = pd.DataFrame(trainResult, columns=['foregroun loss', 'background loss', 'total loss mean', 'total loss std', 'mask L1', 'mask TV', 'foreground TV', 'background TV', 'foreground class loss', 'background class loss', 'accuracy'])

    valResult = np.transpose(valResult)
    val_df = pd.DataFrame(valResult, columns=['foregroun loss', 'background loss', 'total loss mean', 'total loss std', 'mask L1', 'mask TV', 'foreground TV', 'background TV', 'foreground class loss', 'background class loss', 'accuracy'])

    try:
        train_df.to_csv('./excel/Mumford/train/sigBack_mr_{}_ms_{}_ir_{}_cw_{}.csv'.format(args.mr, args.ms, args.ir, args.cw))
    except FileNotFoundError:
        os.makedirs('./excel/Mumford/train')
        train_df.to_csv('./excel/Mumford/train/sigBack_mr_{}_ms_{}_ir_{}_cw_{}.csv'.format(args.mr, args.ms, args.ir, args.cw))

    try:
        val_df.to_csv('./excel/Mumford/val/sigBack_mr_{}_ms_{}_ir_{}_cw_{}.csv'.format(args.mr, args.ms, args.ir, args.cw))
    except FileNotFoundError:
        os.makedirs('./excel/Mumford/val')
        val_df.to_csv('./excel/Mumford/val/sigBack_mr_{}_ms_{}_ir_{}_cw_{}.csv'.format(args.mr, args.ms, args.ir, args.cw))


def saveModelMA(encoder, maskDecoder, foreDecoder, backDecoder, args):
    try:
        torch.save(maskDecoder.state_dict(), './savedModel/Chan_vese/dogCat_maskDecoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
        # torch.save(foreDecoder.state_dict(), './savedModel/Chan_vese/dogCat_foreDecoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
        # torch.save(backDecoder.state_dict(), './savedModel/Chan_vese/dogCat_backDecoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
        torch.save(encoder.state_dict(), './savedModel/Chan_vese/dogCat_encoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
    except FileNotFoundError:
        os.makedirs('./savedModel/Chan_vese')
        torch.save(maskDecoder.state_dict(), './savedModel/Chan_vese/dogCat_maskDecoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
        # torch.save(foreDecoder.state_dict(), './savedModel/Chan_vese/dogCat_foreDecoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
        # torch.save(backDecoder.state_dict(), './savedModel/Chan_vese/dogCat_backDecoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
        torch.save(encoder.state_dict(), './savedModel/Chan_vese/dogCat_encoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))

def saveExcelMA(trainResult, valResult, args):
    trainResult = np.transpose(trainResult)
    train_df = pd.DataFrame(trainResult, columns=['foreground loss', 'background loss', 'total loss mean', 'total loss std', 'mask L1', 'mask TV', 'foreground TV', 'background TV'])

    valResult = np.transpose(valResult)
    val_df = pd.DataFrame(valResult, columns=['foreground loss', 'background loss', 'total loss mean', 'total loss std', 'mask L1', 'mask TV', 'foreground TV', 'background TV'])

    try:
        train_df.to_csv('./excel/Mumford/train/dogCat_mr_{}_ms_{}_ir_{}.csv'.format(args.mr, args.ms, args.ir))
    except FileNotFoundError:
        os.makedirs('./excel/Mumford/train')
        train_df.to_csv('./excel/Mumford/train/dogCat_mr_{}_ms_{}_ir_{}.csv'.format(args.mr, args.ms, args.ir))

    try:
        val_df.to_csv('./excel/Mumford/val/dogCat_mr_{}_ms_{}_ir_{}.csv'.format(args.mr, args.ms, args.ir))
    except FileNotFoundError:
        os.makedirs('./excel/Mumford/val')
        val_df.to_csv('./excel/Mumford/val/dogCat_mr_{}_ms_{}_ir_{}.csv'.format(args.mr, args.ms, args.ir))