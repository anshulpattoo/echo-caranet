# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:41:30 2021

@author: angelou
"""

import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataloader import get_loader,test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter, get_mean_and_std
from utils.echo import Echo
import torch.nn.functional as F
import numpy as np
from torchstat import stat
from CaraNet import caranet
import time
from visualization import VisdomPlotter
import cv2
from collections import deque
import tensorflow as tf
from tensorflow.keras import backend as K

# Evaluation metrics
def numpy_iou(y_true, y_pred, n_class=1):
    IOU = []
    for c in range(n_class):
        for i in range(y_true.shape[0]):
            c = 1

            TP = np.sum((y_true[i] == c) & (y_pred[i] == c))
            FP = np.sum((y_true[i] != c) & (y_pred[i] == c))
            FN = np.sum((y_true[i] == c) & (y_pred[i] != c))

            n = TP
            d = float(TP + FP + FN + 1e-12)

            iou = np.divide(n, d)
            IOU.append(iou)
    return np.mean(IOU)

def numpy_mean_iou(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    if (type(y_pred).__module__ == torch.__name__):
        y_pred = y_pred.cpu().detach().numpy()
    prec = []
    for t in np.arange(0.5, 1.0, 0.5):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score = tf.numpy_function(numpy_iou, [y_true, y_pred_], tf.float64)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def numpy_mean_dice(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    if (type(y_pred).__module__ == torch.__name__):
        y_pred = y_pred.cpu().detach().numpy()
    prec = []
    for t in np.arange(0.5, 1.0, 0.5):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score = tf.numpy_function(dice, [y_true, y_pred_], tf.float64)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def structure_loss(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()

def test(model, path):
    
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####
    
    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, 352)
    b=0.0
    print('[test_size]',test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        res5,res3,res2,res1 = model(image)
        res = res5
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))
 
        intersection = (input_flat*target_flat)
        
        loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        a =  '{:.4f}'.format(loss)
        a = float(a)
        b = b + a
        
    return b/60

def train(train_loader, val_loader, model, optimizer, epoch, test_path):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record1, loss_record2, loss_record3, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    visualization = VisdomPlotter()

    image = np.full([3, 256, 256], 0, dtype=float)
    window = visualization.plot_img(image)
    window2 = visualization.plot_img(image)
    window3 = visualization.plot_img(image)
    window4 = visualization.plot_img(image)
    window5 = visualization.plot_img(image)
    window6 = visualization.plot_img(image)

    # Line plots (one for each candidate segmentation)
    passes = []

    loss_vals = {'loss_record5': [], 'loss_record3': [], 'loss_record2': [], 'loss_record1': [], 'total_loss': []} # Value in each key-value pair represents a running list of loss values

    window7 = None

    X = deque([])
    Y = deque([])
    step = 0

    frame_list = ["Diastolic", "Systolic"] # ind. 0 is diastolic; ind. 1 is systolic

    pass_val = 0
    update = False

    for i, pack in enumerate(train_loader, start=1):
        # Same image, but different sizes, resulting in slightly different segmentations
        for j, rate in enumerate(size_rates, start=1):
            for k, frame in enumerate(frame_list):
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                
                # `images` represents the full set of frames in the video
                # `gts` has relevant frames for tracing, along with segmentations
                images = Variable(gts[k]).cuda()  

                img_np_arr = gts[k][0].cpu().detach().numpy()

                gt = gts[k][0].cpu().detach().numpy()
                norm_image = cv2.normalize(img_np_arr, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                norm_image = norm_image.astype(np.uint8)

                norm_gt = cv2.normalize(gt, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                norm_gt = norm_gt.astype(np.uint8)

                norm_image_prev = norm_image
                norm_gt_prev = norm_gt

                gts = Variable(gts[k + 2]).cuda()

                gts = gts.reshape([6, 1, 112, 112])
                # ---- rescale ----
                trainsize = int(round(opt.trainsize*rate/32)*32)
                # if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                
                # ---- forward ----
                lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1 = model(images)
                # ---- loss function ----
                loss5 = structure_loss(lateral_map_5, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss1 = structure_loss(lateral_map_1, gts)
                            
                loss = loss5 + loss3 + loss2 + loss1

                step += 1

                X.appendleft(step)
                Y.appendleft(loss.cpu().detach().item())

                X_np = np.array(list(X))
                Y_np = np.array(list(Y))

                # Convert candidate segmentations to correct grayscale images
                lateral_maps = [lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_5]
                for j in range(len(lateral_maps)):
                    individual_maps = []
                    for k in range(len(lateral_maps[j])):
                        if (type(lateral_maps[j]).__module__ == torch.__name__):
                            individual_map = lateral_maps[j][k].cpu().detach().numpy()
                        ret, individual_map = cv2.threshold(individual_map, 1, 255, cv2.THRESH_BINARY)
                        individual_maps.append(individual_map)
                    lateral_maps[j] = np.array(individual_maps)

                if (i % 2 == 0):
                    visualization.plot_img(norm_image, window, f'{frame} Frame')
                    visualization.plot_img(norm_gt, window2, "Ground Truth Segmentation")     
                    visualization.plot_img(lateral_map_5[0], window3, "Candidate Segmentation 1")
                    visualization.plot_img(lateral_map_3[0], window4, "Candidate Segmentation 2")
                    visualization.plot_img(lateral_map_2[0], window5, "Candidate Segmentation 3")
                    visualization.plot_img(lateral_map_1[0], window6, "Candidate Segmentation 4")
                        
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record5.update(loss5.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record1.update(loss1.data, opt.batchsize)

                    # Loss is only recorded when the rate is 1; update the plot at such event

                    pass_val += 1
                    passes.append(pass_val)

                    # Multiple forward passes in each training step - instead of steps by loss, we plot forward passes by loss

                    # Individual losses placed on the same plot
                    xlabel = "Num. of Forward Passes"
                    ylabel = "Loss"
                    legend = ['Candidate Seg. 1 Loss', 'Candidate Seg. 2 Loss', 'Candidate Seg. 3 Loss', 'Candidate Seg. 4 Loss']

                    window7 = visualization.plot_line(np.column_stack([[pass_val], [pass_val], [pass_val], [pass_val]]), np.column_stack([[loss_record5.show().item()], [loss_record3.show().item()], [loss_record2.show().item()], [loss_record1.show().item()]]), "Component Losses", window7, update, xlabel, ylabel, legend)
                    update = True
                
                # Obtains meanIoU and meanDice results on training data batch for this iteration
                meanIoUs = []
                meanDiceScores = []
                for j in range(len(lateral_maps)):
                    meanIoUs.append(numpy_mean_iou(gts, lateral_maps[j]))
                    meanDiceScores.append(numpy_mean_dice(gts, lateral_maps[j]))

                # ---- train visualization ----
                if i % 20 == 0 or i == total_step:
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    ' lateral-5: {:0.4f}], lateral-3: {:0.4f}], lateral-2: {:0.4f}], lateral-1: {:0.4f}]'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                            loss_record5.show(),loss_record3.show(),loss_record2.show(),loss_record1.show()))

                    print('lateral-5 meanIoU: {:0.4f}], lateral-3 meanIoU: {:0.4f}], lateral-2 meanIoU: {:0.4f}], lateral-1 meanIoU: {:0.4f}]'.
                            format(meanIoUs[-1],meanIoUs[-2],meanIoUs[-3],meanIoUs[-4]))

                    print('lateral-5 meanDice: {:0.4f}], lateral-3 meanDice: {:0.4f}], lateral-2 meanDice: {:0.4f}], lateral-1 meanDice: {:0.4f}]'.
                            format(meanDiceScores[-1],meanDiceScores[-2],meanDiceScores[-3],meanDiceScores[-4]))

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
        
    
    if (epoch+1) % 1 == 0:
        meandice = test(model,test_path)
        
        fp = open('log/log.txt','a')
        fp.write(str(meandice)+'\n')
        fp.close()
        
        fp = open('log/best.txt','r')
        best = fp.read()
        fp.close()
        
        if meandice > float(best):
            fp = open('log/best.txt','w')
            fp.write(str(meandice))
            fp.close()
            # best = meandice
            fp = open('log/best.txt','r')
            best = fp.read()
            fp.close()
            torch.save(model.state_dict(), save_path + 'CaraNet-best.pth' )
            print('[Saving Snapshot:]', save_path + 'CaraNet-best.pth',meandice,'[best:]',best)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int,
                        default=10, help='epoch number')
    
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    
    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    
    parser.add_argument('--batchsize', type=int,
                        default=6, help='training batch size')
    
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    
    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset', help='path to train dataset')
    
    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/CVC-300' , help='path to testing Kvasir dataset')
    
    parser.add_argument('--train_save', type=str,
                        default='CaraNet-best')
    
    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = caranet().cuda()
    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(model, x)

    params = model.parameters()
    
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay = 1e-4, momentum = 0.9)
        
    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation = opt.augmentation)

    #### Beginning of Echonet initialization code ####

    # Seed RNGs
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_train_patients = None
    batch_size = 6
    num_workers = 4

    # Set device for computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute mean and std
    mean, std = get_mean_and_std(Echo(root="/home/anshul/a4c-video-dir", split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              }

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = Echo(root="/home/anshul/a4c-video-dir", split="train", **kwargs)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = Echo(root="/home/anshul/a4c-video-dir", split="val", **kwargs)
    
    ds = dataset["train"]
    phase = "train"
    dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

    train_loader = dataloader

    val_loader = torch.utils.data.DataLoader(
        dataset["val"], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "val"))


    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, val_loader, model, optimizer, epoch, opt.test_path)

    # Run testing

   



