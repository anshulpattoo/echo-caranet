# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:41:30 2021

@author: angelou
"""

import torch
from torch.autograd import Variable
import os
import argparse
import itertools
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
# from miseval import evaluate

# Evaluation metrics
def numpy_iou(y_true, y_pred, n_class=1):
    IOU = []
    for c in range(1, n_class + 1):
        for i in range(y_true.shape[0]): # Iterate through each ground truth segmentation mask in the batch
            TP = np.sum((y_true[i] == c) & (y_pred[i] == c))
            FP = np.sum((y_true[i] != c) & (y_pred[i] == c))
            FN = np.sum((y_true[i] == c) & (y_pred[i] != c))

            n = TP
            d = float(TP + FP + FN + 1e-12)

            iou = np.divide(n, d)
            IOU.append(iou)
    return np.mean(IOU)

def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
        
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def numpy_mean_iou(y_true, y_pred):
    # Check if arrays are Torch; if so, convert to NumPy
    if (type(y_pred).__module__ == torch.__name__):
        y_pred = y_pred.cpu().detach().numpy()
    if (type(y_true).__module__ == torch.__name__):
        y_true = y_true.cpu().detach().numpy()

    prec = []
    for t in np.arange(0.5, 1.0, 0.5): # 0.5 is threshold
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        # score = tf.numpy_function(numpy_iou, [y_true, y_pred_], tf.float64)
        score = numpy_iou(y_true, y_pred_)
        prec.append(score)
    
    return K.mean(K.stack(prec), axis=0)

def numpy_mean_dice(y_true, y_pred):
    # Check if arrays are Torch; if so, convert to NumPy
    if (type(y_pred).__module__ == torch.__name__):
        y_pred = y_pred.cpu().detach().numpy()
    if (type(y_true).__module__ == torch.__name__):
        y_true = y_true.cpu().detach().numpy()

    prec = []
    for t in np.arange(0.5, 1.0, 0.5):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score = tf.numpy_function(dice, [y_true, y_pred_], tf.float64)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

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

# The computation that takes place for any given train step (happens twice; one for systolic frame and other for diastolic frame)
def train_step(lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1,
loss_record1, loss_record2, loss_record3, loss_record5, visualization, gts, window, window2, window3, window4, window5, window6, rate, norm_image, norm_gt):
    # ---- loss function ----
    loss5 = structure_loss(lateral_map_5, gts)
    loss3 = structure_loss(lateral_map_3, gts)
    loss2 = structure_loss(lateral_map_2, gts)
    loss1 = structure_loss(lateral_map_1, gts)
                
    loss = loss5 + loss3 + loss2 + loss1

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

    return [loss_record5, loss_record3, loss_record2, loss_record1, loss] # *Starts at 5 and goes to 1*

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

    window7 = None
    window8 = None
    window9 = None
    window10 = None

    X = deque([])
    Y = deque([])
    step = 0

    frame_list = ["Diastolic", "Systolic"] # ind. 0 is diastolic; ind. 1 is systolic
    loss_updates = []
    update_val = 0
    updatew7 = False
    updatew8 = False
    updatew9 = False
    updatew10 = False

    # meanIoU and meanDice scores over the whole validation set - forming a broader
    # mean IoU and mean Dice score once training is complete
    meanIoUsVal = []
    meanDiceScoresVal = []
    
    for i, pack in enumerate(itertools.zip_longest(train_loader, val_loader), start=1):
        # Same image, but different sizes, resulting in slightly different segmentations
        for j, rate in enumerate(size_rates, start=1):
            for k, frame in enumerate(frame_list):
                optimizer.zero_grad()
                # ---- data prepare ----

                if i < len(val_loader):
                    split = 2
                    # train_images, train_gts = pack[0]
                    # val_images, val_gts = pack[1]
                else:
                    # train_images, train_gts = pack[0]
                    split = 1
                
                """
                Both training and validation images undergo the same pre-processing prior to inference
                Training data make up the first element; validation data make up the second element
                
                `images` represents the full set of frames in the video
                `gts` has relevant frames for tracing, along with segmentations
                """
                
                # Iterate through the train-val split; first iteration
                # is training data and second iteration is validation
                for l in range(split):
                    # Shared/overlapping implementation
                    images, gts, = pack[l]

                    images = Variable(gts[k]).cuda()  
                    
                    img_np_arr = gts[k][0].cpu().detach().numpy()

                    gt = gts[k + 2][0].cpu().detach().numpy()
                    norm_image = cv2.normalize(img_np_arr, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                    norm_image = norm_image.astype(np.uint8)
                    
                    norm_gt = cv2.normalize(gt, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                    norm_gt = norm_gt.astype(np.uint8)

                    gts = Variable(gts[k + 2]).cuda()
                    
                    try:
                        gts = gts.reshape([6, 1, 112, 112])
                    except:
                        import pdb;pdb.set_trace()
                    # ---- rescale ----
                    trainsize = int(round(opt.trainsize*rate/32)*32)
                    # if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    
                    # ---- forward ----
                    lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1 = model(images)

                    # Convert candidate segmentations to correct grayscale images
                    
                    lateral_maps = [lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_5]
                        
                    for m in range(len(lateral_maps)):
                        individual_maps = []
                        for n in range(len(lateral_maps[m])):
                            if (type(lateral_maps[m]).__module__ == torch.__name__):
                                individual_map = lateral_maps[m][n].cpu().detach().numpy()
                            ret, individual_map = cv2.threshold(individual_map, 1, 255, cv2.THRESH_BINARY)
                            individual_maps.append(individual_map)
                        lateral_maps[m] = np.array(individual_maps)

                    """
                    If we are dealing with the training images, 
                    perform loss update
                    """
                    if l == 0:
                        """
                        Because of the three different rates and two different frames, there are a total of five passes (both forward and backward) through the network for any given training image or step
                        """
                        return_arr = train_step(lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1, loss_record1, loss_record2, loss_record3, loss_record5, visualization, gts, window, window2, window3, window4, window5, window6, rate, norm_image, norm_gt)
                        
                        # Display images on Visdom
                        visualization.plot_img(norm_image, window, f'{frame} Frame')
                        visualization.plot_img(norm_gt, window2, "Ground Truth Segmentation")     
                        visualization.plot_img(lateral_maps[3][0], window3, "Candidate Segmentation 1")
                        visualization.plot_img(lateral_maps[2][0], window4, "Candidate Segmentation 2")
                        visualization.plot_img(lateral_maps[1][0], window5, "Candidate Segmentation 3")
                        visualization.plot_img(lateral_maps[0][0], window6, "Candidate Segmentation 4")

                        # Plot the updated loss (loss is updated twice in any given iteration); loss gets updated when rate == 1   
                        if rate == 1:
                            update_val += 1
                            loss_updates.append(update_val)

                            xlabel = "Num. of Loss Updates"
                            ylabel = "Loss"
                            legend = ['Candidate Seg. 1 Loss', 'Candidate Seg. 2 Loss', 'Candidate Seg. 3 Loss', 'Candidate Seg. 4 Loss']

                            window7 = visualization.plot_line(np.column_stack([[update_val], [update_val], [update_val], [update_val]]), np.column_stack([[return_arr[0].show().item()], [return_arr[1].show().item()], [return_arr[2].show().item()], [return_arr[3].show().item()]]), "Component Losses (as parts of sum of total loss)", window7, updatew7, xlabel, ylabel, legend)
                            updatew7 = True
                            
                            legend = ['Loss']

                            window10 = visualization.plot_line(np.column_stack([[update_val]]), np.column_stack([return_arr[4].item()]), "Total Loss", window10, updatew10, xlabel, ylabel, legend)
                            updatew10 = True 
                    
                    # If we are dealing with the validation images, derive meanIoU and meanDice for such data
                    elif l == 1:
                        # meanIoU and meanDice scores are captured for each batch of the validation set
                        meanIoUs = []
                        meanDiceScores = []                        

                        for m in range(len(lateral_maps)):
                            meanIoUs.append(numpy_mean_iou(gts, lateral_maps[m]))
                            meanDiceScores.append(numpy_mean_dice(gts, lateral_maps[m]))
                            
                            # y_pred = lateral_maps[m]
                            # y_true = gts

                            # if (type(y_pred).__module__ == torch.__name__):
                            #     y_pred = y_pred.cpu().detach().numpy()
                            # if (type(y_true).__module__ == torch.__name__):
                            #     y_true = y_true.cpu().detach().numpy()

                            # meanDiceScoresOther.append(evaluate(y_pred, y_true, metric="DSC")  )

                        meanIoUsVal.append(meanIoUs[-1].numpy())
                        meanDiceScoresVal.append(meanDiceScores[-1].numpy())

                        """
                        For the sake of testing meanIoU and meanDice functions
                        Generates fake segementation masks: i.e., no white space 
                        """
                        # fake_seg_masks = np.full(gts.shape, 1, dtype=float) 
                        
                        # for m in range(len(lateral_maps)):
                        #     meanIoUs.append(numpy_mean_iou(fake_seg_masks, fake_seg_masks))
                        #     meanDiceScores.append(numpy_mean_dice(fake_seg_masks, fake_seg_masks))
                        
        # ---- Visualization during training (at the very end of any given training step - i.e., rate is 1.25 and k is 1) ----  
            
        if (i % 20 == 0 or i == total_step) and rate == 1.25 and k == 1:
            # Print metrics to console
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
            ' lateral-5: {:0.4f}], lateral-3: {:0.4f}], lateral-2: {:0.4f}], lateral-1: {:0.4f}]'.
            format(datetime.now(), epoch, opt.epoch, i, total_step, return_arr[0].show(),return_arr[1].show(),return_arr[2].show(),return_arr[3].show()))

            if i <= len(val_loader):
                print('lateral-5 valMeanIoU: {:0.4f}], lateral-3 valMeanIoU: {:0.4f}], lateral-2 valMeanIoU: {:0.4f}], lateral-1 valMeanIoU: {:0.4f}]'.
                format(meanIoUs[-1],meanIoUs[-2],meanIoUs[-3],meanIoUs[-4]))

                print('lateral-5 valMeanDice: {:0.4f}], lateral-3 valMeanDice: {:0.4f}], lateral-2 valMeanDice: {:0.4f}], lateral-1 valMeanDice: {:0.4f}]'.
                format(meanDiceScores[-1],meanDiceScores[-2],meanDiceScores[-3],meanDiceScores[-4]))

                # Further plotted metrics (aside from loss)

                # Plot val. mean iou
                xlabel = "Num. of Training Steps"
                ylabel = "Val. Mean IoU for Training Batch"
                legend = ['Candidate Seg. 1', 'Candidate Seg. 2', 'Candidate Seg. 3', 'Candidate Seg. 4']
                window8 = visualization.plot_line(np.column_stack([[i], [i], [i], [i]]), np.column_stack([[meanIoUs[-1]], [meanIoUs[-2]], [meanIoUs[-3]], [meanIoUs[-4]]]), "Mean IoU on Validation Set for each Batch", window8, updatew8, xlabel, ylabel, legend)
                updatew8 = True

                # Plot val. mean dice
                xlabel = "Num. of Training Steps"
                ylabel = "Val. Mean Dice Score for Training Batch"
                legend = ['Candidate Seg. 1', 'Candidate Seg. 2', 'Candidate Seg. 3', 'Candidate Seg. 4']
                window9 = visualization.plot_line(np.column_stack([[i], [i], [i], [i]]), np.column_stack([[meanDiceScores[-1]], [meanDiceScores[-2]], [meanDiceScores[-3]], [meanDiceScores[-4]]]), "Mean Dice Score on Validation Set for each Batch", window9, updatew9, xlabel, ylabel, legend)
                updatew9 = True


                if i % 60 == 0:
                    import pdb;pdb.set_trace()


    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    
    # if (epoch+1) % 1 == 0:
    #     meandice = test(model,test_path)
        
    #     fp = open('log/log.txt','a')
    #     fp.write(str(meandice)+'\n')
    #     fp.close()
        
    #     fp = open('log/best.txt','r')
    #     best = fp.read()
    #     fp.close()
        
    #     if meandice > float(best):
    #         fp = open('log/best.txt','w')
    #         fp.write(str(meandice))
    #         fp.close()
    #         # best = meandice
    #         fp = open('log/best.txt','r')
    #         best = fp.read()
    #         fp.close()
    #         torch.save(model.state_dict(), save_path + 'CaraNet-best.pth' )
    #         print('[Saving Snapshot:]', save_path + 'CaraNet-best.pth',meandice,'[best:]',best)

    return meanIoUsVal, meanDiceScoresVal
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int,
                        default=3, help='epoch number')
    
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
    alis = []
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        meanIoUsVal, meanDiceScoresVal = train(train_loader, val_loader, model, optimizer, epoch, opt.test_path)
        alis.extend(meanIoUsVal)
        alis.extend(meanDiceScoresVal)

    import pdb;pdb.set_trace()


   



