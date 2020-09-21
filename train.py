'''
THis is the main training code.
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU id at the very begining
import argparse
import random
import math
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.multiprocessing import freeze_support
import json
import sys
import time
import pdb
# internal package
from dataset import ctw1500, totaltext, synthtext, msra, ic15
from models.pan import PAN
from loss.loss import loss
from utils.helper import adjust_learning_rate, upsample
from utils.average_meter import AverageMeter

# main function:
if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch', type=int, default=16, help='input batch size')
    parser.add_argument(
        '--worker', type=int, default=4, help='number of data loading workers')
    parser.add_argument(
        '--epoch', type=int, default=601, help='number of epochs')
    parser.add_argument('--output', type=str, default='outputs', help='output folder name')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset_type', type=str, default='ctw', help="dataset type - ctw | tt | synthtext | msra | ic15")
    parser.add_argument('--gpu', type=bool, default=False, help="GPU being used or not")

    opt = parser.parse_args()
    print(opt)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed:", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    
    # turn on GPU for models:
    if opt.gpu == False:
        device = torch.device("cpu")
        print("CPU being used!")
    else:
        if torch.cuda.is_available() == True and opt.gpu == True:
            device = torch.device("cuda")
            print("GPU being used!")
        else:
            device = torch.device("cpu")
            print("CPU being used!")

    # set training parameters
    batch_size = opt.batch
    neck_channel = (64, 128, 256, 512)
    pa_in_channels = 512
    hidden_dim = 128
    num_classes = 6
    loss_text_weight = 1.0
    loss_kernel_weight = 0.5
    loss_emb_weight = 0.25
    opt.optimizer = 'Adam'
    opt.lr = 1e-3
    opt.schedule = 'polylr'

    epochs = opt.epoch
    worker = opt.worker
    dataset_type = opt.dataset_type
    output_path = opt.output
    trained_model_path = opt.model

    # create dataset
    print("Create dataset......")
    if dataset_type == 'ctw': # ctw dataset
        train_dataset = ctw1500.PAN_CTW(split='train',
                                        is_transform=True,
                                        img_size=640,
                                        short_size=640,
                                        kernel_scale=0.7,
                                        report_speed=False)
    elif dataset_type == 'tt': # totaltext dataset
        train_dataset = totaltext.PAN_TT(split='train',
                                         is_transform=True,
                                         img_size=640,
                                         short_size=640,
                                         kernel_scale=0.7,
                                         with_rec=False,
                                         report_speed=False)
    elif dataset_type == 'synthtext': # synthtext dataset
        train_dataset = synthtext.PAN_Synth(is_transform=True,
                                            img_size=640,
                                            short_size=640,
                                            kernel_scale=0.5,
                                            with_rec=False)
    elif dataset_type == 'msra': # msra dataset
        train_dataset = msra.PAN_MSRA(split='train',
                                      is_transform=True,
                                      img_size=736,
                                      short_size=736,
                                      kernel_scale=0.7,
                                      report_speed=False)
    elif dataset_type == 'ic15': # msra dataset
        train_dataset = ic15.PAN_IC15(split='test',
                                      is_transform=True,
                                      img_size=736,
                                      short_size=736,
                                      kernel_scale=0.5,
                                      with_rec=False)
    else:
        print("Not supported yet!")
        exit(1)

    # make dataloader
    train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=int(worker),
                    drop_last=True,
                    pin_memory=True)

    print("Length of train dataset is:", len(train_dataset))

    # make model output folder
    try:
        os.makedirs(output_path)
    except OSError:
        pass

    # create model
    print("Create model......")
    model = PAN(pretrained=False, neck_channel=neck_channel, pa_in_channels=pa_in_channels, hidden_dim=hidden_dim, num_classes=num_classes)

    if trained_model_path != '':
        if torch.cuda.is_available() == True and opt.gpu == True:
            model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
            model = torch.nn.DataParallel(model).to(device)
        else:
            model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
    else:
        if torch.cuda.is_available() == True and opt.gpu == True:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.99, weight_decay=5e-4)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    else:
        print("Error: Please specify correct optimizer!")
        exit(1)

    # train, evaluate, and save model
    print("Training starts......")

    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        print('Epoch: [%d | %d]' % (epoch + 1, epochs))
        model.train()

        # meters
        losses = AverageMeter()
        losses_text = AverageMeter()
        losses_kernels = AverageMeter()
        losses_emb = AverageMeter()
        losses_rec = AverageMeter()
        ious_text = AverageMeter()
        ious_kernel = AverageMeter()

        for iter, data in enumerate(train_dataloader):
            
            # adjust learning rate
            adjust_learning_rate(optimizer, train_dataloader, epoch, iter, opt.schedule, opt.lr, epochs)
            
            outputs = dict()
            # forward for detection output
            det_out = model(data['imgs'].to(device))
            det_out = upsample(det_out, data['imgs'].size())
            # retreive ground truth labels
            gt_texts = data['gt_texts'].to(device)
            gt_kernels = data['gt_kernels'].to(device)
            training_masks = data['training_masks'].to(device)
            gt_instances = data['gt_instances'].to(device)
            gt_bboxes = data['gt_bboxes'].to(device)
            # calculate total loss
            det_loss = loss(det_out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes, loss_text_weight, loss_kernel_weight, loss_emb_weight)
            outputs.update(det_loss)
            
            # detection loss
            loss_text = torch.mean(outputs['loss_text'])
            losses_text.update(loss_text.item())

            loss_kernels = torch.mean(outputs['loss_kernels'])
            losses_kernels.update(loss_kernels.item())

            loss_emb = torch.mean(outputs['loss_emb'])
            losses_emb.update(loss_emb.item())

            loss_total = loss_text + loss_kernels + loss_emb

            iou_text = torch.mean(outputs['iou_text'])
            ious_text.update(iou_text.item())
            iou_kernel = torch.mean(outputs['iou_kernel'])
            ious_kernel.update(iou_kernel.item())

            losses.update(loss_total.item())

            # backward
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # print log
            #print("batch: {} / total batch: {}".format(iter+1, len(train_dataloader)))
            if iter % 20 == 0:
                output_log = '({batch}/{size}) LR: {lr:.6f} | ' \
                             'Loss: {loss:.3f} | ' \
                             'Loss (text/kernel/emb): {loss_text:.3f}/{loss_kernel:.3f}/{loss_emb:.3f} ' \
                             '| IoU (text/kernel): {iou_text:.3f}/{iou_kernel:.3f}'.format(
                    batch=iter + 1,
                    size=len(train_dataloader),
                    lr=optimizer.param_groups[0]['lr'],
                    loss_text=losses_text.avg,
                    loss_kernel=losses_kernels.avg,
                    loss_emb=losses_emb.avg,
                    loss=losses.avg,
                    iou_text=ious_text.avg,
                    iou_kernel=ious_kernel.avg,            
                )
                print(output_log)
                sys.stdout.flush()
                with open(os.path.join(output_path,'statistics.txt'), 'a') as f:
                    f.write("{} {} {} {} {} {}\n".format(losses_text.avg, losses_kernels.avg, losses_emb.avg, losses.avg, ious_text.avg, ious_kernel.avg))

        if epoch % 20 == 0:
            print("Save model......")
            if torch.cuda.is_available() == True and opt.gpu == True:
                torch.save(model.module.state_dict(), '%s/model_epoch_%s.pth' % (output_path, str(epoch)))
            else:
                torch.save(model.state_dict(), '%s/model_epoch_%s.pth' % (output_path, str(epoch)))