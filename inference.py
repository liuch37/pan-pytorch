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
from dataset import testdataset
from models.pan import PAN
from utils.helper import get_results, write_result, draw_result, upsample

# main function:
if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--worker', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--input', type=str, default='', required=True, help='input folder name')
    parser.add_argument('--output', type=str, default='results', help='output folder name')
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--gpu', type=bool, default=False, help="GPU being used or not")
    parser.add_argument('--bbox_type', type=str, default='poly', help="bounding box type - poly | rect")

    opt = parser.parse_args()
    print(opt)

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
    batch_size = 1
    neck_channel = (64, 128, 256, 512)
    pa_in_channels = 512
    hidden_dim = 128
    num_classes = 6

    data_dirs = opt.input
    worker = opt.worker
    output_path = opt.output
    trained_model_path = opt.model
    bbox_type = opt.bbox_type
    min_area = 16
    min_score = 0.88

    # create dataset
    print("Create dataset......")
    test_dataset = testdataset.PAN_test(data_dirs, 640)

    # make dataloader    
    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=int(worker))

    print("Length of test dataset is:", len(test_dataset))

    # make model prediction output folder
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
        print("Error: Empty model path!")
        exit(1)

    # model inference
    print("Prediction on testset......")
    timer = []
    model.eval()
    for idx, data in enumerate(test_dataloader):
        print('Testing %d/%d' % (idx, len(test_dataloader)))
        outputs = dict()
        # prepare input
        data['imgs'] = data['imgs'].to(device)
        # forward
        start = time.time()
        with torch.no_grad():
            det_out = model(data['imgs'])
            det_out = upsample(det_out, data['imgs'].size(), 4)
            det_res = get_results(det_out, data['img_metas'], min_area, min_score, bbox_type)
            outputs.update(det_res)
        end = time.time()
        timer.append(end - start)

        # save result
        image_name, _ = os.path.splitext(os.path.basename(test_dataloader.dataset.img_paths[idx]))
        write_result(image_name, outputs, os.path.join(output_path, 'submit_ctw'))

        # draw and save images
        draw_result(test_dataloader.dataset.img_paths[idx], outputs, output_path)

    print("Average FPS:", 1/(sum(timer)/len(timer)))