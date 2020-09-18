'''
This is the evaluation code modified from the originally released code.
1) Remove dependency on Polygon library
2) Fix input prediction format to x0,y0,x1,y1,...
'''

import file_util
import numpy as np
import math
import cv2

project_root = '../../'

pred_root = project_root + 'results/submit_msra/'
gt_root = project_root + 'data/MSRA-TD500/test/'


def get_pred(path):
    lines = file_util.read_file(path).split('\n')
    bboxes = []
    for line in lines:
        if line == '':
            continue
        bbox = line.split(',')
        if len(bbox) % 2 == 1:
            print(path)
        bbox = [int(x) for x in bbox]
        bboxes.append(bbox)
    return bboxes


def get_gt(path):
    lines = file_util.read_file(path).split('\n')
    bboxes = []
    tags = []
    for line in lines:
        if line == '':
            continue
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # gt = util.str.split(line, ' ')
        gt = line.split(' ')

        w_ = np.float(gt[4])
        h_ = np.float(gt[5])
        x1 = np.float(gt[2]) + w_ / 2.0
        y1 = np.float(gt[3]) + h_ / 2.0
        theta = np.float(gt[6]) / math.pi * 180

        bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
        bbox = bbox.reshape(-1)

        bboxes.append(bbox)
        tags.append(np.int(gt[1]))
    return np.array(bboxes), tags

def get_union(pD, pG, H, W):
    # replace original polygon library by opencv function
    blank = np.zeros((H, W))
    image1 = cv2.fillPoly(blank.copy(), [pD], 1)
    image2 = cv2.fillPoly(blank.copy(), [pG], 1)
    areaA = np.sum(image1)
    areaB = np.sum(image2)

    return areaA + areaB - get_intersection(pD, pG, H, W)

def get_intersection(pD, pG, H, W):
    # replace original polygon library by opencv function
    blank = np.zeros((H, W))
    image1 = cv2.fillPoly(blank.copy(), [pD], 1)
    image2 = cv2.fillPoly(blank.copy(), [pG], 1)
    intersection = np.logical_and(image1, image2)

    return np.sum(intersection)

if __name__ == '__main__':
    th = 0.5
    pred_list = file_util.read_dir(pred_root)

    count, tp, fp, tn, ta = 0, 0, 0, 0, 0
    for pred_path in pred_list:
        count = count + 1
        preds = get_pred(pred_path)
        gt_path = gt_root + pred_path.split('/')[-1].split('.')[0] + '.gt'
        img = cv2.imread(gt_root + pred_path.split('/')[-1][:-4] + '.jpg')
        H, W, _ = img.shape
        gts, tags = get_gt(gt_path)

        ta = ta + len(preds)
        for gt, tag in zip(gts, tags):
            gt = np.array(gt)
            gt = gt.reshape(gt.shape[0] // 2, 2)
            #gt_p = plg.Polygon(gt)
            difficult = tag
            flag = 0
            for pred in preds:
                pred = np.array(pred)
                pred = pred.reshape(pred.shape[0] // 2, 2)
                #pred_p = plg.Polygon(pred)

                union = get_union(pred, gt, H, W)
                inter = get_intersection(pred, gt, H, W)
                iou = float(inter) / union
                if iou >= th:
                    flag = 1
                    tp = tp + 1
                    break

            if flag == 0 and difficult == 0:
                fp = fp + 1

    recall = float(tp) / (tp + fp)
    precision = float(tp) / ta
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    print('p: %.4f, r: %.4f, f: %.4f' % (precision, recall, hmean))
