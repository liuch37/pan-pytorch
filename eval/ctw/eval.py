'''
This is the evaluation code modified from the originally released code.
1) Remove dependency on Polygon library
2) Fix input prediction format to x0,y0,x1,y1,...
'''

import file_util
import numpy as np
import cv2

project_root = '../../'

pred_root = project_root + 'results/submit_ctw'
gt_root = project_root + 'data/CTW1500/test/text_label_circum/'
img_root = project_root + 'data/CTW1500/test/text_image/'

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
    for line in lines:
        if line == '':
            continue
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # gt = util.str.split(line, ',')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1, y1] * 14)

        bboxes.append(bbox)
    return bboxes


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

    tp, fp, npos = 0, 0, 0

    for pred_path in pred_list:
        print("evaluting predict path:", pred_path)
        preds = get_pred(pred_path)
        gt_path = gt_root + pred_path.split('/')[-1]
        img = cv2.imread(img_root + pred_path.split('/')[-1][:-4] + '.jpg')
        H, W, _ = img.shape
        gts = get_gt(gt_path)
        npos += len(gts)

        cover = set()
        for pred_id, pred in enumerate(preds):
            pred = np.array(pred)
            pred = pred.reshape(pred.shape[0] // 2, 2)

            #pred_p = plg.Polygon(pred)

            flag = False
            for gt_id, gt in enumerate(gts):
                gt = np.array(gt)
                gt = gt.reshape(gt.shape[0] // 2, 2)
                #gt_p = plg.Polygon(gt)

                union = get_union(pred, gt, H, W)
                inter = get_intersection(pred, gt, H, W)

                if inter * 1.0 / union >= th:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)
            if flag:
                tp += 1.0
            else:
                fp += 1.0

    # print tp, fp, npos
    precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    print('p: %.4f, r: %.4f, f: %.4f' % (precision, recall, hmean))
