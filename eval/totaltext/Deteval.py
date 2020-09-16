'''
This is the evaluation code modified from the originally released code.
1) Remove dependency on Polygon library
2) Fix input prediction format to x0,y0,x1,y1,...
'''

from os import listdir
from scipy import io
import numpy as np
import cv2

"""
Input format: x0,y0, ..... xn,yn. Each detection is separated by the end of line token ('\n')'
"""
project_root = '../../'

input_dir = project_root + 'outputs/submit_tt/'
gt_dir = project_root + 'data/totaltext/Groundtruth/Polygon/Test/'
fid_path = project_root + 'outputs/res_tt.txt'
img_root = project_root + 'data/totaltext/Images/Test/'

allInputs = listdir(input_dir)

'''
def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG)


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()
'''

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

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

def input_reading_mod(input_dir, input):
    """This helper reads input from txt files"""
    with open('%s/%s' % (input_dir, input), 'r') as input_fid:
        pred = input_fid.readlines()
    det = [x.strip('\n') for x in pred]
    return det


def gt_reading_mod(gt_dir, gt_id):
    """This helper reads groundtruths from mat files"""
    gt_id = gt_id.split('.')[0]
    gt = io.loadmat('%s/poly_gt_%s.mat' % (gt_dir, gt_id))
    gt = gt['polygt']
    return gt


def detection_filtering(detections, groundtruths, threshold=0.5, H, W):
    for gt_id, gt in enumerate(groundtruths):
        if (gt[5] == '#') and (gt[1].shape[1] > 1):
            gt_x = map(int, np.squeeze(gt[1]))
            gt_y = map(int, np.squeeze(gt[3]))

            gt_p = np.concatenate((np.array(gt_x), np.array(gt_y)))
            gt_p = gt_p.reshape(2, -1).transpose()
            #gt_p = plg.Polygon(gt_p) # replaced

            for det_id, detection in enumerate(detections):
                detection = detection.split(',')
                # detection = map(int, detection[0:-1])
                detection = map(int, detection)
                det_x = detection[0::2]
                det_y = detection[1::2]

                det_p = np.concatenate((np.array(det_x), np.array(det_y)))
                det_p = det_p.reshape(2, -1).transpose()
                #det_p = plg.Polygon(det_p) replaced

                try:
                    # det_gt_iou = iod(det_x, det_y, gt_x, gt_y)
                    #det_gt_iou = get_intersection(det_p, gt_p, H, W) / det_p.area() # replaced
                    det_gt_iou = get_intersection(det_p, gt_p, H, W) / PolyArea(det_p[:,0], det_p[:,1])
                except:
                    print(det_x, det_y, gt_x, gt_y)
                if det_gt_iou > threshold:
                    detections[det_id] = []

            detections[:] = [item for item in detections if item != []]
    return detections

def sigma_calculation(det_p, gt_p, H, W):
    """
    sigma = inter_area / gt_area
    """
    # return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) / area(gt_x, gt_y)), 2)
    #return get_intersection(det_p, gt_p) / gt_p.area() # replaced
    return get_intersection(det_p, gt_p, H, W) / PolyArea(gt_p[:,0], gt_p[:,1])

def tau_calculation(det_p, gt_p, H, W):
    """
    tau = inter_area / det_area
    """
    #return get_intersection(det_p, gt_p) / det_p.area() # replaced
    return get_intersection(det_p, gt_p, H, W) / PolyArea(det_p[:,0], det_p[:,1])


##############################Initialization###################################
global_tp = 0
global_fp = 0
global_fn = 0
global_sigma = []
global_tau = []
tr = 0.7
tp = 0.6
fsc_k = 0.8
k = 2
###############################################################################


for input_id in allInputs:
    if (input_id != '.DS_Store'):
        img = cv2.imread(img_root + input_id[:-4] + '.jpg')
        H, W, _ = img.shape
        print('input_id', input_id)
        detections = input_reading_mod(input_dir, input_id)
        # from IPython import embed;
        groundtruths = gt_reading_mod(gt_dir, input_id)
        detections = detection_filtering(detections, groundtruths, 0.5, H, W)  # filters detections overlapping with DC area
        dc_id = np.where(groundtruths[:, 5] == '#')
        groundtruths = np.delete(groundtruths, (dc_id), (0))

        local_sigma_table = np.zeros((groundtruths.shape[0], len(detections)))
        local_tau_table = np.zeros((groundtruths.shape[0], len(detections)))

        for gt_id, gt in enumerate(groundtruths):
            if len(detections) > 0:
                for det_id, detection in enumerate(detections):
                    detection = detection.split(',')
                    # print (len(detection))

                    # detection = map(int, detection[:-1])
                    detection = map(int, detection)
                    # print (len(detection))

                    # from IPython import embed;embed()
                    # detection = list(detection)
                    gt_x = map(int, np.squeeze(gt[1]))
                    gt_y = map(int, np.squeeze(gt[3]))

                    gt_p = np.concatenate((np.array(gt_x), np.array(gt_y)))
                    gt_p = gt_p.reshape(2, -1).transpose()
                    #gt_p = plg.Polygon(gt_p) # replaced

                    det_y = detection[1::2]
                    det_x = detection[0::2]

                    det_p = np.concatenate((np.array(det_x), np.array(det_y)))
                    # print (det_p.shape)
                    det_p = det_p.reshape(2, -1).transpose()
                    #det_p = plg.Polygon(det_p) # replaced

                    # gt_x = list(map(int, np.squeeze(gt[1])))
                    # gt_y = list(map(int, np.squeeze(gt[3])))
                    # try:
                    #     local_sigma_table[gt_id, det_id] = sigma_calculation(det_x, det_y, gt_x, gt_y)
                    #     local_tau_table[gt_id, det_id] = tau_calculation(det_x, det_y, gt_x, gt_y)
                    # except:
                    #     embed()
                    local_sigma_table[gt_id, det_id] = sigma_calculation(det_p, gt_p, H, W)
                    local_tau_table[gt_id, det_id] = tau_calculation(det_p, gt_p, H, W)
        # if input_id == 'img1199.txt':
        #    embed()
        global_sigma.append(local_sigma_table)
        global_tau.append(local_tau_table)

global_accumulative_recall = 0
global_accumulative_precision = 0
total_num_gt = 0
total_num_det = 0


def one_to_one(local_sigma_table, local_tau_table, local_accumulative_recall,
               local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
               gt_flag, det_flag):
    for gt_id in xrange(num_gt):
        qualified_sigma_candidates = np.where(local_sigma_table[gt_id, :] > tr)
        num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]
        qualified_tau_candidates = np.where(local_tau_table[gt_id, :] > tp)
        num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]

        if (num_qualified_sigma_candidates == 1) and (num_qualified_tau_candidates == 1):
            global_accumulative_recall = global_accumulative_recall + 1.0
            global_accumulative_precision = global_accumulative_precision + 1.0
            local_accumulative_recall = local_accumulative_recall + 1.0
            local_accumulative_precision = local_accumulative_precision + 1.0

            gt_flag[0, gt_id] = 1
            matched_det_id = np.where(local_sigma_table[gt_id, :] > tr)
            det_flag[0, matched_det_id] = 1
    return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


def one_to_many(local_sigma_table, local_tau_table, local_accumulative_recall,
                local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
                gt_flag, det_flag):
    for gt_id in xrange(num_gt):
        # skip the following if the groundtruth was matched
        if gt_flag[0, gt_id] > 0:
            continue

        non_zero_in_sigma = np.where(local_sigma_table[gt_id, :] > 0)
        num_non_zero_in_sigma = non_zero_in_sigma[0].shape[0]

        if num_non_zero_in_sigma >= k:
            ####search for all detections that overlaps with this groundtruth
            qualified_tau_candidates = np.where((local_tau_table[gt_id, :] >= tp) & (det_flag[0, :] == 0))
            num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]

            if num_qualified_tau_candidates == 1:
                if ((local_tau_table[gt_id, qualified_tau_candidates] >= tp) and (
                        local_sigma_table[gt_id, qualified_tau_candidates] >= tr)):
                    # became an one-to-one case
                    global_accumulative_recall = global_accumulative_recall + 1.0
                    global_accumulative_precision = global_accumulative_precision + 1.0
                    local_accumulative_recall = local_accumulative_recall + 1.0
                    local_accumulative_precision = local_accumulative_precision + 1.0

                    gt_flag[0, gt_id] = 1
                    det_flag[0, qualified_tau_candidates] = 1
            elif (np.sum(local_sigma_table[gt_id, qualified_tau_candidates]) >= tr):
                gt_flag[0, gt_id] = 1
                det_flag[0, qualified_tau_candidates] = 1

                global_accumulative_recall = global_accumulative_recall + fsc_k
                global_accumulative_precision = global_accumulative_precision + num_qualified_tau_candidates * fsc_k

                local_accumulative_recall = local_accumulative_recall + fsc_k
                local_accumulative_precision = local_accumulative_precision + num_qualified_tau_candidates * fsc_k

    return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


def many_to_many(local_sigma_table, local_tau_table, local_accumulative_recall,
                 local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
                 gt_flag, det_flag):
    for det_id in xrange(num_det):
        # skip the following if the detection was matched
        if det_flag[0, det_id] > 0:
            continue

        non_zero_in_tau = np.where(local_tau_table[:, det_id] > 0)
        num_non_zero_in_tau = non_zero_in_tau[0].shape[0]

        if num_non_zero_in_tau >= k:
            ####search for all detections that overlaps with this groundtruth
            qualified_sigma_candidates = np.where((local_sigma_table[:, det_id] >= tp) & (gt_flag[0, :] == 0))
            num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]

            if num_qualified_sigma_candidates == 1:
                if ((local_tau_table[qualified_sigma_candidates, det_id] >= tp) and (
                        local_sigma_table[qualified_sigma_candidates, det_id] >= tr)):
                    # became an one-to-one case
                    global_accumulative_recall = global_accumulative_recall + 1.0
                    global_accumulative_precision = global_accumulative_precision + 1.0
                    local_accumulative_recall = local_accumulative_recall + 1.0
                    local_accumulative_precision = local_accumulative_precision + 1.0

                    gt_flag[0, qualified_sigma_candidates] = 1
                    det_flag[0, det_id] = 1
            elif (np.sum(local_tau_table[qualified_sigma_candidates, det_id]) >= tp):
                det_flag[0, det_id] = 1
                gt_flag[0, qualified_sigma_candidates] = 1

                global_accumulative_recall = global_accumulative_recall + num_qualified_sigma_candidates * fsc_k
                global_accumulative_precision = global_accumulative_precision + fsc_k

                local_accumulative_recall = local_accumulative_recall + num_qualified_sigma_candidates * fsc_k
                local_accumulative_precision = local_accumulative_precision + fsc_k
    return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


for idx in xrange(len(global_sigma)):
    print(allInputs[idx])
    local_sigma_table = global_sigma[idx]
    local_tau_table = global_tau[idx]

    num_gt = local_sigma_table.shape[0]
    num_det = local_sigma_table.shape[1]

    total_num_gt = total_num_gt + num_gt
    total_num_det = total_num_det + num_det

    local_accumulative_recall = 0
    local_accumulative_precision = 0
    gt_flag = np.zeros((1, num_gt))
    det_flag = np.zeros((1, num_det))

    #######first check for one-to-one case##########
    local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
    gt_flag, det_flag = one_to_one(local_sigma_table, local_tau_table,
                                   local_accumulative_recall, local_accumulative_precision,
                                   global_accumulative_recall, global_accumulative_precision,
                                   gt_flag, det_flag)

    #######then check for one-to-many case##########
    local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
    gt_flag, det_flag = one_to_many(local_sigma_table, local_tau_table,
                                    local_accumulative_recall, local_accumulative_precision,
                                    global_accumulative_recall, global_accumulative_precision,
                                    gt_flag, det_flag)

    #######then check for many-to-many case##########
    local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
    gt_flag, det_flag = many_to_many(local_sigma_table, local_tau_table,
                                     local_accumulative_recall, local_accumulative_precision,
                                     global_accumulative_recall, global_accumulative_precision,
                                     gt_flag, det_flag)
    # for det_id in xrange(num_det):
    #     # skip the following if the detection was matched
    #     if det_flag[0, det_id] > 0:
    #         continue
    #
    #     non_zero_in_tau = np.where(local_tau_table[:, det_id] > 0)
    #     num_non_zero_in_tau = non_zero_in_tau[0].shape[0]
    #
    #     if num_non_zero_in_tau >= k:
    #         ####search for all detections that overlaps with this groundtruth
    #         qualified_sigma_candidates = np.where((local_sigma_table[:, det_id] >= tp) & (gt_flag[0, :] == 0))
    #         num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]
    #
    #         if num_qualified_sigma_candidates == 1:
    #             if ((local_tau_table[qualified_sigma_candidates, det_id] >= tp) and (local_sigma_table[qualified_sigma_candidates, det_id] >= tr)):
    #                 #became an one-to-one case
    #                 global_accumulative_recall = global_accumulative_recall + 1.0
    #                 global_accumulative_precision = global_accumulative_precision + 1.0
    #                 local_accumulative_recall = local_accumulative_recall + 1.0
    #                 local_accumulative_precision = local_accumulative_precision + 1.0
    #
    #                 gt_flag[0, qualified_sigma_candidates] = 1
    #                 det_flag[0, det_id] = 1
    #         elif (np.sum(local_tau_table[qualified_sigma_candidates, det_id]) >= tp):
    #             det_flag[0, det_id] = 1
    #             gt_flag[0, qualified_sigma_candidates] = 1
    #
    #             global_accumulative_recall = global_accumulative_recall + num_qualified_sigma_candidates * fsc_k
    #             global_accumulative_precision = global_accumulative_precision + fsc_k
    #
    #             local_accumulative_recall = local_accumulative_recall + num_qualified_sigma_candidates * fsc_k
    #             local_accumulative_precision = local_accumulative_precision + fsc_k

    fid = open(fid_path, 'a+')
    try:
        local_precision = local_accumulative_precision / num_det
    except ZeroDivisionError:
        local_precision = 0

    try:
        local_recall = local_accumulative_recall / num_gt
    except ZeroDivisionError:
        local_recall = 0

    temp = ('%s______/Precision:_%s_______/Recall:_%s\n' % (allInputs[idx], str(local_precision), str(local_recall)))
    fid.write(temp)
    fid.close()
try:
    recall = global_accumulative_recall / total_num_gt
except ZeroDivisionError:
    recall = 0

try:
    precision = global_accumulative_precision / total_num_det
except ZeroDivisionError:
    precision = 0

try:
    f_score = 2 * precision * recall / (precision + recall)
except ZeroDivisionError:
    f_score = 0

fid = open(fid_path, 'a')
hmean = 2 * precision * recall / (precision + recall)
temp = ('Precision:_%s_______/Recall:_%s/Hmean:_%s\n' % (str(precision), str(recall), str(hmean)))
print(temp)
fid.write(temp)
fid.close()

print('pb')