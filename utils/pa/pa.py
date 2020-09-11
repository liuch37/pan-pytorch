'''
This is an unofficial implementation of pixel aggregation function in pure python modified from pa.pyx
'''

import numpy as np
import cv2
import pdb

def _pa(kernels, emb, label, cc, kernel_num, label_num, min_area=0):
    pred = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)
    mean_emb = np.zeros((label_num, 4), dtype=np.float32)
    area = np.full((label_num,), -1, dtype=np.float32)
    flag = np.zeros((label_num,), dtype=np.int32)
    inds = np.zeros((label_num, label.shape[0], label.shape[1]), dtype=np.uint8)
    p = np.zeros((label_num, 2), dtype=np.int32)

    max_rate = 1024
    for i in range(1, label_num):
        ind = label == i
        inds[i] = ind

        area[i] = np.sum(ind)

        if area[i] < min_area:
            label[ind] = 0
            continue

        px, py = np.where(ind)
        p[i] = (px[0], py[0])

        for j in range(1, i):
            if area[j] < min_area:
                continue
            if cc[p[i, 0], p[i, 1]] != cc[p[j, 0], p[j, 1]]:
                continue
            rate = area[i] / area[j]
            if rate < 1 / max_rate or rate > max_rate:
                flag[i] = 1
                mean_emb[i] = np.mean(emb[:, ind], axis=1)

                if flag[j] == 0:
                    flag[j] = 1
                    mean_emb[j] = np.mean(emb[:, inds[j].astype(np.bool)], axis=1)

    que = []
    nxt_que = []
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    points = np.array(np.where(label > 0)).transpose((1, 0))
    for point_idx in range(points.shape[0]):
        tmpx, tmpy = points[point_idx, 0], points[point_idx, 1]
        que.append((tmpx, tmpy))
        pred[tmpx, tmpy] = label[tmpx, tmpy]

    for kernel_idx in range(kernel_num - 2, -1, -1):
        while que:
            cur = que[0]
            que.pop(0)
            cur_label = pred[cur[0], cur[1]]

            is_edge = True
            for j in range(4):
                tmpx = cur[0] + dx[j]
                tmpy = cur[1] + dy[j]
                if tmpx < 0 or tmpx >= label.shape[0] or tmpy < 0 or tmpy >= label.shape[1]:
                    continue
                if kernels[kernel_idx, tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue
                if flag[cur_label] == 1 and np.linalg.norm(emb[:, tmpx, tmpy] - mean_emb[cur_label]) > 3:
                    continue

                que.append((tmpx, tmpy))
                pred[tmpx, tmpy] = cur_label
                is_edge = False
            if is_edge:
                nxt_que.append(cur)

        que, nxt_que = nxt_que, que

    return pred

def pa(kernels, emb, min_area=0):
    kernel_num = kernels.shape[0]
    _, cc = cv2.connectedComponents(kernels[0], connectivity=4) # text region connected components
    label_num, label = cv2.connectedComponents(kernels[1], connectivity=4) # kernel region connected components
    return _pa(kernels[:-1], emb, label, cc, kernel_num, label_num, min_area)