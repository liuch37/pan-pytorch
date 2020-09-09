'''
This function is to compute total loss for training
'''
import torch
from .ohem import ohem_batch
from .iou import iou
from .dice_loss import DiceLoss
from .emb_loss_v1 import EmbLoss_v1

def text_loss(input, target, mask, reduce, loss_weight):
    loss = DiceLoss(loss_weight)
    return loss(input, target, mask, reduce)
    
def kernel_loss(input, target, mask, reduce, loss_weight):
    loss = DiceLoss(loss_weight)
    return loss(input, target, mask, reduce)

def emb_loss(emb, instance, kernel, training_mask, bboxes, reduce, loss_weight):
    loss = EmbLoss_v1(feature_dim=4, loss_weight=loss_weight)
    return loss(emb, instance, kernel, training_mask, bboxes, reduce)

def loss(out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes, loss_text_weight, loss_kernel_weight, loss_emb_weight):
        # output
        texts = out[:, 0, :, :]
        kernels = out[:, 1:2, :, :]
        embs = out[:, 2:, :, :]

        # text loss
        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = text_loss(texts, gt_texts, selected_masks, False, loss_text_weight)
        iou_text = iou((texts > 0).long(), gt_texts, training_masks, reduce=False)
        losses = dict(
            loss_text=loss_text,
            iou_text=iou_text
        )

        # kernel loss
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = kernel_loss(kernel_i, gt_kernel_i, selected_masks, False, loss_kernel_weight)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou(
            (kernels[:, -1, :, :] > 0).long(), gt_kernels[:, -1, :, :], training_masks * gt_texts, reduce=False)
        losses.update(dict(
            loss_kernels=loss_kernels,
            iou_kernel=iou_kernel
        ))

        # embedding loss
        loss_emb = emb_loss(embs, gt_instances, gt_kernels[:, -1, :, :], training_masks, gt_bboxes, False, loss_emb_weight)
        losses.update(dict(
            loss_emb=loss_emb
        ))

        return losses