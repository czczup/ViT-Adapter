# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.detr import DETR
from mmdet.core import (bbox2result, bbox_cxcywh_to_xyxy,
                        bbox_xyxy_to_cxcywh, bbox_flip)
from mmdet.core.bbox.iou_calculators import BboxOverlaps2D
import torch
from mmcv.runner import auto_fp16
from mmseg.models.decode_heads import FPNHead
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, reduce=True):
        batch_size = input.size(0)
        input = torch.sigmoid(input)

        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = self.loss_weight * loss

        if reduce:
            loss = torch.mean(loss)

        return loss


@DETECTORS.register_module()
class GroundingDINO(DETR):

    def __init__(self, with_aux_loss=False, mul_aux_seg=False, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)
        self.iou_calculator = BboxOverlaps2D()
        self.with_aux_loss = with_aux_loss
        self.mul_aux_seg = mul_aux_seg
        
        if self.with_aux_loss:
            self.aux_seg_head = FPNHead(
                in_channels=[256, 256, 256],
                in_index=[0, 1, 2],
                feature_strides=[8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                num_classes=1,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
            )
            self.loss_aux = DiceLoss(loss_weight=1.0)

    def forward_dummy(self, img):
        raise NotImplementedError

    def onnx_export(self, img, img_metas):
        raise NotImplementedError

    def extract_feat(self, img, refer, r_mask):
        x = self.backbone(img, refer, r_mask)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def extract_feats(self, imgs, refers, r_masks):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img, refer, r_mask) for img, refer, r_mask in zip(imgs, refers, r_masks)]
    
    def forward_train(self,
                      img,
                      img_metas,
                      refer,
                      r_mask,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        x = self.extract_feat(img, refer, r_mask)
        
        losses = {}
        if hasattr(self, "aux_seg_head"):
            b, _, h, w = x[0].shape
            gt_masks = torch.zeros(b, 1, h, w, device=x[0].device)
            for index, bbox in enumerate(gt_bboxes):
                x1, y1, x2, y2 = (bbox / 8).int()[0]
                gt_masks[index, :, y1:y2, x1:x2] = 1
            seg = self.aux_seg_head(x)
            aux_loss = self.loss_aux(seg, gt_masks)
            losses.update(aux_loss=aux_loss)
            
            if self.mul_aux_seg:
                seg_8s = torch.sigmoid(seg) # [b, 1, h, w]
                x = list(x)
                seg_16s = F.interpolate(seg_8s, size=x[1].shape[-2:], mode="nearest")
                seg_32s = F.interpolate(seg_16s, size=x[2].shape[-2:], mode="nearest")
                x[0] = x[0] * seg_8s
                x[1] = x[1] * seg_16s
                x[2] = x[2] * seg_32s

        loss_head = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        losses.update(loss_head)
        
        return losses

    def simple_test(self, img, img_metas, refer, r_mask, rescale=False):
        feat = self.extract_feat(img, refer, r_mask)

        # if hasattr(self, "aux_seg_head"):
        #     seg = self.aux_seg_head(feat)
        
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    @auto_fp16(apply_to=('img', 'refer', 'r_mask'))
    def forward(self, img, img_metas, refer, r_mask, return_loss=True, **kwargs):
        if torch.onnx.is_in_onnx_export():
            raise NotImplementedError

        if return_loss:
            return self.forward_train(img, img_metas, refer, r_mask, **kwargs)
        else:
            return self.forward_test(img, img_metas, refer, r_mask, **kwargs)

    def forward_test(self, imgs, img_metas, refers, r_masks, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas'), (refers, 'refers'), (r_masks, 'r_masks')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], refers[0], r_masks[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, refers, r_masks, **kwargs)

    def aug_test(self, imgs, img_metas, refers, r_masks, rescale=False):
        return [self.aug_test_vote(imgs, img_metas, refers, r_masks, rescale)]

    def rescale_boxes(self, det_bboxes, det_scores, img_meta):
        det_scores = det_scores.sigmoid()  # [900, 80]
        scores, indexes = det_scores.view(-1).topk(self.test_cfg.max_per_img)
        bbox_index = indexes // self.bbox_head.num_classes
        det_labels = indexes % self.bbox_head.num_classes
        det_bboxes = det_bboxes[bbox_index]
        det_scores = det_scores[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(det_bboxes)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        flip = img_meta[0]['flip']
        flip_direction = img_meta[0]['flip_direction']

        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * \
            img_shape[1]  # to image-scale
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        det_bboxes = bbox_flip(det_bboxes, img_shape,
                               flip_direction) if flip else det_bboxes
        # to object-scale
        det_bboxes = det_bboxes.view(-1, 4) / \
            det_bboxes.new_tensor(scale_factor)
        return det_bboxes, scores, det_labels

    def scale_boxes(self, det_bboxes, img_meta, normalize):
        scale_factor = img_meta[0]['scale_factor']
        img_shape = img_meta[0]['img_shape']

        det_bboxes = det_bboxes.view(-1, 4) * \
            det_bboxes.new_tensor(scale_factor)

        if normalize:
            det_bboxes[:, 0::2] = det_bboxes[:, 0::2] / img_shape[1]
            det_bboxes[:, 1::2] = det_bboxes[:, 1::2] / img_shape[0]
            det_bboxes = bbox_xyxy_to_cxcywh(det_bboxes)

        return det_bboxes

    def aug_test_vote(self, imgs, img_metas, refers, r_masks, rescale=False):
        feats = self.extract_feats(imgs, refers, r_masks)

        aug_bboxes, aug_scores, aug_labels = [], [], []

        for i, (feat, img_meta) in enumerate(zip(feats, img_metas)):
            det_bboxes, det_logits = self.bbox_head.tta_test_bboxes(
                feat, img_meta, rescale=True)  # [1, 900, 4] & [1, 900, 80]
            # cxcywh, [0-1]
            det_bboxes = det_bboxes[0]  # [900, 4]
            det_logits = det_logits[0]  # [900, 80]
            det_bboxes, det_scores, det_labels = self.rescale_boxes(
                det_bboxes, det_logits, img_meta)

            aug_bboxes.append(det_bboxes)  # [n, 4]
            aug_scores.append(det_scores)  # [n]
            aug_labels.append(det_labels)  # [n]
        aug_bboxes = torch.cat(aug_bboxes, dim=0)
        aug_scores = torch.cat(aug_scores, dim=0)
        aug_labels = torch.cat(aug_labels, dim=0)

        iou = self.iou_calculator(aug_bboxes, aug_bboxes).mean(1)
        # aug_scores = aug_scores + 2 * iou # 77.5
        aug_scores = aug_scores + iou # 77.4


        max_index = torch.argmax(aug_scores).item()
        aug_bboxes = aug_bboxes[max_index].unsqueeze(0)
        aug_scores = aug_scores[max_index].unsqueeze(0)
        aug_labels = aug_labels[max_index].unsqueeze(0)
        
        out_bboxes = torch.cat((aug_bboxes, aug_scores.unsqueeze(1)), -1)  # [300, 5]
        bbox_results = bbox2result(
            out_bboxes, aug_labels, self.bbox_head.num_classes)
        return bbox_results
