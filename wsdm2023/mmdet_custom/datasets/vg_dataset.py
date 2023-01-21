import json
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from collections import OrderedDict
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmcv.utils import print_log


@DATASETS.register_module()
class VGDataset(CustomDataset):

    CLASSES = ('target',)

    def load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data_infos = json.load(f)

        return data_infos

    def get_ann_info(self, idx):
        info = self.data_infos[idx]

        bboxes = [info['bbox']]
        labels = [0]
        ann_info = dict(
            bboxes=np.array(bboxes).astype(np.float32),
            labels=np.array(labels).astype(np.int64))

        return ann_info

    def _filter_imgs(self, min_size=32):
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                if self.filter_empty_gt:
                    bbox = img_info.get('bbox')
                    if bbox != None and len(bbox) == 4:
                        valid_inds.append(i)
                else:
                    valid_inds.append(i)
        return valid_inds

    def evaluate(self,
                 results,
                 metric='Acc',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        metrics = metric if type(metric) == list else [metric]
        eval_results = OrderedDict()

        allowed_metrics = ['Acc', 'IoU']
        ious = None
        for m in metrics:
            if m not in allowed_metrics:
                raise KeyError(f'metric {m} is not supported')
            msg = f'Evaluating {m}...'
            if logger is None:
                msg = '\n'+msg
            print_log(msg, logger=logger)

            if m == 'IoU':
                if ious == None:
                    ious = self.eval_ious(results)
                eval_results['mIoU'] = np.mean(ious)
            elif m == 'Acc':
                if ious is None:
                    ious = self.eval_ious(results)
                valid_preds = np.sum(ious >= iou_thr)
                eval_results['mAcc'] = valid_preds/len(results)

        return eval_results

    def eval_ious(self, results):
        gt_bboxes = []
        for info in self.data_infos:
            gt = np.array([info['bbox']], dtype=np.float32)
            gt_bboxes.append(gt)
        assert len(results) == len(
            gt_bboxes), f'Num of pred_bboxes {len(results)} not same with gt_bboxes {len(gt_bboxes)}.'

        all_ious = []
        for pred, gt in zip(results, gt_bboxes):
            pred = pred[0]
            if pred.shape[0] == 0:
                ious = np.zeros([1, 1])
            else:
                if pred.ndim == 2 and pred.shape[1] == 5:
                    scores = pred[:, 4]
                    sort_idx = np.argsort(scores)[-1]
                    img_proposal = pred[sort_idx, :4]
                else:
                    img_proposal = pred[:4]
                if img_proposal.ndim == 1:
                    img_proposal = img_proposal.reshape(
                        1, img_proposal.shape[0])
                ious = bbox_overlaps(
                    img_proposal, gt, use_legacy_coordinate=False)  # (n, 1)
            if ious.ndim == 2:
                ious = ious.max(axis=0)
            all_ious.append(ious.item())

        return np.array(all_ious)
